"""Sound2Motion - Diffusion Training with Newton Physics"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.realimpact_loader import RealImpactLoader
from core.material_properties import parse_material_from_name
from torch_geometric.data import Batch
from ml.encoders import AudioEncoder, TextEncoder, MeshGNN
from ml.diffusion import TrajectoryDiffusion, DDPMScheduler
from ml.diffusion.model import ConditionEncoder
from ml.data_generator import TRAJ_DIM, generate_training_sample, compute_dataset_stats
from ml.trajectory_utils import normalize_trajectory, denormalize_trajectory


class Sound2MotionV2:
    def __init__(self, device: str = "cuda", d_model: int = 256):
        self.device = device
        self.audio_encoder = AudioEncoder(d_model=d_model).to(device)
        self.text_encoder = TextEncoder(d_model=d_model, device=device).to(device)
        self.mesh_gnn = MeshGNN(d_model=d_model).to(device)
        self.condition_encoder = ConditionEncoder(d_model=d_model).to(device)
        self.diffusion = TrajectoryDiffusion(traj_dim=TRAJ_DIM, d_model=d_model).to(device)
        self.scheduler = DDPMScheduler(num_timesteps=1000).to(device)

    def parameters(self):
        return (list(self.audio_encoder.parameters()) +
                list(self.mesh_gnn.parameters()) +
                list(self.condition_encoder.parameters()) +
                list(self.diffusion.parameters()))

    def train(self):
        for m in [self.audio_encoder, self.mesh_gnn, self.condition_encoder, self.diffusion]:
            m.train()

    def eval(self):
        for m in [self.audio_encoder, self.mesh_gnn, self.condition_encoder, self.diffusion]:
            m.eval()


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Trajectory dim: {TRAJ_DIM} (rot_vel_z:1 + lin_vel_xz:2 + height:1 + rot6d:6 + contact:1)")

    # Load pre-generated dataset if specified
    fixed_dataset = None
    mean, std = None, None
    if args.dataset_file:
        print(f"Loading pre-generated dataset: {args.dataset_file}")
        fixed_dataset = torch.load(args.dataset_file, weights_only=False)
        print(f"Loaded {len(fixed_dataset)} samples from file")

        # Compute or load Z-normalization stats
        stats_file = Path(args.dataset_file).with_suffix('.stats.pt')
        if stats_file.exists():
            stats = torch.load(stats_file)
            mean, std = stats['mean'].to(device), stats['std'].to(device)
            print(f"Loaded stats from {stats_file}")
        else:
            print("Computing dataset statistics...")
            mean, std = compute_dataset_stats(fixed_dataset)
            torch.save({'mean': mean, 'std': std}, stats_file)
            mean, std = mean.to(device), std.to(device)
            print(f"Saved stats to {stats_file}")
        print(f"  Mean: {mean.cpu().numpy()}")
        print(f"  Std:  {std.cpu().numpy()}")

    loader = RealImpactLoader(args.dataset)

    # When using fixed dataset, load only the objects present in dataset
    if fixed_dataset:
        object_names = list(set(s['object_name'] for s in fixed_dataset))
        print(f"Dataset contains {len(object_names)} unique objects: {object_names}")
    else:
        object_names = loader.list_objects()
        print(f"Found {len(object_names)} objects in RealImpact dataset")
        if len(object_names) == 0:
            print("No objects found! Check dataset path.")
            return
        if args.max_objects > 0:
            object_names = object_names[:args.max_objects]
        print(f"Using {len(object_names)} objects: {object_names[:5]}...")

    # Load objects with audio (for mesh data and text prompts)
    objects = []
    for name in object_names:
        try:
            obj = loader.load_object(name, load_audio=not fixed_dataset)
            material, _ = parse_material_from_name(name)
            if args.material_filter and material.lower() != args.material_filter.lower():
                continue
            if args.object_filter and name != args.object_filter:
                continue
            objects.append({"name": name, "material": material, "obj": obj})
        except Exception as e:
            print(f"  Skip {name}: {e}")

    print(f"Loaded {len(objects)} objects")

    model = Sound2MotionV2(device=device, d_model=args.d_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=48000, n_fft=2048, hop_length=512, n_mels=128
    )

    batch_size = args.batch_size
    if fixed_dataset:
        steps_per_epoch = len(fixed_dataset) // batch_size
    else:
        steps_per_epoch = args.samples_per_epoch // batch_size
    print(f"\nTraining for {args.epochs} epochs, {steps_per_epoch} steps/epoch (batch_size={batch_size})")

    # Build object lookup for fixed dataset
    obj_lookup = {o["name"]: o for o in objects}

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        # Shuffle fixed dataset indices each epoch
        if fixed_dataset:
            indices = np.random.permutation(len(fixed_dataset))

        for step in range(steps_per_epoch):
            batch_mels, batch_texts, batch_graphs, batch_trajs = [], [], [], []

            for b in range(batch_size):
                if fixed_dataset:
                    # Load from pre-generated dataset
                    idx = indices[step * batch_size + b]
                    sample = fixed_dataset[idx]
                    obj_name = sample['object_name']
                    obj_data = obj_lookup.get(obj_name)
                    if obj_data is None:
                        continue
                    obj = obj_data["obj"]
                    mdm_traj = sample['trajectory']
                    mel = sample['mel']
                    drop_height = sample['drop_height']
                else:
                    # Generate on-the-fly
                    obj_data = objects[np.random.randint(len(objects))]
                    obj = obj_data["obj"]
                    drop_height = np.random.uniform(0.3, 0.8)
                    mdm_traj, audio, _ = generate_training_sample(obj, drop_height, device=device)
                    waveform = torch.from_numpy(audio).unsqueeze(0)
                    mel = mel_transform(waveform).squeeze(0)
                    mel = torch.log(mel + 1e-9)

                batch_mels.append(mel)
                text = f"A {obj_data['material']} {obj_data['name'].split('_')[-1].lower()} drops from {drop_height:.1f}m"
                batch_texts.append(text)

                verts, faces = obj.vertices, obj.faces
                if len(verts) > 500:
                    idx = np.random.choice(len(verts), 500, replace=False)
                    verts = verts[idx]
                    faces = np.random.randint(0, 500, (200, 3))
                batch_graphs.append(MeshGNN.mesh_to_graph(verts, faces, device))

                batch_trajs.append(mdm_traj)

            if len(batch_mels) == 0:
                continue

            mels = torch.stack(batch_mels, dim=0).to(device)
            trajs = torch.stack(batch_trajs, dim=0).to(device)
            batched_graph = Batch.from_data_list(batch_graphs)

            audio_feat = model.audio_encoder(mels)
            text_feat = model.text_encoder(batch_texts)
            mesh_feat = model.mesh_gnn(batched_graph.x, batched_graph.edge_index, batched_graph.batch)

            n_frames = trajs.shape[1]
            uncond = np.random.random() < 0.1
            audio_aligned, text_out, mesh_out = model.condition_encoder(
                audio_feat, text_feat, mesh_feat, n_frames, uncond=uncond
            )

            # Z-normalize trajectories if stats available
            trajs_norm = normalize_trajectory(trajs, mean, std) if mean is not None else trajs

            actual_batch = trajs.shape[0]
            t = torch.randint(0, model.scheduler.num_timesteps, (actual_batch,), device=device)
            noise = torch.randn_like(trajs_norm)
            x_t = model.scheduler.add_noise(trajs_norm, noise, t)
            pred_x0 = model.diffusion(x_t, t, audio_aligned, text_out, mesh_out)

            # Position loss (all dims except contact)
            pos_loss = torch.nn.functional.mse_loss(pred_x0[..., :10], trajs_norm[..., :10])
            # Height loss - separate term since lin_vel_xz doesn't contain vertical velocity
            height_loss = torch.nn.functional.mse_loss(pred_x0[..., 3], trajs_norm[..., 3])
            # Rotation loss - rot6d (dims 4:10)
            rot_loss = torch.nn.functional.mse_loss(pred_x0[..., 4:10], trajs_norm[..., 4:10])
            contact_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_x0[..., 10:11], trajs[..., 10:11])

            # Contact-aware velocity loss: only smooth non-contact frames
            # Contact is at dim 10, use GT contact to mask
            contact = trajs[..., 10]  # [B, T]
            # Mask: 1 for non-contact frames (allow smoothing), 0 for contact (allow sudden changes)
            non_contact_mask = (contact[:, 1:] < 0.5) & (contact[:, :-1] < 0.5)  # [B, T-1]
            non_contact_mask = non_contact_mask.unsqueeze(-1).float()  # [B, T-1, 1]

            pred_vel = pred_x0[:, 1:, :4] - pred_x0[:, :-1, :4]
            gt_vel = trajs_norm[:, 1:, :4] - trajs_norm[:, :-1, :4]
            # Masked velocity loss - only on non-contact frames
            vel_diff = (pred_vel - gt_vel) ** 2
            vel_loss = (vel_diff * non_contact_mask).sum() / (non_contact_mask.sum() * 4 + 1e-6)

            # Acceleration loss - also contact-aware
            non_contact_mask_acc = non_contact_mask[:, 1:] * non_contact_mask[:, :-1]  # [B, T-2, 1]
            pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
            gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
            acc_diff = (pred_acc - gt_acc) ** 2
            acc_loss = (acc_diff * non_contact_mask_acc).sum() / (non_contact_mask_acc.sum() * 4 + 1e-6)

            # Ground friction loss: penalize velocity when contact is detected
            # This enforces that after landing, movement should be restricted
            contact_mask = (contact[:, 1:] > 0.5).unsqueeze(-1).float()  # [B, T-1, 1]
            friction_loss = (pred_vel ** 2 * contact_mask).sum() / (contact_mask.sum() * 4 + 1e-6)

            # Total loss: position + height + rotation + velocity + acceleration + contact + friction
            loss = pos_loss + 1.0 * height_loss + 5.0 * rot_loss + 0.5 * vel_loss + 0.25 * acc_loss + 0.5 * contact_loss + 1.0 * friction_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.6f}")

        if args.wandb:
            wandb.log({
                "train/loss": avg_loss,
                "train/pos_loss": pos_loss.item(),
                "train/height_loss": height_loss.item(),
                "train/rot_loss": rot_loss.item(),
                "train/vel_loss": vel_loss.item(),
                "train/acc_loss": acc_loss.item(),
                "train/contact_loss": contact_loss.item(),
                "train/friction_loss": friction_loss.item(),
                "epoch": epoch + 1
            })

        if (epoch + 1) % 10 == 0:
            save_path = Path(args.output_dir) / f"diffusion_epoch{epoch + 1}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "audio_encoder": model.audio_encoder.state_dict(),
                "mesh_gnn": model.mesh_gnn.state_dict(),
                "condition_encoder": model.condition_encoder.state_dict(),
                "diffusion": model.diffusion.state_dict(),
            }
            if mean is not None:
                ckpt["mean"] = mean.cpu()
                ckpt["std"] = std.cpu()
            torch.save(ckpt, save_path)
            print(f"  Saved: {save_path}")

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                obj_data = objects[0]
                obj = obj_data["obj"]

                gt_mdm, audio, _ = generate_training_sample(obj, drop_height=0.5, device=device)
                gt_mdm = gt_mdm.unsqueeze(0).to(device)

                waveform = torch.from_numpy(audio).unsqueeze(0)
                mel = mel_transform(waveform).squeeze(0)
                mel = torch.log(mel + 1e-9).unsqueeze(0).to(device)

                text = f"A {obj_data['material']} bowl drops from 0.5m"
                verts, faces = obj.vertices, obj.faces
                if len(verts) > 500:
                    idx = np.random.choice(len(verts), 500, replace=False)
                    verts, faces = verts[idx], np.random.randint(0, 500, (200, 3))
                graph = MeshGNN.mesh_to_graph(verts, faces, device)

                audio_feat = model.audio_encoder(mel)
                text_feat = model.text_encoder([text])
                mesh_feat = model.mesh_gnn(graph.x, graph.edge_index, None)

                n_frames = gt_mdm.shape[1]
                audio_cond, text_cond, mesh_cond = model.condition_encoder(
                    audio_feat, text_feat, mesh_feat, n_frames, uncond=False
                )
                audio_uncond, text_uncond, mesh_uncond = model.condition_encoder(
                    audio_feat, text_feat, mesh_feat, n_frames, uncond=True
                )

                cfg_scale = 2.5
                x = torch.randn(1, n_frames, TRAJ_DIM, device=device)
                for t in reversed(range(0, model.scheduler.num_timesteps, 10)):
                    t_tensor = torch.tensor([t], device=device)
                    x0_cond = model.diffusion(x, t_tensor, audio_cond, text_cond, mesh_cond)
                    x0_uncond = model.diffusion(x, t_tensor, audio_uncond, text_uncond, mesh_uncond)
                    pred_x0 = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
                    x = model.scheduler.step(pred_x0, t, x)

                # Denormalize predictions for evaluation
                pred_norm = x[0].cpu()
                if mean is not None:
                    mean_cpu = mean.cpu()
                    std_cpu = std.cpu()
                    pred_mdm = denormalize_trajectory(pred_norm, mean_cpu, std_cpu).numpy()
                    gt_mdm_np = gt_mdm[0].cpu().numpy()
                else:
                    pred_mdm = pred_norm.numpy()
                    gt_mdm_np = gt_mdm[0].cpu().numpy()

                val_dir = Path(args.output_dir) / "val_samples"
                val_dir.mkdir(parents=True, exist_ok=True)
                np.save(val_dir / f"gt_epoch{epoch+1}.npy", gt_mdm_np)
                np.save(val_dir / f"pred_epoch{epoch+1}.npy", pred_mdm)

                # MDM encoding: [0] rot_vel_z, [1:3] lin_vel_xz, [3] height, [4:10] rot6d, [10] contact
                vel_mse = np.mean((pred_mdm[:, :3] - gt_mdm_np[:, :3]) ** 2)
                rot_mse = np.mean((pred_mdm[:, 4:10] - gt_mdm_np[:, 4:10]) ** 2)
                pred_height = pred_mdm[:, 3]  # Height is at index 3
                print(f"  Val MSE - vel: {vel_mse:.6f}, rot: {rot_mse:.6f}, height: [{pred_height.min():.3f}, {pred_height.max():.3f}]")

                if args.wandb:
                    wandb.log({"val/vel_mse": vel_mse, "val/rot_mse": rot_mse, "val/height_min": pred_height.min(), "val/height_max": pred_height.max(), "epoch": epoch + 1})

                # Save best model
                val_loss = vel_mse + rot_mse
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = Path(args.output_dir) / "diffusion_best.pt"
                    best_ckpt = {
                        "audio_encoder": model.audio_encoder.state_dict(),
                        "mesh_gnn": model.mesh_gnn.state_dict(),
                        "condition_encoder": model.condition_encoder.state_dict(),
                        "diffusion": model.diffusion.state_dict(),
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                    }
                    if mean is not None:
                        best_ckpt["mean"] = mean.cpu()
                        best_ckpt["std"] = std.cpu()
                    torch.save(best_ckpt, best_path)
                    print(f"  New best model! val_loss={val_loss:.6f}")
            model.train()

    save_path = Path(args.output_dir) / "diffusion_final.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "audio_encoder": model.audio_encoder.state_dict(),
        "mesh_gnn": model.mesh_gnn.state_dict(),
        "condition_encoder": model.condition_encoder.state_dict(),
        "diffusion": model.diffusion.state_dict(),
    }
    if mean is not None:
        ckpt["mean"] = mean.cpu()
        ckpt["std"] = std.cpu()
    torch.save(ckpt, save_path)

    if args.wandb:
        wandb.finish()
    print(f"Training complete. Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/run/media/jim_z/SRC/dev/RealImpact/dataset/")
    parser.add_argument("--dataset-file", type=str, default=None, help="Pre-generated dataset .pt file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples-per-epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=20, help="Max objects to use (0=all)")
    parser.add_argument("--material-filter", type=str, default=None)
    parser.add_argument("--object-filter", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="ml/checkpoints_newton")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="sound2motion")
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    train(args)
