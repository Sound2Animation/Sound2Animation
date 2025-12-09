"""Inference with x0-prediction diffusion model (Newton physics)"""
import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.realimpact_loader import RealImpactLoader
from core.material_properties import parse_material_from_name
from ml.encoders import AudioEncoder, TextEncoder, MeshGNN
from ml.diffusion import TrajectoryDiffusion, DDPMScheduler
from ml.diffusion.model import ConditionEncoder
from ml.diffusion.utils import rotation_6d_to_matrix, matrix_to_quaternion
from ml.data_generator import TRAJ_DIM, generate_training_sample
from ml.trajectory_utils import denormalize_trajectory


def load_model(checkpoint_path: str, device: str = "cuda"):
    audio_encoder = AudioEncoder(d_model=256).to(device)
    text_encoder = TextEncoder(d_model=256, device=device).to(device)
    mesh_gnn = MeshGNN(d_model=256).to(device)
    condition_encoder = ConditionEncoder(d_model=256).to(device)
    diffusion = TrajectoryDiffusion(traj_dim=TRAJ_DIM, d_model=256).to(device)
    scheduler = DDPMScheduler(num_timesteps=1000).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    audio_encoder.load_state_dict(ckpt["audio_encoder"])
    mesh_gnn.load_state_dict(ckpt["mesh_gnn"])
    condition_encoder.load_state_dict(ckpt["condition_encoder"])
    diffusion.load_state_dict(ckpt["diffusion"])

    # Load normalization stats if available
    mean = ckpt.get("mean", None)
    std = ckpt.get("std", None)
    if mean is not None:
        mean = mean.to(device)
        std = std.to(device)

    return audio_encoder, text_encoder, mesh_gnn, condition_encoder, diffusion, scheduler, mean, std


def mdm_to_positions_quats(mdm: np.ndarray, fps: float = 120):
    """Convert MDM trajectory to positions and quaternions (wxyz for Blender)

    MDM encoding: [0] rot_vel_z, [1:3] lin_vel_xz (world frame), [3] height, [4:10] rot6d, [10] contact

    Recovery process:
    1. Get height directly from index 3
    2. Recover XY positions by cumulating world-frame velocities
    3. Get rotation directly from rot6d
    """
    n_frames = len(mdm)

    # Extract components
    lin_vel_xy = mdm[:, 1:3]  # World frame XY velocity
    height = mdm[:, 3]        # Absolute height (Z)
    rot6d = torch.tensor(mdm[:, 4:10], dtype=torch.float32)

    # Get rotation matrices and quaternions from rot6d
    rot_mat = rotation_6d_to_matrix(rot6d)
    quat_wxyz = matrix_to_quaternion(rot_mat).numpy()

    # Recover XY positions by cumulating world-frame velocity
    positions = np.zeros((n_frames, 3), dtype=np.float32)
    positions[:, 2] = height  # Z is the height directly

    for i in range(1, n_frames):
        # World-frame velocity - directly integrate
        positions[i, 0] = positions[i-1, 0] + lin_vel_xy[i, 0] / fps
        positions[i, 1] = positions[i-1, 1] + lin_vel_xy[i, 1] / fps

    return positions, quat_wxyz


def generate_blend(anim_file: str, blend_file: str, obj_file: str = None):
    script_path = Path(__file__).parent / "visualize_diffusion.py"
    cmd = ["blender", "--background", "--python", str(script_path), "--", str(anim_file), str(blend_file)]
    if obj_file:
        cmd.extend([str(obj_file), "1.0"])
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Generated: {blend_file}")
    except subprocess.CalledProcessError as e:
        print(f"Blender failed: {e.stderr.decode()[:200]}")
    except FileNotFoundError:
        print("Blender not found, skipping .blend generation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ml/checkpoints_newton/diffusion_epoch100.pt")
    parser.add_argument("--object", default="3_CeramicKoiBowl")
    parser.add_argument("--dataset", default="/run/media/jim_z/SRC/dev/RealImpact/dataset/")
    parser.add_argument("--drop-height", type=float, default=0.5)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=100, help="Number of diffusion sampling steps (max 1000)")
    parser.add_argument("--output-dir", default="ml/output")
    parser.add_argument("--blend", action="store_true", help="Generate .blend file")
    parser.add_argument("--text", type=str, default=None, help="Custom text prompt (default: auto-generated)")
    parser.add_argument("--audio", type=str, default=None, help="Custom audio file path (.wav)")
    parser.add_argument("--mesh", type=str, default=None, help="Custom mesh file path (.obj)")
    args = parser.parse_args()

    device = "cuda"
    print("Loading model...")
    audio_encoder, text_encoder, mesh_gnn, condition_encoder, diffusion, scheduler, mean, std = load_model(args.checkpoint, device)
    if mean is not None:
        print(f"Loaded normalization stats (mean shape: {mean.shape})")

    print("Loading object...")
    loader = RealImpactLoader(args.dataset)
    obj = loader.load_object(args.object, load_audio=True)
    material, _ = parse_material_from_name(args.object)

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=2048, hop_length=512, n_mels=128)

    # Load audio: custom file or generate from physics
    if args.audio:
        print(f"Loading custom audio: {args.audio}")
        waveform, sr = torchaudio.load(args.audio)
        if sr != 48000:
            waveform = torchaudio.functional.resample(waveform, sr, 48000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    else:
        print("Generating GT trajectory with Newton physics...")
        gt_traj, audio, events = generate_training_sample(obj, drop_height=args.drop_height, device=device)
        print(f"  Collisions: {len(events)}")
        waveform = torch.from_numpy(audio).unsqueeze(0)

    mel = mel_transform(waveform).squeeze(0)
    mel = torch.log(mel + 1e-9).unsqueeze(0).to(device)

    # Text prompt: custom or auto-generated
    text = args.text if args.text else f"A {material} bowl drops from {args.drop_height}m"
    print(f"Text prompt: {text}")

    # Mesh: custom file or from object
    if args.mesh:
        print(f"Loading custom mesh: {args.mesh}")
        import trimesh
        mesh = trimesh.load(args.mesh)
        verts, faces = np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces, dtype=np.int64)
    else:
        verts, faces = obj.vertices, obj.faces
    if len(verts) > 500:
        idx = np.random.choice(len(verts), 500, replace=False)
        verts = verts[idx]
        faces = np.random.randint(0, 500, (200, 3))
    graph = MeshGNN.mesh_to_graph(verts, faces, device)

    print("Encoding conditions...")
    audio_encoder.eval()
    mesh_gnn.eval()
    condition_encoder.eval()
    diffusion.eval()

    n_frames = 360
    with torch.no_grad():
        audio_feat = audio_encoder(mel)
        text_feat = text_encoder([text])
        mesh_feat = mesh_gnn(graph.x, graph.edge_index, None)

        audio_cond, text_cond, mesh_cond = condition_encoder(
            audio_feat, text_feat, mesh_feat, n_frames, uncond=False
        )
        audio_uncond, text_uncond, mesh_uncond = condition_encoder(
            audio_feat, text_feat, mesh_feat, n_frames, uncond=True
        )

        # Calculate step size: 1000 timesteps / desired steps
        step_size = max(1, scheduler.num_timesteps // args.steps)
        print(f"Sampling with CFG scale {args.cfg_scale}, {args.steps} steps (step_size={step_size})...")
        x = torch.randn(1, n_frames, TRAJ_DIM, device=device)

        for t in reversed(range(0, scheduler.num_timesteps, step_size)):
            t_tensor = torch.tensor([t], device=device)
            x0_cond = diffusion(x, t_tensor, audio_cond, text_cond, mesh_cond)
            x0_uncond = diffusion(x, t_tensor, audio_uncond, text_uncond, mesh_uncond)
            pred_x0 = x0_uncond + args.cfg_scale * (x0_cond - x0_uncond)
            x = scheduler.step(pred_x0, t, x)

        # Denormalize predictions
        if mean is not None:
            pred_mdm = denormalize_trajectory(x[0].cpu(), mean.cpu(), std.cpu()).numpy()
        else:
            pred_mdm = x[0].cpu().numpy()

    print("Converting trajectory...")
    positions, quaternions = mdm_to_positions_quats(pred_mdm)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    anim_file = output_dir / f"{args.object}_diffusion_anim.txt"

    with open(anim_file, 'w') as f:
        f.write(f"# ground_level: 0.0\n")
        f.write(f"# material: {material}\n")
        f.write(f"# drop_height: {args.drop_height}\n")
        for i in range(len(positions)):
            t = i / 120.0
            px, py, pz = positions[i]
            qw, qx, qy, qz = quaternions[i]
            f.write(f"{t:.6f} {px:.6f} {py:.6f} {pz:.6f} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}\n")

    print(f"Saved: {anim_file}")
    print(f"Height (Z) range: {positions[:, 2].min():.3f} to {positions[:, 2].max():.3f}")

    # MDM encoding: contact is at index 10
    contact = pred_mdm[:, 10]
    contact_frames = np.where(contact > 0.5)[0]
    print(f"Contact frames: {len(contact_frames)} (first: {contact_frames[0] if len(contact_frames) > 0 else 'none'})")

    if args.blend:
        blend_file = output_dir / f"{args.object}_diffusion.blend"
        obj_file = Path(args.dataset) / args.object / "preprocessed" / "transformed.obj"
        generate_blend(str(anim_file), str(blend_file), str(obj_file) if obj_file.exists() else None)


if __name__ == "__main__":
    main()
