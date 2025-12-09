"""Data generation using Newton physics and RealImpact audio"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from realimpact_sim import NewtonRigidBodySim
from core.realimpact_loader import RealImpactObject
from ml.diffusion.utils import quaternion_to_rotation_6d

# MDM-style encoding: rot_vel_z(1) + lin_vel_xz(2) + height(1) + rot6d(6) + contact(1)
TRAJ_DIM = 11


def qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply quaternions (w,x,y,z)"""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def qinv(q: np.ndarray) -> np.ndarray:
    """Inverse quaternion (w,x,y,z)"""
    return q * np.array([1, -1, -1, -1])


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], axis=-1),
        np.stack([2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)], axis=-1),
        np.stack([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)], axis=-1),
    ], axis=-2)


def compute_mdm_trajectory(positions: np.ndarray, quaternions: np.ndarray,
                           contacts: np.ndarray, fps: float = 120) -> np.ndarray:
    """Convert absolute trajectory to MDM-style representation

    Args:
        positions: [N, 3] absolute positions
        quaternions: [N, 4] quaternions (w,x,y,z)
        contacts: [N] contact flags
        fps: frames per second

    Returns:
        [N, 11] MDM trajectory:
            [0]    rot_velocity_z - Z-axis angular velocity
            [1:3]  linear_velocity_xz - XZ velocity in local frame
            [3]    height - absolute Z height
            [4:10] rot6d - 6D rotation
            [10]   contact - contact state
    """
    n_frames = len(positions)

    # 1. Height (absolute Z)
    height = positions[:, 2:3]

    # 2. Linear velocity in world frame (XZ plane)
    # Use world coordinates so free-fall has zero horizontal velocity
    velocity = np.diff(positions, axis=0, prepend=positions[:1]) * fps
    linear_velocity_xz = velocity[:, [0, 1]]  # World X, Y (horizontal plane)

    # 3. Angular velocity around Z-axis
    q_diff = qmul(quaternions[1:], qinv(quaternions[:-1]))
    q_diff = np.concatenate([np.array([[1, 0, 0, 0]]), q_diff], axis=0)
    rot_velocity_z = np.arcsin(np.clip(q_diff[:, 3:4], -1, 1)) * 2  # z component of quat

    # 4. 6D rotation
    rot6d = quaternion_to_rotation_6d(torch.tensor(quaternions, dtype=torch.float32)).numpy()

    # 5. Contact
    contact = contacts.reshape(-1, 1)

    return np.concatenate([
        rot_velocity_z,       # [0]
        linear_velocity_xz,   # [1:3]
        height,               # [3]
        rot6d,                # [4:10]
        contact               # [10]
    ], axis=-1).astype(np.float32)


def resample_trajectory_balanced(positions: np.ndarray, quaternions: np.ndarray,
                                  contacts: np.ndarray, target_frames: int = 360,
                                  height_threshold: float = 0.05) -> tuple:
    """Resample trajectory to balance falling vs ground phases.

    Interpolates falling phase (height > threshold) to have more frames,
    downsamples ground phase to keep total frame count constant.
    """
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Slerp, Rotation

    n_frames = len(positions)
    heights = positions[:, 2]

    # Find falling frames (before first ground contact)
    falling_mask = heights > height_threshold
    if not falling_mask.any():
        # No falling phase, return as-is with resampling to target
        return _resample_to_target(positions, quaternions, contacts, target_frames)

    # Find first ground contact
    first_ground = np.where(~falling_mask)[0]
    if len(first_ground) == 0:
        first_ground_idx = n_frames
    else:
        first_ground_idx = first_ground[0]

    # Split into falling and ground phases
    fall_pos = positions[:first_ground_idx]
    fall_quat = quaternions[:first_ground_idx]
    fall_contact = contacts[:first_ground_idx]

    ground_pos = positions[first_ground_idx:]
    ground_quat = quaternions[first_ground_idx:]
    ground_contact = contacts[first_ground_idx:]

    n_fall = len(fall_pos)
    n_ground = len(ground_pos)

    if n_fall == 0 or n_ground == 0:
        return _resample_to_target(positions, quaternions, contacts, target_frames)

    # Target: 50% falling, 50% ground (balanced distribution)
    target_fall = target_frames // 2
    target_ground = target_frames - target_fall

    # Resample falling phase (upsample)
    fall_pos_new = _interpolate_positions(fall_pos, target_fall)
    fall_quat_new = _interpolate_quaternions(fall_quat, target_fall)
    fall_contact_new = np.zeros(target_fall, dtype=np.float32)

    # Resample ground phase (downsample)
    ground_pos_new = _interpolate_positions(ground_pos, target_ground)
    ground_quat_new = _interpolate_quaternions(ground_quat, target_ground)
    ground_contact_new = np.ones(target_ground, dtype=np.float32)

    # Concatenate
    new_pos = np.concatenate([fall_pos_new, ground_pos_new], axis=0)
    new_quat = np.concatenate([fall_quat_new, ground_quat_new], axis=0)
    new_contact = np.concatenate([fall_contact_new, ground_contact_new], axis=0)

    return new_pos, new_quat, new_contact


def _interpolate_positions(pos: np.ndarray, target_n: int) -> np.ndarray:
    """Interpolate positions to target number of frames."""
    from scipy.interpolate import interp1d
    n = len(pos)
    if n == target_n:
        return pos
    t_old = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, target_n)
    interp = interp1d(t_old, pos, axis=0, kind='linear')
    return interp(t_new).astype(np.float32)


def _interpolate_quaternions(quat: np.ndarray, target_n: int) -> np.ndarray:
    """Interpolate quaternions using SLERP."""
    from scipy.spatial.transform import Slerp, Rotation
    n = len(quat)
    if n == target_n:
        return quat
    # scipy Rotation uses (x,y,z,w), we have (w,x,y,z)
    quat_xyzw = quat[:, [1, 2, 3, 0]]
    t_old = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, target_n)
    rotations = Rotation.from_quat(quat_xyzw)
    slerp = Slerp(t_old, rotations)
    new_rot = slerp(t_new)
    new_quat_xyzw = new_rot.as_quat()
    # Convert back to (w,x,y,z)
    return new_quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)


def _resample_to_target(pos, quat, contact, target_n):
    """Simple resampling to target frames."""
    new_pos = _interpolate_positions(pos, target_n)
    new_quat = _interpolate_quaternions(quat, target_n)
    # For contact, use nearest neighbor
    n = len(contact)
    indices = np.round(np.linspace(0, n-1, target_n)).astype(int)
    new_contact = contact[indices]
    return new_pos, new_quat, new_contact


def compute_dataset_stats(dataset: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std for Z-normalization"""
    all_trajs = torch.stack([s['trajectory'] for s in dataset])
    mean = all_trajs.mean(dim=(0, 1))
    std = all_trajs.std(dim=(0, 1))
    std = torch.clamp(std, min=1e-6)
    # Contact channel (last dim) stays as raw logits, so override stats to avoid normalization
    contact_dim = TRAJ_DIM - 1
    mean[contact_dim] = 0.0
    std[contact_dim] = 1.0
    return mean, std


def generate_training_sample(
    obj: RealImpactObject,
    drop_height: float = None,
    duration: float = 3.0,
    device: str = "cuda",
    balanced_resample: bool = True,
    target_frames: int = 360,
) -> tuple[torch.Tensor, np.ndarray, list]:
    """Generate a training sample using Newton physics

    Args:
        balanced_resample: If True, resample to balance falling/ground phases (50/50)

    Returns:
        mdm_trajectory: [N, 11] MDM format (rot_vel_z + lin_vel_xz + height + rot6d + contact)
        audio: [n_samples] float32
        collision_events: list of collision dicts
    """
    if drop_height is None:
        drop_height = np.random.uniform(0.3, 1.5)

    sim = NewtonRigidBodySim(obj, drop_height=drop_height, device=device, viewer=None)
    n_frames = int(duration * 120)
    for _ in range(n_frames):
        sim.step()

    # Extract data from animation_frames
    positions = np.array([f['position'] for f in sim.animation_frames], dtype=np.float32)
    quaternions = np.array([f['quaternion'] for f in sim.animation_frames], dtype=np.float32)
    contacts = np.array([float(f['contact']) for f in sim.animation_frames], dtype=np.float32)

    # Balance falling/ground phases if requested
    if balanced_resample:
        positions, quaternions, contacts = resample_trajectory_balanced(
            positions, quaternions, contacts, target_frames=target_frames
        )

    # Convert to MDM-style trajectory [N, 11]
    mdm_traj = compute_mdm_trajectory(positions, quaternions, contacts)
    mdm_trajectory = torch.tensor(mdm_traj, dtype=torch.float32)

    # Synthesize audio using sim's method (uses contact_point for accurate hit position)
    audio = sim.synthesize_audio(duration)

    return mdm_trajectory, audio, sim.collision_events


def _generate_one_sample(args):
    """Worker function for parallel generation"""
    obj, sample_idx, duration, device = args
    import torchaudio

    try:
        drop_height = np.random.uniform(0.3, 1.5)
        traj, audio, _ = generate_training_sample(obj, drop_height, duration, device)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=48000, n_fft=2048, hop_length=512, n_mels=128
        )
        audio_tensor = torch.from_numpy(audio).float()
        mel = mel_transform(audio_tensor)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

        return {
            'object_name': obj.name,
            'trajectory': traj,
            'mel': mel_db,
            'drop_height': drop_height,
        }
    except Exception as e:
        print(f"Error generating {obj.name} sample {sample_idx}: {e}")
        return None


def generate_batch_samples(
    objects: list[RealImpactObject],
    batch_size: int = 10,
    duration: float = 3.0,
    device: str = "cuda",
) -> list[dict]:
    """Generate samples using batch simulation (multiple objects at once)"""
    from realimpact_sim import BatchRigidBodySim
    import torchaudio

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=48000, n_fft=2048, hop_length=512, n_mels=128
    )

    drop_heights = [np.random.uniform(0.3, 1.5) for _ in objects]
    sim = BatchRigidBodySim(objects, drop_heights, spacing=3.0, device=device)

    n_frames = int(duration * 120)
    for _ in range(n_frames):
        sim.step()

    # Collect simulation results
    sim_results = [sim.get_results(i) for i in range(len(objects))]

    def process_sample(args):
        i, obj, (frames, collision_events) = args
        positions = np.array([f['position'] for f in frames], dtype=np.float32)
        quaternions = np.array([f['quaternion'] for f in frames], dtype=np.float32)
        contacts = np.array([float(f['contact']) for f in frames], dtype=np.float32)

        mdm_traj = compute_mdm_trajectory(positions, quaternions, contacts)
        mdm_trajectory = torch.tensor(mdm_traj, dtype=torch.float32)

        from core.sound_synthesizer import SoundSynthesizer
        synth = SoundSynthesizer(sample_rate=obj.sample_rate)
        synth.init_recording(duration + 1.0)
        camera_pos = np.array([0.5, 0.25, 0.5], dtype=np.float32)
        for event in collision_events:
            audio, _ = obj.query_sound(event['contact_point'], camera_pos)
            if audio is not None:
                synth.trigger_sound(audio, event['impulse'], time_offset=event['time'])
        audio = synth.get_recording()[:int(obj.sample_rate * duration)]

        audio_tensor = torch.from_numpy(audio).float()
        mel = mel_transform(audio_tensor)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

        return {
            'object_name': obj.name,
            'trajectory': mdm_trajectory,
            'mel': mel_db,
            'drop_height': drop_heights[i],
        }

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_sample, [(i, objects[i], sim_results[i]) for i in range(len(objects))]))

    return results


def _subprocess_worker(worker_id, object_indices, args_dict, output_path):
    """Subprocess worker - runs in separate process with its own CUDA context"""
    import subprocess
    import sys

    cmd = [
        sys.executable, __file__,
        "--dataset", args_dict["dataset"],
        "--samples-per-object", str(args_dict["samples_per_object"]),
        "--duration", str(args_dict["duration"]),
        "--output", output_path,
        "--batch-size", str(args_dict["batch_size"]),
        "--start-obj", str(object_indices[0]),
        "--end-obj", str(object_indices[1]),
    ]
    if args_dict.get("material_filter"):
        cmd.extend(["--material-filter", args_dict["material_filter"]])
    if args_dict.get("object_filter"):
        cmd.extend(["--object-filter", args_dict["object_filter"]])

    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def generate_dataset_multiproc(
    n_objects: int,
    num_workers: int,
    args_dict: dict,
    output_path: str,
):
    """Spawn multiple subprocesses and merge results"""
    import time
    import select
    import tempfile
    import os

    # Split objects across workers
    objects_per_worker = n_objects // num_workers
    remainder = n_objects % num_workers

    tmp_dir = tempfile.mkdtemp(prefix="dataset_")
    processes = []
    tmp_files = []

    start_idx = 0
    for i in range(num_workers):
        count = objects_per_worker + (1 if i < remainder else 0)
        if count == 0:
            continue
        end_idx = start_idx + count
        tmp_file = os.path.join(tmp_dir, f"part_{i}.pt")
        tmp_files.append(tmp_file)

        proc = _subprocess_worker(i, (start_idx, end_idx), args_dict, tmp_file)
        processes.append((i, proc, start_idx, end_idx))
        print(f"[Main] Started worker {i}: objects {start_idx}-{end_idx}")
        start_idx = end_idx

    # Monitor all processes
    while processes:
        for i, proc, s, e in processes[:]:
            line = proc.stdout.readline()
            if line:
                print(f"[W{i}] {line.rstrip()}")
            if proc.poll() is not None:
                # Process finished
                for line in proc.stdout:
                    print(f"[W{i}] {line.rstrip()}")
                processes.remove((i, proc, s, e))
                print(f"[Main] Worker {i} finished (exit={proc.returncode})")
        time.sleep(0.1)

    # Merge results
    print(f"\n[Main] Merging {len(tmp_files)} partial datasets...")
    dataset = []
    for tmp_file in tmp_files:
        if os.path.exists(tmp_file):
            part = torch.load(tmp_file)
            dataset.extend(part)
            os.remove(tmp_file)
    os.rmdir(tmp_dir)

    torch.save(dataset, output_path)
    print(f"[Main] Saved {len(dataset)} samples to {output_path}")
    return dataset


def generate_dataset(
    objects: list[RealImpactObject],
    samples_per_object: int = 20,
    duration: float = 3.0,
    output_path: str = "ml/dataset.pt",
    device: str = "cuda",
    batch_size: int = 1,
    num_workers: int = 1,
):
    """Generate and save a training dataset

    Args:
        batch_size: Number of samples per batch (uses BatchRigidBodySim)
        num_workers: Number of parallel processes (spawns separate CUDA contexts)
    """
    import torchaudio
    import time

    total = len(objects) * samples_per_object
    print(f"Generating {total} samples ({len(objects)} objects × {samples_per_object} samples)")
    print(f"Device: {device}, Batch size: {batch_size}, Workers: {num_workers}")

    if batch_size > 1:
        # Single process batch mode
        dataset = []
        for obj_idx, obj in enumerate(objects):
            print(f"\n[{obj_idx+1}/{len(objects)}] {obj.name}")
            obj_start = time.time()
            for batch_start in range(0, samples_per_object, batch_size):
                batch_end = min(batch_start + batch_size, samples_per_object)
                batch_objs = [obj] * (batch_end - batch_start)
                try:
                    results = generate_batch_samples(batch_objs, batch_size, duration, device)
                    dataset.extend(results)
                    print(f"  {batch_end}/{samples_per_object} done")
                except Exception as e:
                    print(f"  Error batch {batch_start}-{batch_end}: {e}")
            print(f"  Completed in {time.time()-obj_start:.1f}s")
    else:
        # Sequential mode
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=48000, n_fft=2048, hop_length=512, n_mels=128
        )
        dataset = []
        for obj_idx, obj in enumerate(objects):
            print(f"\n[{obj_idx+1}/{len(objects)}] {obj.name}")
            obj_start = time.time()
            for sample_idx in range(samples_per_object):
                try:
                    drop_height = np.random.uniform(0.3, 1.5)
                    traj, audio, _ = generate_training_sample(obj, drop_height, duration, device)

                    audio_tensor = torch.from_numpy(audio).float()
                    mel = mel_transform(audio_tensor)
                    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

                    dataset.append({
                        'object_name': obj.name,
                        'trajectory': traj,
                        'mel': mel_db,
                        'drop_height': drop_height,
                    })
                    if (sample_idx + 1) % 10 == 0:
                        print(f"  {sample_idx+1}/{samples_per_object} done")
                except Exception as e:
                    print(f"  Error sample {sample_idx}: {e}")
            print(f"  Completed in {time.time()-obj_start:.1f}s")

    torch.save(dataset, output_path)
    print(f"\nSaved {len(dataset)} samples to {output_path}")
    return dataset


if __name__ == "__main__":
    import argparse
    from core.realimpact_loader import RealImpactLoader
    from core.material_properties import parse_material_from_name

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/run/media/jim_z/SRC/dev/RealImpact/dataset")
    parser.add_argument("--samples-per-object", type=int, default=20)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--output", default="ml/dataset.pt")
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--material-filter", type=str, default=None, help="Filter by material (ceramic, iron, wood, etc)")
    parser.add_argument("--object-filter", type=str, default=None, help="Filter by object name (exact match)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for parallel simulation (>1 enables BatchRigidBodySim)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("--start-obj", type=int, default=0, help="Start object index (for splitting)")
    parser.add_argument("--end-obj", type=int, default=0, help="End object index (0=all)")
    args = parser.parse_args()

    loader = RealImpactLoader(args.dataset)
    object_names = loader.list_objects()

    if args.material_filter:
        object_names = [n for n in object_names if parse_material_from_name(n)[0] == args.material_filter]
        print(f"Filtered to {len(object_names)} {args.material_filter} objects")

    if args.object_filter:
        object_names = [n for n in object_names if args.object_filter in n]
        print(f"Filtered to {len(object_names)} objects matching '{args.object_filter}'")

    if args.max_objects > 0:
        object_names = object_names[:args.max_objects]

    # Support splitting objects across processes
    if args.end_obj > 0:
        object_names = object_names[args.start_obj:args.end_obj]
    elif args.start_obj > 0:
        object_names = object_names[args.start_obj:]

    # Multi-process mode: spawn subprocesses and merge (don't load objects in main process)
    if args.workers > 1 and args.start_obj == 0 and args.end_obj == 0:
        print(f"Starting {args.workers} worker processes...")
        args_dict = {
            "dataset": args.dataset,
            "samples_per_object": args.samples_per_object,
            "duration": args.duration,
            "batch_size": args.batch_size,
            "material_filter": args.material_filter,
            "object_filter": args.object_filter,
        }
        generate_dataset_multiproc(len(object_names), args.workers, args_dict, args.output)
    else:
        # Single process or subprocess: load objects and generate
        print(f"Loading {len(object_names)} objects...")
        objects = []
        for name in object_names:
            try:
                obj = loader.load_object(name)
                objects.append(obj)
                print(f"  Loaded {name}")
            except Exception as e:
                print(f"  Failed to load {name}: {e}")

        print(f"Generating dataset with {len(objects)} objects, {args.samples_per_object} samples each")
        generate_dataset(objects, args.samples_per_object, args.duration, args.output,
                         batch_size=args.batch_size)
