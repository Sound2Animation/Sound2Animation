"""Blender visualization for diffusion-generated trajectories"""
import bpy
import sys
from pathlib import Path
from mathutils import Vector, Quaternion
import numpy as np


def rotation_6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert rotation 6D representation to rotation matrices with column vectors."""
    a1, a2 = rot6d[:, :3], rot6d[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def rotation_matrix_to_quaternion(rot_mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to quaternions (wxyz)."""
    trace = rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2]
    quats = np.zeros((len(rot_mat), 4), dtype=np.float32)
    for i, (mat, tr) in enumerate(zip(rot_mat, trace)):
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2.0
            quats[i] = [
                0.25 * s,
                (mat[2, 1] - mat[1, 2]) / s,
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[1, 0] - mat[0, 1]) / s,
            ]
        elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
            quats[i] = [
                (mat[2, 1] - mat[1, 2]) / s,
                0.25 * s,
                (mat[0, 1] + mat[1, 0]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
            ]
        elif mat[1, 1] > mat[2, 2]:
            s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
            quats[i] = [
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[0, 1] + mat[1, 0]) / s,
                0.25 * s,
                (mat[1, 2] + mat[2, 1]) / s,
            ]
        else:
            s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
            quats[i] = [
                (mat[1, 0] - mat[0, 1]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
                (mat[1, 2] + mat[2, 1]) / s,
                0.25 * s,
            ]
    return quats


def mdm_to_positions_quats(mdm: np.ndarray, fps: float = 120.0):
    """Convert 11-D MDM trajectory to positions and quaternions."""
    if mdm.shape[1] != 11:
        raise ValueError("Expected [N,11] MDM trajectory (rot_vel_z, lin_vel_xz, height, rot6d, contact)")

    lin_vel = mdm[:, 1:3]
    height = mdm[:, 3]
    rot6d = mdm[:, 4:10]
    rot_mat = rotation_6d_to_matrix(rot6d)
    quats = rotation_matrix_to_quaternion(rot_mat)

    n = mdm.shape[0]
    positions = np.zeros((n, 3), dtype=np.float32)
    positions[:, 2] = height

    for i in range(1, n):
        local_v = np.array([lin_vel[i, 0], lin_vel[i, 1], 0.0], dtype=np.float32)
        world_v = rot_mat[i] @ local_v
        positions[i, :2] = positions[i - 1, :2] + world_v[:2] / fps

    return positions, quats


def load_animation_from_npy(filepath: str, fps: float = 120.0) -> list[dict]:
    """Load animation from MDM format .npy file [N, 11]."""
    mdm = np.load(filepath)
    positions, quats = mdm_to_positions_quats(mdm, fps)
    frames = []
    for i in range(len(mdm)):
        frames.append({
            "time": i / fps,
            "pos": tuple(positions[i]),
            "quat": tuple(quats[i])
        })
    return frames


def load_animation_absolute(filepath: str) -> list[dict]:
    """Load animation with absolute positions (w,x,y,z quaternion)"""
    frames = []
    with open(filepath) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                t, x, y, z, qw, qx, qy, qz = map(float, parts[:8])
                frames.append({
                    "time": t,
                    "pos": (x, y, z),
                    "quat": (qw, qx, qy, qz)
                })
    return frames


def load_animation(filepath: str) -> list[dict]:
    """Load animation from .txt or .npy file"""
    if filepath.endswith('.npy'):
        return load_animation_from_npy(filepath)
    return load_animation_absolute(filepath)


def apply_animation_absolute(obj, frames: list[dict], fps: int = 120):
    """Apply animation with absolute values"""
    for i, frame in enumerate(frames):
        frame_num = int(frame["time"] * fps)
        obj.location = Vector(frame["pos"])
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = Quaternion(frame["quat"])
        obj.keyframe_insert(data_path="location", frame=frame_num)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)


def create_scene_with_mesh(obj_path: str = None, scale: float = 1.0):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Ground
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "Ground"
    ground_mat = bpy.data.materials.new(name="Ground_Mat")
    ground_mat.diffuse_color = (0.3, 0.3, 0.3, 1.0)
    ground.data.materials.append(ground_mat)

    # Load or create object
    if obj_path and Path(obj_path).exists():
        bpy.ops.wm.obj_import(filepath=obj_path)
        target = bpy.context.selected_objects[0]
        target.name = "Object"
        target.scale = (scale, scale, scale)
        # Center geometry
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(0, 0, 0.5))
        target = bpy.context.object
        target.name = "Object"

    target_mat = bpy.data.materials.new(name="Object_Mat")
    target_mat.diffuse_color = (0.8, 0.4, 0.2, 1.0)
    target.data.materials.append(target_mat)

    # Camera
    bpy.ops.object.camera_add(location=(2, -2, 1.5))
    camera = bpy.context.object
    camera.rotation_euler = (1.1, 0, 0.8)
    bpy.context.scene.camera = camera

    # Light
    bpy.ops.object.light_add(type='SUN', location=(2, -2, 3))
    light = bpy.context.object
    light.data.energy = 3

    return target


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    anim_file = argv[0] if len(argv) > 0 else "ml/output/diffusion_anim.txt"
    output_blend = argv[1] if len(argv) > 1 else "ml/output/diffusion.blend"
    obj_file = argv[2] if len(argv) > 2 else None
    scale = float(argv[3]) if len(argv) > 3 else 1.0

    print(f"Loading animation: {anim_file}")
    print(f"Mesh file: {obj_file}")
    frames = load_animation(anim_file)
    print(f"Frames: {len(frames)}")

    target = create_scene_with_mesh(obj_file, scale)
    apply_animation_absolute(target, frames)

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(frames)
    bpy.context.scene.render.fps = 60

    Path(output_blend).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_blend)
    print(f"Saved: {output_blend}")


if __name__ == "__main__":
    main()
