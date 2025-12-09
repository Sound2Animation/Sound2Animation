"""
Blender rendering script for RealImpact simulation
Run with: blender --background --python blender_render.py -- --object 3_CeramicKoiBowl
"""

import sys
import argparse
import json
from pathlib import Path


def get_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/run/media/jim_z/SRC/dev/RealImpact/dataset')
    parser.add_argument('--object', type=str, default='3_CeramicKoiBowl')
    parser.add_argument('--animation', type=str, default=None, help='Animation file path')
    parser.add_argument('--metadata', type=str, default=None, help='Metadata JSON file path')
    parser.add_argument('--output', type=str, default='render_output')
    parser.add_argument('--fps', type=int, default=60, help='Animation FPS')
    parser.add_argument('--preview', action='store_true', help='Save .blend file only')
    parser.add_argument('--render-video', action='store_true', help='Render animation as video')
    parser.add_argument('--eevee', action='store_true', help='Use EEVEE for fast rendering')
    return parser.parse_args(argv)


def setup_scene(use_eevee=False):
    import bpy

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    if use_eevee:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
        bpy.context.scene.eevee.taa_render_samples = 32
    else:
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.samples = 128

    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.film_transparent = True


def load_object(obj_path: Path):
    import bpy

    bpy.ops.wm.obj_import(filepath=str(obj_path))
    obj = bpy.context.selected_objects[0]
    obj.name = "realimpact_object"

    # Create simple material without texture
    mat = bpy.data.materials.new(name="ObjectMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (0.8, 0.6, 0.4, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.5
    bsdf.inputs['Metallic'].default_value = 0.1

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    return obj


def setup_lighting():
    import bpy

    bpy.ops.object.light_add(type='SUN', location=(2, 2, 3))
    sun = bpy.context.object
    sun.data.energy = 3.0
    sun.rotation_euler = (0.8, 0.2, 0.5)

    bpy.ops.object.light_add(type='AREA', location=(-2, -2, 2))
    fill = bpy.context.object
    fill.data.energy = 100.0
    fill.data.size = 2.0


def setup_camera(obj):
    import bpy
    import mathutils

    bpy.ops.object.camera_add(location=(0.6, -0.6, 0.5))
    camera = bpy.context.object
    bpy.context.scene.camera = camera

    direction = mathutils.Vector((0, 0, 0.08)) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def setup_ground(ground_level: float = 0.0):
    import bpy

    # Blender uses Z-up, so ground_level (Y in sim) becomes Z in Blender
    bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, ground_level))
    ground = bpy.context.object
    ground.name = "ground"

    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1)
    bsdf.inputs['Roughness'].default_value = 0.9

    ground.data.materials.append(mat)
    return ground


def load_metadata(meta_path: Path) -> dict:
    """Load metadata from JSON file"""
    if not meta_path.exists():
        return {"ground_level": 0.0, "material": "default", "mass_g": 1000}
    with open(meta_path, 'r') as f:
        return json.load(f)


def load_animation(obj, anim_path: Path, metadata: dict, target_fps: int = 60):
    import bpy
    from mathutils import Quaternion

    if not anim_path.exists():
        print(f"Animation file not found: {anim_path}")
        return

    with open(anim_path, 'r') as f:
        lines = [l for l in f.readlines() if not l.startswith('#') and l.strip()]

    if not lines:
        print("No animation data found")
        return

    bpy.context.scene.render.fps = target_fps

    # Parse animation data (absolute format: time tx ty tz qw qx qy qz)
    absolute_frames = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        t, x, y, z, qw, qx, qy, qz = map(float, parts[:8])
        absolute_frames.append((t, [x, y, z], [qw, qx, qy, qz]))

    if not absolute_frames:
        print("No animation data found")
        return

    start_time = absolute_frames[0][0]
    end_time = absolute_frames[-1][0]
    duration = end_time - start_time
    total_frames = int(duration * target_fps) + 1

    print(f"Animation: {len(lines)} samples, {duration:.3f}s, {total_frames} frames @ {target_fps}fps")

    # Build time-indexed lookup for resampling
    time_to_frame = {t: (pos, rot) for t, pos, rot in absolute_frames}
    times = [t for t, _, _ in absolute_frames]

    obj.rotation_mode = 'QUATERNION'

    # Second pass: insert keyframes at target fps
    for frame_idx in range(total_frames):
        target_time = start_time + frame_idx / target_fps

        # Find closest source frame
        closest_time = min(times, key=lambda t: abs(t - target_time))
        pos, rot = time_to_frame[closest_time]

        obj.location = tuple(pos)
        obj.rotation_quaternion = Quaternion(rot)
        obj.keyframe_insert(data_path="location", frame=frame_idx)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = total_frames - 1
    print(f"Loaded {total_frames} keyframes")


def main():
    args = get_args()

    import bpy

    dataset_path = Path(args.dataset)
    obj_path = dataset_path / args.object / 'preprocessed' / 'transformed.obj'

    print(f"Loading: {args.object}")
    print(f"  OBJ: {obj_path}")

    setup_scene(use_eevee=args.eevee)
    obj = load_object(obj_path)
    setup_lighting()

    ground_level = 0.0
    has_animation = False
    metadata = {}

    # Load metadata from JSON if available
    if args.metadata:
        meta_path = Path(args.metadata)
        metadata = load_metadata(meta_path)
    elif args.animation:
        # Try to find metadata JSON alongside animation file
        anim_path = Path(args.animation)
        meta_path = anim_path.parent / anim_path.name.replace('_anim.txt', '_meta.json')
        if meta_path.exists():
            metadata = load_metadata(meta_path)

    if args.animation:
        anim_path = Path(args.animation)
        load_animation(obj, anim_path, metadata, target_fps=args.fps)
        ground_level = metadata.get("ground_level", 0.0)
        has_animation = True
        print(f"Material: {metadata.get('material', 'unknown')}, Mass: {metadata.get('mass_g', 0):.1f}g")
        print(f"Description: {metadata.get('description', 'N/A')}")

    setup_ground(ground_level)
    setup_camera(obj)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.preview:
        blend_path = output_dir / f"{args.object}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        print(f"Saved: {blend_path}")
        print(f"Open with: blender {blend_path}")
    elif args.render_video and has_animation:
        bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
        bpy.context.scene.render.ffmpeg.format = 'MPEG4'
        bpy.context.scene.render.ffmpeg.codec = 'H264'
        bpy.context.scene.render.ffmpeg.constant_rate_factor = 'HIGH'
        bpy.context.scene.render.filepath = str(output_dir / f"{args.object}")
        print(f"Rendering animation: {bpy.context.scene.frame_end + 1} frames...")
        bpy.ops.render.render(animation=True)
        print(f"Rendered: {output_dir / args.object}.mp4")
    else:
        render_path = output_dir / f"{args.object}.png"
        bpy.context.scene.render.filepath = str(render_path)
        bpy.ops.render.render(write_still=True)
        print(f"Rendered: {render_path}")


if __name__ == "__main__":
    main()
