#!/usr/bin/env python3
"""Batch render all simulated objects to video"""

import subprocess
from pathlib import Path


def get_simulated_objects(output_dir: Path):
    """Get list of objects that have animation files"""
    objects = set()
    for anim_file in output_dir.glob("*_anim.txt"):
        obj_name = anim_file.stem.replace("_anim", "")
        final_video = output_dir / f"{obj_name}_final.mp4"
        if not final_video.exists():
            objects.add(obj_name)
    return sorted(objects)


def render_object(obj_name: str, output_dir: Path, dataset: str, fps: int, use_eevee: bool):
    """Render video for a single object"""
    anim_path = output_dir / f"{obj_name}_anim.txt"
    audio_path = output_dir / f"{obj_name}_audio.wav"

    if not anim_path.exists():
        print(f"  Error: Animation file not found: {anim_path}")
        return False

    cmd = [
        "blender", "--background", "--python", "blender_render.py",
        "--", "--object", obj_name,
        "--dataset", dataset,
        "--animation", str(anim_path),
        "--output", str(output_dir),
        "--render-video",
        "--fps", str(fps),
    ]
    if use_eevee:
        cmd.append("--eevee")

    print(f"  Rendering video...")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"  Error rendering: {e}")
        return False

    video_files = list(output_dir.glob(f"{obj_name}*.mp4"))
    video_files = [f for f in video_files if "final" not in f.name]

    if not video_files:
        print(f"  Error: No video file found")
        return False

    video_path = video_files[0]
    final_path = output_dir / f"{obj_name}_final.mp4"

    if not audio_path.exists():
        print(f"  Warning: No audio file, copying video without audio")
        video_path.rename(final_path)
        return True

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(final_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        video_path.unlink()
        print(f"  Done: {final_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error combining audio: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/run/media/jim_z/SRC/dev/RealImpact/dataset")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--eevee", action="store_true", help="Use EEVEE for fast rendering")
    parser.add_argument("--objects", nargs="*", help="Specific objects to render")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.objects:
        objects = args.objects
    else:
        objects = get_simulated_objects(output_dir)

    print(f"Found {len(objects)} objects to render")

    success = 0
    failed = []

    for i, obj in enumerate(objects):
        print(f"\n[{i+1}/{len(objects)}] {obj}")
        if render_object(obj, output_dir, args.dataset, args.fps, args.eevee):
            success += 1
        else:
            failed.append(obj)

    print(f"\n{'='*60}")
    print(f"Complete: {success}/{len(objects)} successful")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
