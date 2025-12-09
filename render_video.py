#!/usr/bin/env python3
"""
Full pipeline: Physics simulation -> Blender render -> Video with audio
Each object outputs to its own subdirectory: output/<object_name>/
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_simulation(args, output_dir: Path):
    """Run physics simulation to generate animation and audio"""
    cmd = [
        sys.executable, "realimpact_sim.py",
        "--dataset", args.dataset,
        "--object", args.object,
        "--duration", str(args.duration),
        "--output-dir", str(output_dir),
        "--headless"
    ]
    print(f"\n=== Running simulation ===")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def render_video(args, output_dir: Path):
    """Render video using Blender"""
    anim_path = output_dir / f"{args.object}_anim.txt"
    cmd = [
        "blender", "--background", "--python", "blender_render.py",
        "--", "--object", args.object,
        "--dataset", args.dataset,
        "--animation", str(anim_path),
        "--output", str(output_dir),
        "--render-video",
        "--fps", str(args.fps),
    ]
    if args.eevee:
        cmd.append("--eevee")
    print(f"\n=== Rendering video ===")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def combine_audio(args, output_dir: Path):
    """Combine video with audio using ffmpeg"""
    # Find the rendered video file
    video_files = list(output_dir.glob(f"{args.object}*.mp4"))
    video_files = [f for f in video_files if "final" not in f.name]

    if not video_files:
        print(f"Error: No video file found in {output_dir}")
        return None

    video_path = video_files[0]
    audio_path = output_dir / f"{args.object}_audio.wav"
    final_path = output_dir / f"{args.object}_final.mp4"

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return None

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(final_path)
    ]
    print(f"\n=== Combining audio ===")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Clean up intermediate video file
    video_path.unlink()

    print(f"\n=== Complete ===")
    print(f"Final video: {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Full render pipeline")
    parser.add_argument("--dataset", type=str, default="/run/media/jim_z/SRC/dev/RealImpact/dataset")
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--eevee", action="store_true", help="Use EEVEE for fast rendering")
    parser.add_argument("--skip-sim", action="store_true", help="Skip simulation")
    parser.add_argument("--skip-render", action="store_true", help="Skip Blender render")
    args = parser.parse_args()

    # Create object-specific output directory
    output_dir = Path(args.output) / args.object
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    if not args.skip_sim:
        run_simulation(args, output_dir)

    if not args.skip_render:
        render_video(args, output_dir)

    combine_audio(args, output_dir)


if __name__ == "__main__":
    main()
