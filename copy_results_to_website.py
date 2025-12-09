#!/usr/bin/env python3
"""
Helper script to copy simulation results to the website assets folder.
"""

import os
import shutil
from pathlib import Path
import argparse

def copy_results(source_dir='output', dry_run=False):
    """
    Copy video and audio files from output directory to website assets.

    Args:
        source_dir: Directory containing simulation results
        dry_run: If True, only print what would be copied without copying
    """
    source_path = Path(source_dir)
    video_dest = Path('docs/assets/videos')
    audio_dest = Path('docs/assets/audio')

    # Create destination directories if they don't exist
    video_dest.mkdir(parents=True, exist_ok=True)
    audio_dest.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_files = list(source_path.glob('*_final.mp4'))
    audio_files = list(source_path.glob('*_audio.wav'))

    print(f"Found {len(video_files)} video files and {len(audio_files)} audio files\n")

    # Copy videos
    for video in video_files:
        dest = video_dest / video.name
        if dry_run:
            print(f"[DRY RUN] Would copy: {video} -> {dest}")
        else:
            print(f"Copying: {video.name} -> {dest}")
            shutil.copy2(video, dest)

    print()

    # Copy audio
    for audio in audio_files:
        dest = audio_dest / audio.name
        if dry_run:
            print(f"[DRY RUN] Would copy: {audio} -> {dest}")
        else:
            print(f"Copying: {audio.name} -> {dest}")
            shutil.copy2(audio, dest)

    if dry_run:
        print("\n[DRY RUN] No files were actually copied. Run without --dry-run to copy.")
    else:
        print(f"\n✓ Copied {len(video_files)} videos and {len(audio_files)} audio files")
        print("\nNext steps:")
        print("1. Update docs/index.html to reference your new files")
        print("2. Test locally: cd docs && python -m http.server 8000")
        print("3. Commit and push to deploy: git add docs/ && git commit -m 'Add results' && git push")

def generate_html_cards(source_dir='output'):
    """
    Generate HTML video card snippets for easy copy-paste into index.html
    """
    source_path = Path(source_dir)
    video_files = sorted(source_path.glob('*_final.mp4'))

    print("\n" + "="*60)
    print("HTML Video Card Snippets")
    print("="*60 + "\n")
    print("Copy these into the <div class=\"video-gallery\"> section in docs/index.html:\n")

    for video in video_files:
        # Extract object name from filename (remove _final.mp4)
        name = video.stem.replace('_final', '').replace('_', ' ').title()

        # Try to extract material from name (first word is often material)
        parts = name.split()
        material = parts[0] if parts else "TBD"

        html = f'''    <div class="video-card">
        <div class="video-container">
            <video controls preload="metadata">
                <source src="assets/videos/{video.name}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        <div class="video-info">
            <h3>{name}</h3>
            <p>Material: {material} | Drop height: TBD | Duration: 5s</p>
        </div>
    </div>
'''
        print(html)

def main():
    parser = argparse.ArgumentParser(description='Copy simulation results to website')
    parser.add_argument('--source', default='output', help='Source directory (default: output)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be copied without copying')
    parser.add_argument('--generate-html', action='store_true', help='Generate HTML snippets for video cards')

    args = parser.parse_args()

    if args.generate_html:
        generate_html_cards(args.source)
    else:
        copy_results(args.source, args.dry_run)

if __name__ == '__main__':
    main()
