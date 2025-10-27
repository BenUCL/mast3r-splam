#!/usr/bin/env python3
"""
Script to rename dual-camera images with sequential prefixes.
Camera 1: GPAA prefix
Camera 2: GPAB (first) then GPAC (continuing sequence)
"""

import os
import shutil
from pathlib import Path

# Configuration
INPUT_DIR = "/home/bwilliams/encode/data/sweet-coral_indo_tabuhan_p1_20250210/_indonesia_tabuhan_p1_20250210/corrected/combined_small_prep"
OUTPUT_DIR = "/home/bwilliams/encode/data/sweet-coral_indo_tabuhan_p1_20250210/_indonesia_tabuhan_p1_20250210/corrected/combined_small"


def extract_number(filename):
    """Extract the numeric part from the filename."""
    # Extract the 4-digit number from filenames like GPAA0483.jpg
    return int(filename[4:8])


def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    all_files = sorted([f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    # Separate into camera 1 and camera 2
    cam1_files = []  # GPAA
    cam2_gpab_files = []  # GPAB
    cam2_gpac_files = []  # GPAC
    
    for file in all_files:
        name = file.name
        if name.startswith('GPAA'):
            cam1_files.append(file)
        elif name.startswith('GPAB'):
            cam2_gpab_files.append(file)
        elif name.startswith('GPAC'):
            cam2_gpac_files.append(file)
    
    # Sort each list by the numeric part
    cam1_files.sort(key=lambda f: extract_number(f.name))
    cam2_gpab_files.sort(key=lambda f: extract_number(f.name))
    cam2_gpac_files.sort(key=lambda f: extract_number(f.name))
    
    # Combine cam2 files: GPAB first, then GPAC
    cam2_files = cam2_gpab_files + cam2_gpac_files
    
    print(f"Found {len(cam1_files)} Camera 1 (GPAA) images")
    print(f"Found {len(cam2_gpab_files)} Camera 2 (GPAB) images")
    print(f"Found {len(cam2_gpac_files)} Camera 2 (GPAC) images")
    print(f"Total Camera 2 images: {len(cam2_files)}")
    print()
    
    # Rename and copy Camera 1 files
    print("Processing Camera 1 images...")
    for idx, file in enumerate(cam1_files, start=1):
        new_name = f"{idx:04d}_cam1_{file.name}"
        new_path = output_path / new_name
        shutil.copy2(file, new_path)
        if idx <= 3 or idx == len(cam1_files):
            print(f"  {file.name} → {new_name}")
        elif idx == 4:
            print(f"  ...")
    
    print()
    
    # Rename and copy Camera 2 files
    print("Processing Camera 2 images...")
    for idx, file in enumerate(cam2_files, start=1):
        new_name = f"{idx:04d}_cam2_{file.name}"
        new_path = output_path / new_name
        shutil.copy2(file, new_path)
        if idx <= 3 or idx == len(cam2_files) or idx == len(cam2_gpab_files) or idx == len(cam2_gpab_files) + 1:
            print(f"  {file.name} → {new_name}")
        elif idx == 4 and len(cam2_files) > 6:
            print(f"  ...")
    
    print()
    print(f"✓ Done! Renamed images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
