#!/usr/bin/env python3
"""
estimate_intrinsics.py

Fast self-calibration: extract intrinsics from RAW images using COLMAP.
Uses a subset of images with sequential matcher for speed.

Python version of estimate_intrinsics.sh with improved error handling and progress feedback.

Usage:
    python estimate_intrinsics.py --images_path /path/to/images --dataset reef_soneva
    python estimate_intrinsics.py --images_path /path/to/images --dataset reef_soneva --num_images 150
    python estimate_intrinsics.py --images_path /path/to/images --output_dir /custom/path --camera_model OPENCV
"""

import argparse
import subprocess
import shutil
import sys
import os
from pathlib import Path
import datetime


INTERMEDIATE_DATA_ROOT = Path('/home/bwilliams/encode/data/intermediate_data')


def get_clean_colmap_env():
    """Get environment for COLMAP with snap/conda interference removed (like bash script does)."""
    env = os.environ.copy()
    
    # Remove the same variables the bash script unsets
    vars_to_remove = [
        'XDG_DATA_DIRS',
        'XDG_CONFIG_DIRS', 
        'GIO_MODULE_DIR',
        'GSETTINGS_SCHEMA_DIR',
        'GTK_PATH',
        'LOCPATH'
    ]
    
    for var in vars_to_remove:
        env.pop(var, None)
    
    return env


def run_command(cmd, description, check=True):
    """Run a shell command with nice output."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Must get cleaned environment for COLMAP commands
        result = subprocess.run(cmd, check=check, capture_output=False, text=True, env=get_clean_colmap_env())
        print(f"✓ {description} completed")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        if check:
            raise
        return False


def count_images(images_path):
    """Count PNG/JPG images in directory."""
    png_count = len(list(images_path.glob('*.png'))) + len(list(images_path.glob('*.PNG')))
    jpg_count = len(list(images_path.glob('*.jpg'))) + len(list(images_path.glob('*.JPG'))) + \
                len(list(images_path.glob('*.jpeg'))) + len(list(images_path.glob('*.JPEG')))
    return png_count + jpg_count


def select_images(images_path, num_images):
    """Select first N images (consecutive for good overlap)."""
    # Find all images
    all_images = []
    for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
        all_images.extend(images_path.glob(ext))
    
    all_images = sorted(all_images)
    total = len(all_images)
    
    if total == 0:
        raise ValueError(f"No images found in {images_path}")
    
    print(f"Found {total} total images")
    
    if total <= num_images:
        print(f"Using all {total} images (less than requested {num_images})")
        return all_images
    else:
        print(f"Selected first {num_images} images (consecutive frames for overlap)")
        return all_images[:num_images]


def create_image_subset(selected_images, subset_dir):
    """Create symlinks to selected images."""
    subset_dir.mkdir(parents=True, exist_ok=True)
    
    for img in selected_images:
        link_path = subset_dir / img.name
        if link_path.exists():
            link_path.unlink()
        link_path.symlink_to(img)
    
    print(f"✓ Created {len(selected_images)} symlinks in {subset_dir}")


def parse_cameras_txt(cameras_txt):
    """Parse cameras.txt to extract camera parameters."""
    with open(cameras_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            return {
                'camera_id': int(parts[0]),
                'model': parts[1],
                'width': int(parts[2]),
                'height': int(parts[3]),
                'params': ' '.join(parts[4:])
            }
    return None


def count_registered_images(images_txt):
    """Count registered images (images.txt has 2 lines per image)."""
    count = 0
    with open(images_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            count += 1
    return count // 2  # 2 lines per image


def count_points(points3d_txt):
    """Count 3D points in points3D.txt."""
    count = 0
    with open(points3d_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line[0].isdigit():
                count += 1
    return count


def write_summary(work_dir, dataset_name, camera_model, num_images, cam_data, num_registered, num_points):
    """Write calibration summary file."""
    summary_file = work_dir / 'calibration_summary.txt'
    
    registration_pct = (num_registered / num_images) * 100 if num_images > 0 else 0
    
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COLMAP Camera Calibration Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.datetime.now()}\n")
        f.write(f"Dataset: {dataset_name}\n\n")
        
        f.write(f"Camera Model:  {camera_model}\n")
        f.write(f"Resolution:    {cam_data['width']} x {cam_data['height']}\n")
        f.write(f"Images Used:   {num_images}\n")
        f.write(f"Registered:    {num_registered} ({registration_pct:.1f}%)\n")
        f.write(f"3D Points:     {num_points}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("Calibrated Intrinsics (cameras.txt)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Camera ID: {cam_data['camera_id']}\n")
        f.write(f"Model:     {cam_data['model']}\n")
        f.write(f"Parameters: {cam_data['params']}\n\n")
        
        if camera_model == 'OPENCV':
            f.write("Format: fx, fy, cx, cy, k1, k2, p1, p2\n")
            f.write("  fx, fy: focal lengths\n")
            f.write("  cx, cy: principal point\n")
            f.write("  k1, k2: radial distortion\n")
            f.write("  p1, p2: tangential distortion\n\n")
        elif camera_model == 'OPENCV_FISHEYE':
            f.write("Format: fx, fy, cx, cy, k1, k2, k3, k4\n")
            f.write("  fx, fy: focal lengths\n")
            f.write("  cx, cy: principal point\n")
            f.write("  k1-k4: fisheye distortion coefficients\n\n")
        
        f.write("NOTE: This calibration is only for extracting intrinsics from cameras.txt.\n")
        f.write("      The sparse reconstruction quality is NOT important.\n")
        f.write("      What matters: intrinsics are estimated from feature correspondences.\n\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"Registered Images ({num_registered} total)\n")
        f.write("-" * 80 + "\n")
        
        # Extract and list registered image names from images.txt
        # Format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (10 fields)
        #         POINTS2D[] (second line, usually empty)
        images_txt = work_dir / 'images.txt'
        registered_images = []
        with open(images_txt, 'r') as img_f:
            for line in img_f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                # Image metadata line has exactly 10 fields: ID, quat(4), trans(3), cam_id, name
                if len(parts) == 10:
                    registered_images.append(parts[-1])  # Last field is filename
        
        # Sort and write image names
        for img_name in sorted(set(registered_images)):
            f.write(f"  {img_name}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✓ Summary written to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract camera intrinsics from RAW images using COLMAP"
    )
    parser.add_argument(
        '--images_path',
        type=str,
        required=True,
        help='Path to directory containing raw images'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset name (used to auto-generate output directory)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Custom output directory (default: auto-generated from dataset name)'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=100,
        help='Number of images to use for calibration (default: 100)'
    )
    parser.add_argument(
        '--camera_model',
        type=str,
        default='OPENCV',
        choices=['OPENCV', 'OPENCV_FISHEYE'],
        help='Camera model (default: OPENCV for GoPro/wide-angle, OPENCV_FISHEYE for extreme fisheye)'
        #TODO: should be able to pass any camera model colmap accepts, not just these two
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output directory without prompting'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    images_path = Path(args.images_path).expanduser().resolve()
    if not images_path.exists():
        print(f"Error: Images path does not exist: {images_path}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif args.dataset:
        output_dir = INTERMEDIATE_DATA_ROOT / args.dataset
    else:
        print("Error: Must provide either --dataset or --output_dir")
        sys.exit(1)
    
    # Handle existing output directory
    if output_dir.exists() and not args.overwrite:
        response = input(f"Output directory exists: {output_dir}\nOverwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
        print("Overwriting existing directory...")
    
    # Create output structure
    work_dir = output_dir / 'colmap_outputs'
    work_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = work_dir / 'database.db'
    sparse_dir = work_dir / 'sparse'
    sparse_dir.mkdir(exist_ok=True)
    image_subset_dir = work_dir / 'images_subset'
    
    # Print configuration
    print("\n" + "="*70)
    print("COLMAP Intrinsics Calibration")
    print("="*70)
    print(f"Images path:   {images_path}")
    print(f"Output dir:    {output_dir}")
    if args.dataset:
        print(f"Dataset name:  {args.dataset}")
    print(f"Num images:    {args.num_images}")
    print(f"Camera model:  {args.camera_model}")
    print(f"Overwrite:     {args.overwrite}")
    print()
    
    # Step 1: Create image subset
    print("[1/5] Creating subset of images...")
    total_images = count_images(images_path)
    print(f"Found {total_images} images in {images_path}")
    
    selected_images = select_images(images_path, args.num_images)
    create_image_subset(selected_images, image_subset_dir)
    
    # Step 2: Create COLMAP database
    run_command(
        ['colmap', 'database_creator', '--database_path', str(db_path)],
        "[2/5] Creating COLMAP database"
    )
    
    # Step 3: Feature extraction
    run_command([
        'colmap', 'feature_extractor',
        '--database_path', str(db_path),
        '--image_path', str(image_subset_dir),
        '--ImageReader.camera_model', args.camera_model,
        '--ImageReader.single_camera', '1'
    ], "[3/5] Extracting features")
    
    # Step 4: Sequential matching
    run_command([
        'colmap', 'sequential_matcher',
        '--database_path', str(db_path)
    ], "[4/5] Matching features (sequential matcher)")
    
    # Step 5: Mapper (bundle adjustment)
    mapper_success = run_command([
        'colmap', 'mapper',
        '--database_path', str(db_path),
        '--image_path', str(image_subset_dir),
        '--output_path', str(sparse_dir),
        '--Mapper.ba_refine_focal_length', '1',
        '--Mapper.ba_refine_principal_point', '1',
        '--Mapper.ba_refine_extra_params', '1'
    ], "[5/5] Running bundle adjustment to refine intrinsics", check=False)
    
    if not mapper_success or not (sparse_dir / '0').exists():
        print("\n" + "="*70)
        print("ERROR: Bundle adjustment failed")
        print("="*70)
        print("Possible causes:")
        print("  - Not enough images with overlapping features")
        print("  - Images too blurry or low quality")
        print("  - Scene lacks sufficient texture/features")
        print("\nSuggestions:")
        print("  - Try another part of the image sequence")
        print("  - Use images with more overlap")
        print("  - Check image quality")
        sys.exit(1)
    
    # Export to text format
    print("\nExporting to text format...")
    run_command([
        'colmap', 'model_converter',
        '--input_path', str(sparse_dir / '0'),
        '--output_path', str(work_dir),
        '--output_type', 'TXT'
    ], "Exporting model to text format")
    
    # Verify output
    cameras_txt = work_dir / 'cameras.txt'
    if not cameras_txt.exists():
        print(f"Error: cameras.txt not found at {cameras_txt}")
        sys.exit(1)
    
    # Parse results and write summary
    print("\nGenerating calibration summary...")
    cam_data = parse_cameras_txt(cameras_txt)
    if not cam_data:
        print("Error: Could not parse cameras.txt")
        sys.exit(1)
    
    num_registered = count_registered_images(work_dir / 'images.txt')
    num_points = count_points(work_dir / 'points3D.txt')
    
    write_summary(
        work_dir,
        args.dataset or 'unknown',
        args.camera_model,
        len(selected_images),
        cam_data,
        num_registered,
        num_points
    )
    
    # Final summary
    print("\n" + "="*70)
    print("✅ SUCCESS")
    print("="*70)
    print(f"\nCalibrated intrinsics saved to:")
    print(f"  {cameras_txt}")
    print(f"\nSummary report:")
    print(f"  {work_dir / 'calibration_summary.txt'}")
    print(f"\nNext steps:")
    print(f"  Run shuttle_intrinsics.py --dataset {args.dataset or '<dataset>'}")
    print()


if __name__ == '__main__':
    main()
