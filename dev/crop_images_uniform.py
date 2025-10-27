#!/usr/bin/env python3
"""
crop_images_uniform.py

Crop all images in a directory to uniform dimensions (minimum width x minimum height).
This ensures all images have identical dimensions, which is required by MASt3R-SLAM.

Usage:
    python crop_images_uniform.py --images_path /path/to/images
    python crop_images_uniform.py --images_path /path/to/images --no-backup
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
from datetime import datetime
import shutil


def get_all_images(images_path):
    """Find all image files (PNG, JPG, JPEG) in directory."""
    image_files = []
    
    # Find all image extensions (case-insensitive)
    for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
        image_files.extend(images_path.glob(ext))
    
    return sorted(image_files)


def get_image_dimensions(image_path):
    """Get dimensions of an image without loading the entire file."""
    with Image.open(image_path) as img:
        return img.size  # Returns (width, height)


def analyze_dimensions(image_files):
    """Analyze dimensions of all images and return min width/height and distribution."""
    dimensions = {}
    min_width = float('inf')
    min_height = float('inf')
    
    print("Analyzing image dimensions...")
    for img_path in image_files:
        width, height = get_image_dimensions(img_path)
        
        # Track minimum
        min_width = min(min_width, width)
        min_height = min(min_height, height)
        
        # Track distribution
        dim_key = f"{width}x{height}"
        dimensions[dim_key] = dimensions.get(dim_key, 0) + 1
    
    return min_width, min_height, dimensions


def crop_image(image_path, target_width, target_height):
    """Crop image to target dimensions (center crop)."""
    with Image.open(image_path) as img:
        width, height = img.size
        
        # Calculate crop box (center crop)
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        # Crop and save (overwrite original)
        cropped = img.crop((left, top, right, bottom))
        cropped.save(image_path)


def main():
    parser = argparse.ArgumentParser(
        description="Crop all images in a directory to uniform dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python crop_images_uniform.py --images_path /path/to/images
  python crop_images_uniform.py --images_path /path/to/images --backup
  
This script:
  1. Finds the minimum width and height across all images
  2. Center-crops all images to the minimum dimensions (in-place by default)
  3. Optional: Create timestamped backup with --backup flag
  
This ensures all images have identical dimensions, which is required by MASt3R-SLAM.
"""
    )
    parser.add_argument(
        '--images_path',
        type=str,
        required=True,
        help='Path to directory containing images to crop'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create timestamped backup before cropping'
    )
    
    args = parser.parse_args()
    
    # Validate input
    images_path = Path(args.images_path).expanduser().resolve()
    if not images_path.exists():
        print(f"Error: Directory does not exist: {images_path}")
        sys.exit(1)
    
    if not images_path.is_dir():
        print(f"Error: Path is not a directory: {images_path}")
        sys.exit(1)
    
    # Find all images
    print(f"Searching for images in: {images_path}")
    image_files = get_all_images(images_path)
    
    if not image_files:
        print("Error: No images found (looking for .png, .jpg, .jpeg)")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images")
    print()
    
    # Analyze dimensions
    min_width, min_height, dimensions = analyze_dimensions(image_files)
    
    print(f"\nMinimum dimensions: {min_width}x{min_height}")
    print(f"\nDimension distribution:")
    for dim, count in sorted(dimensions.items()):
        print(f"  {dim}: {count} images")
    
    # Check if all images already have the same dimensions
    if len(dimensions) == 1:
        print("\n✓ All images already have identical dimensions!")
        print("No cropping needed.")
        sys.exit(0)
    
    # Confirm with user
    print()
    response = input(f"Crop all images to {min_width}x{min_height} (center crop)? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # Create backup if --backup flag provided
    if args.backup:
        print("\nCreating backup...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = images_path.parent / f"{images_path.name}_backup_{timestamp}"
        
        try:
            shutil.copytree(images_path, backup_dir)
            print(f"✓ Backup created: {backup_dir}")
        except Exception as e:
            print(f"Error creating backup: {e}")
            response = input("Continue without backup? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(1)
    else:
        print("\n⚠️  No backup will be created (use --backup to create one)")
    
    # Crop images
    print(f"\nCropping {len(image_files)} images to {min_width}x{min_height}...")
    
    for i, img_path in enumerate(image_files, 1):
        crop_image(img_path, min_width, min_height)
        print(f"  [{i}/{len(image_files)}] {img_path.name}", end='\r')
    
    print()  # New line after progress
    print(f"\n✓ Done! All images cropped to {min_width}x{min_height}")
    
    if args.backup:
        print(f"  Original images backed up to: {backup_dir}")
    
    # Verify result
    print("\nVerifying result...")
    _, _, new_dimensions = analyze_dimensions(image_files)
    
    print("Final dimension distribution:")
    for dim, count in sorted(new_dimensions.items()):
        print(f"  {dim}: {count} images")
    
    if len(new_dimensions) == 1:
        print("\n✅ SUCCESS: All images now have identical dimensions!")
    else:
        print("\n⚠️  WARNING: Some images still have different dimensions")


if __name__ == '__main__':
    main()
