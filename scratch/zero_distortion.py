#!/usr/bin/env python3
"""
zero_distortion.py

Remove distortion coefficients from cameras.bin since MASt3R-SLAM
already undistorts keyframes with cv2.remap().

This creates a copy of cameras.bin with k1=k2=p1=p2=0.

Usage:
    python zero_distortion.py --dataset reef_soneva
"""

import argparse
import struct
from pathlib import Path
import shutil

INTERMEDIATE_DATA_ROOT = Path('/home/bwilliams/encode/data/intermediate_data')


def read_cameras_bin(cameras_bin):
    """Read cameras.bin and return camera data."""
    # Model ID to number of params
    MODEL_NUM_PARAMS = {
        0: 3,   # SIMPLE_PINHOLE
        1: 4,   # PINHOLE
        2: 4,   # SIMPLE_RADIAL
        3: 5,   # RADIAL
        4: 8,   # OPENCV
        5: 8,   # OPENCV_FISHEYE
        6: 12,  # FULL_OPENCV
        7: 5,   # FOV
        8: 5,   # SIMPLE_RADIAL_FISHEYE
        9: 6,   # RADIAL_FISHEYE
        10: 12  # THIN_PRISM_FISHEYE
    }
    
    with open(cameras_bin, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        
        if num_cameras != 1:
            raise ValueError(f"Expected 1 camera, got {num_cameras}")
        
        camera_id = struct.unpack('<i', f.read(4))[0]
        model_id = struct.unpack('<i', f.read(4))[0]
        width = struct.unpack('<Q', f.read(8))[0]
        height = struct.unpack('<Q', f.read(8))[0]
        
        # Get num params from model_id
        num_params = MODEL_NUM_PARAMS.get(model_id)
        if num_params is None:
            raise ValueError(f"Unknown model_id: {model_id}")
        
        params = []
        for _ in range(num_params):
            params.append(struct.unpack('<d', f.read(8))[0])
    
    return {
        'camera_id': camera_id,
        'model_id': model_id,
        'width': width,
        'height': height,
        'params': params
    }


def write_cameras_bin(output_path, camera_data):
    """Write cameras.bin with modified params."""
    with open(output_path, 'wb') as f:
        # Number of cameras
        f.write(struct.pack('<Q', 1))
        
        # Camera data
        f.write(struct.pack('<i', camera_data['camera_id']))
        f.write(struct.pack('<i', camera_data['model_id']))
        f.write(struct.pack('<Q', camera_data['width']))
        f.write(struct.pack('<Q', camera_data['height']))
        
        # Params (no num_params field - determined by model_id)
        for param in camera_data['params']:
            f.write(struct.pack('<d', param))


def main():
    parser = argparse.ArgumentParser(
        description="Remove distortion from cameras.bin (keyframes already undistorted by MASt3R-SLAM)"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., reef_soneva)'
    )
    
    args = parser.parse_args()
    
    # Paths
    dataset_root = INTERMEDIATE_DATA_ROOT / args.dataset
    sparse_dir = dataset_root / 'for_splat' / 'sparse' / '0'
    cameras_bin = sparse_dir / 'cameras.bin'
    cameras_bin_backup = sparse_dir / 'cameras.bin.with_distortion'
    
    if not cameras_bin.exists():
        raise FileNotFoundError(f"cameras.bin not found: {cameras_bin}")
    
    print(f"\n{'='*70}")
    print(f"Removing distortion from cameras.bin")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"File:    {cameras_bin}")
    print()
    
    # Read original
    camera = read_cameras_bin(cameras_bin)
    
    print(f"Original parameters:")
    print(f"  fx={camera['params'][0]:.2f}, fy={camera['params'][1]:.2f}")
    print(f"  cx={camera['params'][2]:.2f}, cy={camera['params'][3]:.2f}")
    print(f"  k1={camera['params'][4]:.6f}, k2={camera['params'][5]:.6f}")
    print(f"  p1={camera['params'][6]:.6f}, p2={camera['params'][7]:.6f}")
    print()
    
    # Backup original
    if not cameras_bin_backup.exists():
        shutil.copy(cameras_bin, cameras_bin_backup)
        print(f"✓ Backed up to: {cameras_bin_backup.name}")
    else:
        print(f"✓ Backup already exists: {cameras_bin_backup.name}")
    
    # Zero out distortion (keep fx, fy, cx, cy)
    camera['params'][4] = 0.0  # k1
    camera['params'][5] = 0.0  # k2
    camera['params'][6] = 0.0  # p1
    camera['params'][7] = 0.0  # p2
    
    # Write modified
    write_cameras_bin(cameras_bin, camera)
    
    print()
    print(f"New parameters (distortion removed):")
    print(f"  fx={camera['params'][0]:.2f}, fy={camera['params'][1]:.2f}")
    print(f"  cx={camera['params'][2]:.2f}, cy={camera['params'][3]:.2f}")
    print(f"  k1={camera['params'][4]:.6f}, k2={camera['params'][5]:.6f}")
    print(f"  p1={camera['params'][6]:.6f}, p2={camera['params'][7]:.6f}")
    print()
    
    print(f"{'='*70}")
    print(f"✅ SUCCESS")
    print(f"{'='*70}")
    print()
    print(f"Reasoning:")
    print(f"  MASt3R-SLAM undistorts images using cv2.remap() before saving keyframes.")
    print(f"  The keyframes are already undistorted (verified with visualize_distortion.py).")
    print(f"  LichtFeld-Studio with --gut would re-apply distortion correction to")
    print(f"  already-undistorted images, causing geometric errors.")
    print()
    print(f"Next steps:")
    print(f"  1. Run LichtFeld-Studio on reef_soneva WITHOUT --gut flag")
    print(f"  2. Or use the modified cameras.bin with --gut (now has zero distortion)")
    print()
    print(f"To restore original:")
    print(f"  cp {cameras_bin_backup} {cameras_bin}")
    print()


if __name__ == '__main__':
    main()
