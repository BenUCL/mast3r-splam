#!/usr/bin/env python3
"""
convert_to_pinhole.py

Convert cameras.bin from OPENCV model to PINHOLE model since
the keyframes are already undistorted by MASt3R-SLAM.

PINHOLE model only has 4 params (fx, fy, cx, cy) - no distortion.

Usage:
    python convert_to_pinhole.py --dataset reef_soneva
"""

import argparse
import struct
from pathlib import Path
import shutil

INTERMEDIATE_DATA_ROOT = Path('/home/bwilliams/encode/data/intermediate_data')


def read_cameras_bin(cameras_bin):
    """Read cameras.bin and return camera data."""
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
    """Write cameras.bin with PINHOLE model."""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<Q', 1))
        f.write(struct.pack('<i', camera_data['camera_id']))
        f.write(struct.pack('<i', camera_data['model_id']))
        f.write(struct.pack('<Q', camera_data['width']))
        f.write(struct.pack('<Q', camera_data['height']))
        
        for param in camera_data['params']:
            f.write(struct.pack('<d', param))


def main():
    parser = argparse.ArgumentParser(
        description="Convert cameras.bin to PINHOLE model (keyframes already undistorted)"
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
    cameras_bin_backup = sparse_dir / 'cameras.bin.opencv_backup'
    
    if not cameras_bin.exists():
        raise FileNotFoundError(f"cameras.bin not found: {cameras_bin}")
    
    print(f"\n{'='*70}")
    print(f"Converting to PINHOLE model")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"File:    {cameras_bin}")
    print()
    
    # Read original
    camera = read_cameras_bin(cameras_bin)
    
    # Get model name
    MODEL_ID_TO_NAME = {
        0: 'SIMPLE_PINHOLE',
        1: 'PINHOLE',
        2: 'SIMPLE_RADIAL',
        3: 'RADIAL',
        4: 'OPENCV',
        5: 'OPENCV_FISHEYE',
        6: 'FULL_OPENCV',
        7: 'FOV',
        8: 'SIMPLE_RADIAL_FISHEYE',
        9: 'RADIAL_FISHEYE',
        10: 'THIN_PRISM_FISHEYE'
    }
    
    old_model = MODEL_ID_TO_NAME.get(camera['model_id'], f"Unknown({camera['model_id']})")
    
    print(f"Original model: {old_model}")
    print(f"Original parameters: {camera['params'][:4]}")
    if len(camera['params']) > 4:
        print(f"Distortion (being removed): {camera['params'][4:]}")
    print()
    
    # Backup original
    if not cameras_bin_backup.exists():
        shutil.copy(cameras_bin, cameras_bin_backup)
        print(f"✓ Backed up to: {cameras_bin_backup.name}")
    else:
        print(f"✓ Backup already exists: {cameras_bin_backup.name}")
    
    # Convert to PINHOLE (model_id=1, only fx, fy, cx, cy)
    camera['model_id'] = 1
    camera['params'] = camera['params'][:4]  # Keep only fx, fy, cx, cy
    
    # Write modified
    write_cameras_bin(cameras_bin, camera)
    
    print()
    print(f"New model: PINHOLE")
    print(f"New parameters: fx={camera['params'][0]:.2f}, fy={camera['params'][1]:.2f}, cx={camera['params'][2]:.2f}, cy={camera['params'][3]:.2f}")
    print()
    
    print(f"{'='*70}")
    print(f"✅ SUCCESS")
    print(f"{'='*70}")
    print()
    print(f"Reasoning:")
    print(f"  - MASt3R-SLAM undistorts images using cv2.remap()")
    print(f"  - Keyframes are already undistorted (verified)")
    print(f"  - PINHOLE model = no distortion, just focal length and principal point")
    print(f"  - LichtFeld-Studio won't require --gut flag with PINHOLE model")
    print()
    print(f"Next steps:")
    print(f"  Run LichtFeld-Studio WITHOUT --gut flag (PINHOLE model has no distortion)")
    print()
    print(f"To restore original:")
    print(f"  cp {cameras_bin_backup} {cameras_bin}")
    print()


if __name__ == '__main__':
    main()
