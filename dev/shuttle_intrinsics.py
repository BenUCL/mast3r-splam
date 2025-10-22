#!/usr/bin/env python3
"""
shuttle_intrinsics.py

Convert and move COLMAP calibration intrinsics for use in:
1. MASt3R-SLAM (intrinsics.yaml - no conversion)
2. LichtFeld splat (cameras.txt + cameras.bin - adjusted for SLAM resolution)

MUST RUN WITH: conda activate mast3r-slam
"""

import argparse
import numpy as np
import struct
from pathlib import Path
import yaml

# Import MASt3R-SLAM resize function for intrinsics adjustment
from mast3r_slam.dataloader import resize_img

# Dataset
INTERMEDIATE_DATA_ROOT = Path('/home/bwilliams/encode/data/intermediate_data')
# The keyframe size parameter for mast3r-slam (single int, aspect ratio preserved)
SLAM_SIZE = 512


def read_colmap_cameras_txt(cameras_txt_path):
    """Read COLMAP cameras.txt and extract intrinsics."""
    with open(cameras_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            
            return {
                'camera_id': camera_id,
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    raise ValueError(f"No camera found in {cameras_txt_path}")


def scale_intrinsics(K, raw_w, raw_h, target_size):
    """
    Scale intrinsics from raw resolution to target resolution using
    MASt3R-SLAM's resize_img transformation.
    
    Args:
        K: 3x3 intrinsic matrix
        raw_w, raw_h: Raw image dimensions
        target_size: Target size (512 for MASt3R-SLAM) - will maintain aspect ratio
    
    Returns:
        K_scaled: Adjusted 3x3 intrinsic matrix
    """
    _, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
        np.zeros((raw_h, raw_w, 3)), target_size, return_transformation=True
    )
    
    K_scaled = K.copy()
    K_scaled[0, 0] = K[0, 0] / scale_w
    K_scaled[1, 1] = K[1, 1] / scale_h
    K_scaled[0, 2] = K[0, 2] / scale_w - half_crop_w
    K_scaled[1, 2] = K[1, 2] / scale_h - half_crop_h
    return K_scaled


def opencv_params_to_matrix(width, height, params):
    """Convert OPENCV model params to 3x3 intrinsic matrix K."""
    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    return K


def write_mast3r_yaml(output_path, width, height, K, distortion_params):
    """
    Write intrinsics.yaml for MASt3R-SLAM.
    
    Args:
        output_path: Path to save intrinsics.yaml
        width, height: Image dimensions
        K: 3x3 intrinsic matrix
        distortion_params: Distortion coefficients (k1, k2, p1, p2 for OPENCV)
    """
    # Build calibration list: [fx, fy, cx, cy] or [fx, fy, cx, cy, k1, k2, p1, p2]
    calibration = [
        float(K[0, 0]),  # fx
        float(K[1, 1]),  # fy
        float(K[0, 2]),  # cx
        float(K[1, 2])   # cy
    ]
    
    # Check if distortion is significant (not all zeros/near-zeros)
    has_distortion = any(abs(d) > 1e-6 for d in distortion_params)
    
    if has_distortion:
        # Add distortion parameters
        calibration.extend([float(d) for d in distortion_params])
    
    intrinsics_dict = {
        'width': int(width),
        'height': int(height),
        'calibration': calibration
    }
    
    with open(output_path, 'w') as f:
        # Write with comments
        f.write(f"width: {int(width)}\n")
        f.write(f"height: {int(height)}\n")
        if has_distortion:
            f.write("# With distortion (fx, fy, cx, cy, k1, k2, p1, p2)\n")
        else:
            f.write("# Without distortion (fx, fy, cx, cy)\n")
        f.write(f"calibration: {calibration}\n")
    
    print(f"✓ Saved MASt3R-SLAM intrinsics: {output_path}")


def write_colmap_cameras_txt(output_path, camera_id, model, width, height, params):
    """Write COLMAP cameras.txt format."""
    with open(output_path, 'w') as f:
        # Write header
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        
        # Write camera line
        params_str = ' '.join(f'{p}' for p in params)
        f.write(f"{camera_id} {model} {width} {height} {params_str}\n")
    
    print(f"✓ Saved COLMAP cameras.txt: {output_path}")


def write_colmap_cameras_bin(output_path, camera_id, model, width, height, params):
    """
    Write COLMAP cameras.bin in binary format.
    
    Binary format:
        num_cameras (uint64)
        For each camera:
            camera_id (uint64)
            model_id (int32)
            width (uint64)
            height (uint64)
            num_params (uint64)
            params (double[num_params])
    """
    # COLMAP model name to ID mapping
    MODEL_NAME_TO_ID = {
        'SIMPLE_PINHOLE': 0,
        'PINHOLE': 1,
        'SIMPLE_RADIAL': 2,
        'RADIAL': 3,
        'OPENCV': 4,
        'OPENCV_FISHEYE': 5,
        'FULL_OPENCV': 6,
        'FOV': 7,
        'SIMPLE_RADIAL_FISHEYE': 8,
        'RADIAL_FISHEYE': 9,
        'THIN_PRISM_FISHEYE': 10
    }
    
    model_id = MODEL_NAME_TO_ID.get(model)
    if model_id is None:
        raise ValueError(f"Unknown camera model: {model}")
    
    with open(output_path, 'wb') as f:
        # Write number of cameras
        f.write(struct.pack('<Q', 1))  # uint64: 1 camera
        
        # Write camera data
        f.write(struct.pack('<i', camera_id))  # int32: camera_id
        f.write(struct.pack('<i', model_id))   # int32: model_id
        f.write(struct.pack('<Q', width))      # uint64: width
        f.write(struct.pack('<Q', height))     # uint64: height
        
        # Write params (no num_params field - model_id determines count)
        for param in params:
            f.write(struct.pack('<d', param))   # double: param value
    
    print(f"✓ Saved COLMAP cameras.bin: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert COLMAP intrinsics for MASt3R-SLAM and LichtFeld splat"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., soneva_reef)'
    )
    
    args = parser.parse_args()
    
    # Construct paths from dataset name
    dataset_root = INTERMEDIATE_DATA_ROOT / args.dataset
    colmap_cameras = dataset_root / 'colmap_outputs' / 'cameras.txt'
    
    if not colmap_cameras.exists():
        raise FileNotFoundError(
            f"COLMAP cameras.txt not found at: {colmap_cameras}\n"
            f"Expected path: /home/bwilliams/encode/data/intermediate_data/{args.dataset}/colmap_outputs/cameras.txt"
        )
    
    # Read COLMAP intrinsics
    print(f"\nReading COLMAP cameras.txt: {colmap_cameras}")
    cam = read_colmap_cameras_txt(colmap_cameras)
    
    print(f"  Camera ID: {cam['camera_id']}")
    print(f"  Model: {cam['model']}")
    print(f"  Resolution: {cam['width']}x{cam['height']}")
    print(f"  Params: {cam['params']}")
    
    if cam['model'] != 'OPENCV':
        raise ValueError(f"Expected OPENCV model, got {cam['model']}")
    
    # Extract intrinsics
    raw_width = cam['width']
    raw_height = cam['height']
    fx, fy, cx, cy = cam['params'][:4]
    distortion = cam['params'][4:]  # k1, k2, p1, p2
    
    K_raw = opencv_params_to_matrix(raw_width, raw_height, cam['params'])
    
    # Setup output paths
    yaml_output = dataset_root / 'intrinsics.yaml'
    splat_dir = dataset_root / 'for_splat' / 'sparse' / '0'
    splat_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1. MASt3R-SLAM intrinsics.yaml (no adjustment needed)
    # =========================================================================
    print(f"\n[1/3] Generating MASt3R-SLAM intrinsics.yaml...")
    write_mast3r_yaml(yaml_output, raw_width, raw_height, K_raw, distortion)
    
    # =========================================================================
    # 2. Adjusted intrinsics for splat (cameras.txt)
    # =========================================================================
    print(f"\n[2/3] Adjusting intrinsics for SLAM resolution ({SLAM_SIZE} with aspect ratio)...")
    K_scaled = scale_intrinsics(K_raw, raw_width, raw_height, SLAM_SIZE)
    
    # Get actual output dimensions from resize_img
    result, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
        np.zeros((raw_height, raw_width, 3)), SLAM_SIZE, return_transformation=True
    )
    resized_img = result['unnormalized_img']
    output_height, output_width = resized_img.shape[:2]
    
    print(f"  Output resolution: {output_width}x{output_height}")
    
    # Scaled params for COLMAP format
    scaled_params = [
        K_scaled[0, 0],  # fx
        K_scaled[1, 1],  # fy
        K_scaled[0, 2],  # cx
        K_scaled[1, 2],  # cy
        *distortion      # k1, k2, p1, p2 (unchanged)
    ]
    
    print(f"  Scaled intrinsics:")
    print(f"    fx={scaled_params[0]:.2f}, fy={scaled_params[1]:.2f}")
    print(f"    cx={scaled_params[2]:.2f}, cy={scaled_params[3]:.2f}")
    
    cameras_txt = splat_dir / 'cameras.txt'
    write_colmap_cameras_txt(
        cameras_txt,
        cam['camera_id'],
        cam['model'],
        output_width,
        output_height,
        scaled_params
    )
    
    # =========================================================================
    # 3. Adjusted intrinsics for splat (cameras.bin)
    # =========================================================================
    print(f"\n[3/3] Generating COLMAP binary format...")
    cameras_bin = splat_dir / 'cameras.bin'
    write_colmap_cameras_bin(
        cameras_bin,
        cam['camera_id'],
        cam['model'],
        output_width,
        output_height,
        scaled_params
    )
    
    print(f"\n{'='*70}")
    print(f"✅ SUCCESS: Intrinsics conversion complete")
    print(f"{'='*70}")
    print(f"\n📁 Output files:")
    print(f"   {yaml_output}")
    print(f"   {cameras_txt}")
    print(f"   {cameras_bin}")
    print(f"\n📝 Next steps:")
    print(f"   1. Use intrinsics.yaml with MASt3R-SLAM")
    print(f"   2. Use for_splat/sparse/0/cameras.bin with LichtFeld Studio")
    print()


if __name__ == '__main__':
    main()
