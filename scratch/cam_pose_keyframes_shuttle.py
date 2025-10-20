#!/usr/bin/env python3
"""
mslam2colmap.py

Convert MASt3R-SLAM keyframes and TUM formatted poses to COLMAP format for LichtFeld Studio.

This script:
1. Copies/symlinks keyframes to the LichtFeld images directory
2. Converts TUM poses (camera→world) to COLMAP format (world→camera)
3. Generates images.txt with proper COLMAP headers and pose format

Input:
  - MASt3R-SLAM keyframes: logs/keyframes/<dataset>/
  - TUM poses: logs/<dataset>.txt (timestamp tx ty tz qx qy qz qw)

Output:
  - Images: intermediate_data/<dataset>/for_splat/images/
  - COLMAP: intermediate_data/<dataset>/for_splat/sparse/0/images.txt
  - COLMAP: intermediate_data/<dataset>/for_splat/sparse/0/images.bin

Usage:
  python mslam2colmap.py --dataset reef_soneva
  python mslam2colmap.py --dataset reef_soneva --link  # symlink instead of copy
  python mslam2colmap.py --dataset reef_soneva --camera_id 2
"""

import argparse
import shutil
import struct
import numpy as np
from pathlib import Path


# Hardcoded paths
MSLAM_ROOT = Path('/home/bwilliams/encode/code/MASt3R-SLAM')
INTERMEDIATE_DATA_ROOT = Path('/home/bwilliams/encode/data/intermediate_data')


def quaternion_conjugate(q):
    """
    Compute quaternion conjugate.
    Input: (qx, qy, qz, qw)
    Output: (qw, -qx, -qy, -qz) normalized
    """
    qx, qy, qz, qw = q
    q_conj = np.array([qw, -qx, -qy, -qz])
    # Normalize
    q_conj = q_conj / np.linalg.norm(q_conj)
    return q_conj


def quat_to_rotation_matrix(q):
    """
    Convert quaternion to 3x3 rotation matrix.
    Input: (qx, qy, qz, qw)
    """
    qx, qy, qz, qw = q
    
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R


def tum_to_colmap_pose(t_wc, q_wc):
    """
    Convert TUM pose (camera→world) to COLMAP pose (world→camera).
    
    Args:
        t_wc: Translation vector [tx, ty, tz] (camera in world frame)
        q_wc: Quaternion (qx, qy, qz, qw) (camera→world rotation)
    
    Returns:
        q_cw: Quaternion (QW, QX, QY, QZ) for COLMAP (world→camera)
        t_cw: Translation vector [TX, TY, TZ] for COLMAP
    """
    # 1. Quaternion: conjugate and reorder to (qw, qx, qy, qz)
    q_cw = quaternion_conjugate(q_wc)
    
    # 2. Translation: t_cw = -R_wc^T * t_wc
    R_wc = quat_to_rotation_matrix(q_wc)
    t_cw = -R_wc.T @ t_wc
    
    return q_cw, t_cw


def parse_tum_poses(tum_file):
    """
    Parse TUM format poses file.
    
    Format: timestamp tx ty tz qx qy qz qw
    
    Returns:
        List of dicts: [{'timestamp': float, 'pose': (t_wc, q_wc)}, ...]
    """
    poses = []
    
    with open(tum_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) != 8:
                continue
            
            timestamp = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            
            t_wc = np.array([tx, ty, tz])
            q_wc = np.array([qx, qy, qz, qw])
            
            poses.append({
                'timestamp': timestamp,
                'pose': (t_wc, q_wc)
            })
    
    return poses


def read_camera_id_from_cameras_txt(cameras_txt):
    """Read camera ID from existing cameras.txt if present."""
    if not cameras_txt.exists():
        return None
    
    with open(cameras_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
            parts = line.split()
            return int(parts[0])
    
    return None


def write_colmap_images_txt(output_path, keyframe_files, poses, camera_id):
    """
    Write COLMAP images.txt format.
    
    Format:
        # Header
        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        (empty line - no 2D points)
    
    Returns:
        List of image data tuples for binary writing:
        [(image_id, q_cw, t_cw, camera_id, name), ...]
    """
    if len(keyframe_files) != len(poses):
        raise ValueError(
            f"Mismatch: {len(keyframe_files)} keyframes but {len(poses)} poses"
        )
    
    # Sort keyframes and poses by timestamp (keyframe filename is timestamp)
    keyframe_timestamps = []
    for kf in keyframe_files:
        # Extract timestamp from filename (e.g., "0.6000000238418579.png")
        ts = float(kf.stem)
        keyframe_timestamps.append((ts, kf))
    
    keyframe_timestamps.sort(key=lambda x: x[0])
    
    # Match poses to keyframes by timestamp
    pose_dict = {p['timestamp']: p['pose'] for p in poses}
    
    image_data = []
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(keyframe_files)}\n")
        
        for image_id, (ts, kf) in enumerate(keyframe_timestamps, start=1):
            if ts not in pose_dict:
                print(f"⚠️  Warning: No pose found for keyframe {kf.name} (timestamp={ts})")
                continue
            
            t_wc, q_wc = pose_dict[ts]
            
            # Convert TUM → COLMAP
            q_cw, t_cw = tum_to_colmap_pose(t_wc, q_wc)
            
            # Store for binary writing
            image_data.append((image_id, q_cw, t_cw, camera_id, kf.name))
            
            # Write image line
            # Format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            f.write(f"{image_id} ")
            f.write(f"{q_cw[0]} {q_cw[1]} {q_cw[2]} {q_cw[3]} ")  # QW QX QY QZ
            f.write(f"{t_cw[0]} {t_cw[1]} {t_cw[2]} ")            # TX TY TZ
            f.write(f"{camera_id} {kf.name}\n")
            
            # Write empty second line (no 2D points)
            f.write("\n")
    
    print(f"✓ Wrote {len(keyframe_timestamps)} images to {output_path}")
    return image_data


def write_colmap_images_bin(output_path, image_data):
    """
    Write COLMAP images.bin in binary format.
    
    Binary format:
        num_images (uint64)
        For each image:
            image_id (uint64)
            qw, qx, qy, qz (double[4])
            tx, ty, tz (double[3])
            camera_id (uint64)
            name (null-terminated string)
            num_points2D (uint64)
            points2D (x, y, point3D_id) - empty for us
    
    Args:
        image_data: List of tuples (image_id, q_cw, t_cw, camera_id, name)
    """
    with open(output_path, 'wb') as f:
        # Write number of images
        f.write(struct.pack('Q', len(image_data)))  # uint64
        
        for image_id, q_cw, t_cw, camera_id, name in image_data:
            # Write image_id
            f.write(struct.pack('Q', image_id))  # uint64
            
            # Write quaternion (qw, qx, qy, qz)
            f.write(struct.pack('d', q_cw[0]))  # QW
            f.write(struct.pack('d', q_cw[1]))  # QX
            f.write(struct.pack('d', q_cw[2]))  # QY
            f.write(struct.pack('d', q_cw[3]))  # QZ
            
            # Write translation (tx, ty, tz)
            f.write(struct.pack('d', t_cw[0]))  # TX
            f.write(struct.pack('d', t_cw[1]))  # TY
            f.write(struct.pack('d', t_cw[2]))  # TZ
            
            # Write camera_id
            f.write(struct.pack('Q', camera_id))  # uint64
            
            # Write image name (null-terminated string)
            name_bytes = name.encode('utf-8') + b'\x00'
            f.write(name_bytes)
            
            # Write num_points2D (0 - no 2D points)
            f.write(struct.pack('Q', 0))  # uint64
    
    print(f"✓ Wrote {len(image_data)} images to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MASt3R-SLAM outputs to COLMAP format for LichtFeld Studio"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., reef_soneva)'
    )
    parser.add_argument(
        '--link',
        action='store_true',
        help='Symlink keyframes instead of copying (default: copy)'
    )
    parser.add_argument(
        '--camera_id',
        type=int,
        default=None,
        help='Camera ID to use (default: auto-detect from cameras.txt or use 1)'
    )
    
    args = parser.parse_args()
    
    # Construct paths
    keyframes_src = MSLAM_ROOT / 'logs' / 'keyframes' / args.dataset
    tum_poses = MSLAM_ROOT / 'logs' / f'{args.dataset}.txt'
    
    dataset_root = INTERMEDIATE_DATA_ROOT / args.dataset
    images_dir = dataset_root / 'for_splat' / 'images'
    sparse_dir = dataset_root / 'for_splat' / 'sparse' / '0'
    
    # Validate inputs
    if not keyframes_src.exists():
        raise FileNotFoundError(f"Keyframes not found: {keyframes_src}")
    
    if not tum_poses.exists():
        raise FileNotFoundError(f"TUM poses not found: {tum_poses}")
    
    print(f"\n{'='*70}")
    print(f"MASt3R-SLAM → COLMAP Conversion")
    print(f"{'='*70}")
    print(f"Dataset:       {args.dataset}")
    print(f"Keyframes:     {keyframes_src}")
    print(f"TUM poses:     {tum_poses}")
    print(f"Output images: {images_dir}")
    print(f"Output sparse: {sparse_dir}")
    print(f"Link mode:     {args.link}")
    print()
    
    # Create output directories
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Copy or symlink keyframes
    print(f"[1/3] {'Symlinking' if args.link else 'Copying'} keyframes...")
    
    keyframe_files = sorted(keyframes_src.glob('*.png'))
    if not keyframe_files:
        raise ValueError(f"No PNG keyframes found in {keyframes_src}")
    
    for kf in keyframe_files:
        dst = images_dir / kf.name
        if dst.exists():
            dst.unlink()  # Remove existing
        
        if args.link:
            dst.symlink_to(kf)
        else:
            shutil.copy2(kf, dst)
    
    print(f"  ✓ Processed {len(keyframe_files)} keyframes")
    
    # Step 2: Parse TUM poses
    print(f"\n[2/3] Parsing TUM poses...")
    poses = parse_tum_poses(tum_poses)
    print(f"  ✓ Loaded {len(poses)} poses")
    
    # Step 3: Determine camera ID
    camera_id = args.camera_id
    if camera_id is None:
        cameras_txt = sparse_dir / 'cameras.txt'
        camera_id = read_camera_id_from_cameras_txt(cameras_txt)
        if camera_id is None:
            camera_id = 1  # Default
            print(f"  ⚠️  No cameras.txt found, using default CAMERA_ID=1")
        else:
            print(f"  ✓ Read CAMERA_ID={camera_id} from cameras.txt")
    else:
        print(f"  ✓ Using user-specified CAMERA_ID={camera_id}")
    
    # Step 4: Write images.txt
    print(f"\n[3/4] Writing COLMAP images.txt...")
    images_txt = sparse_dir / 'images.txt'
    image_data = write_colmap_images_txt(images_txt, keyframe_files, poses, camera_id)
    
    # Step 5: Write images.bin
    print(f"\n[4/4] Writing COLMAP images.bin...")
    images_bin = sparse_dir / 'images.bin'
    write_colmap_images_bin(images_bin, image_data)
    
    print(f"\n{'='*70}")
    print(f"✅ SUCCESS: COLMAP conversion complete")
    print(f"{'='*70}")
    print(f"\n📁 Output files:")
    print(f"   {images_dir}/ ({len(keyframe_files)} images)")
    print(f"   {images_txt}")
    print(f"   {images_bin}")
    print(f"\n📝 Next steps:")
    print(f"   1. Verify cameras.bin exists in {sparse_dir}/")
    print(f"   2. Use {dataset_root}/for_splat/ with LichtFeld Studio")
    print()


if __name__ == '__main__':
    main()