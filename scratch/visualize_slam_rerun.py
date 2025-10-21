#TODO: this was allowing viewing with camera poses overload but lost this, think I was pressing
# wrong buttons on the gui!


#!/usr/bin/env python3
"""
visualize_slam_rerun.py

Visualize MASt3R-SLAM outputs using Rerun Viewer:
- 3D point cloud from .ply file
- Camera poses from COLMAP-format images.txt
- Camera intrinsics from COLMAP-format cameras.txt
- Optional: keyframe images

Usage:
    python visualize_slam_rerun.py --dataset reef_soneva
    python visualize_slam_rerun.py --dataset reef_soneva --no-images  # skip loading images
    python visualize_slam_rerun.py --dataset reef_soneva --downsample 0.1  # show 10% of points
"""

import argparse
import numpy as np
from pathlib import Path
from plyfile import PlyData
import rerun as rr
from PIL import Image

# Hardcoded paths
MSLAM_ROOT = Path('/home/bwilliams/encode/code/MASt3R-SLAM')
INTERMEDIATE_DATA_ROOT = Path('/home/bwilliams/encode/data/intermediate_data')


def read_ply_point_cloud(ply_path, downsample=1.0):
    """
    Read point cloud from .ply file.
    
    Args:
        ply_path: Path to .ply file
        downsample: Fraction of points to keep (0.0-1.0). Use <1.0 for large clouds.
    
    Returns:
        positions: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB colors (0-255)
    """
    print(f"Loading point cloud: {ply_path}")
    plydata = PlyData.read(ply_path)
    
    # Extract vertex data
    vertex = plydata['vertex']
    positions = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    
    # Try to extract colors (RGB)
    has_color = all(prop in vertex for prop in ['red', 'green', 'blue'])
    if has_color:
        colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T
    else:
        # Default gray if no color
        colors = np.ones_like(positions) * 128
    
    # Downsample if requested
    if downsample < 1.0:
        n_points = len(positions)
        n_keep = int(n_points * downsample)
        indices = np.random.choice(n_points, n_keep, replace=False)
        positions = positions[indices]
        colors = colors[indices]
        print(f"  Downsampled: {n_points} → {n_keep} points ({downsample*100:.1f}%)")
    
    print(f"  Loaded {len(positions)} points")
    return positions, colors


def parse_colmap_cameras_txt(cameras_txt_path):
    """
    Parse COLMAP cameras.txt format.
    
    Format:
        # Comment lines
        CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
    
    Returns:
        Dict with camera parameters
    """
    with open(cameras_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            
            # For OPENCV model: fx, fy, cx, cy, k1, k2, p1, p2
            return {
                'camera_id': camera_id,
                'model': model,
                'width': width,
                'height': height,
                'fx': params[0],
                'fy': params[1],
                'cx': params[2],
                'cy': params[3],
                'distortion': params[4:] if len(params) > 4 else []
            }
    
    raise ValueError(f"No camera found in {cameras_txt_path}")


def parse_colmap_images_txt(images_txt_path):
    """
    Parse COLMAP images.txt format.
    
    Format (2 lines per image):
        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        (empty or POINTS2D line)
    
    Returns:
        List of dicts with image_id, quaternion, translation, camera_id, name
    """
    images = []
    
    with open(images_txt_path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#')]
    
    # Process every 2 lines (image data + points2D/empty line)
    # Filter out completely empty lines for robustness
    non_empty_lines = [line for line in lines if line]
    
    for line in non_empty_lines:
        parts = line.split()
        
        # Skip if this looks like a POINTS2D line (doesn't start with IMAGE_ID)
        if len(parts) < 10:
            continue
        
        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]
        
        images.append({
            'image_id': image_id,
            'quaternion': np.array([qw, qx, qy, qz]),  # COLMAP format: QW first
            'translation': np.array([tx, ty, tz]),
            'camera_id': camera_id,
            'name': name
        })
    
    print(f"Loaded {len(images)} camera poses")
    return images


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        q: Quaternion as [qw, qx, qy, qz] (COLMAP format)
    
    Returns:
        3x3 rotation matrix
    """
    qw, qx, qy, qz = q
    
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R


def log_camera_to_rerun(entity_path, image_data, camera_params, images_dir=None, load_images=True):
    """
    Log a single camera pose to Rerun.
    
    Args:
        entity_path: Rerun entity path (e.g., "world/cameras/frame_0")
        image_data: Dict from parse_colmap_images_txt
        camera_params: Dict from parse_colmap_cameras_txt
        images_dir: Path to directory with keyframe images (optional)
        load_images: Whether to load and display the actual images
    """
    # 1. Transform3D: camera pose (world→camera in COLMAP)
    #    Rerun expects camera→world, so we need to invert the COLMAP pose
    q_cw = image_data['quaternion']  # world→camera rotation (COLMAP)
    t_cw = image_data['translation']  # world→camera translation (COLMAP)
    
    # Convert to camera→world for Rerun
    R_cw = quaternion_to_rotation_matrix(q_cw)
    R_wc = R_cw.T  # Invert rotation
    t_wc = -R_wc @ t_cw  # Invert translation
    
    # Log the camera transform (camera→world)
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=t_wc,
            mat3x3=R_wc
        )
    )
    
    # 2. Pinhole: camera intrinsics (defines frustum shape)
    rr.log(
        f"{entity_path}/pinhole",
        rr.Pinhole(
            resolution=[camera_params['width'], camera_params['height']],
            focal_length=[camera_params['fx'], camera_params['fy']],
            principal_point=[camera_params['cx'], camera_params['cy']]
        )
    )
    
    # 3. Optional: load and display the actual image
    if load_images and images_dir is not None:
        image_path = images_dir / image_data['name']
        if image_path.exists():
            try:
                img = Image.open(image_path)
                img_array = np.array(img)
                rr.log(
                    f"{entity_path}/pinhole",
                    rr.Image(img_array)
                )
            except Exception as e:
                print(f"  ⚠️  Failed to load image {image_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MASt3R-SLAM outputs with Rerun Viewer"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., reef_soneva)'
    )
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Skip loading keyframe images (faster, shows only camera frustums)'
    )
    parser.add_argument(
        '--downsample',
        type=float,
        default=1.0,
        help='Downsample point cloud (0.0-1.0). Use 0.1 for 10%% of points. Default: 1.0 (all points)'
    )
    
    args = parser.parse_args()
    
    # Construct paths
    ply_path = MSLAM_ROOT / 'logs' / f'{args.dataset}.ply'
    dataset_root = INTERMEDIATE_DATA_ROOT / args.dataset / 'for_splat'
    sparse_dir = dataset_root / 'sparse' / '0'
    images_dir = dataset_root / 'images'
    
    cameras_txt = sparse_dir / 'cameras.txt'
    images_txt = sparse_dir / 'images.txt'
    
    # Validate inputs
    if not ply_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {ply_path}")
    if not cameras_txt.exists():
        raise FileNotFoundError(f"Cameras file not found: {cameras_txt}")
    if not images_txt.exists():
        raise FileNotFoundError(f"Images file not found: {images_txt}")
    
    print(f"\n{'='*70}")
    print(f"Rerun Visualization: {args.dataset}")
    print(f"{'='*70}")
    print(f"Point cloud:  {ply_path}")
    print(f"Camera poses: {images_txt}")
    print(f"Intrinsics:   {cameras_txt}")
    print(f"Images:       {images_dir if not args.no_images else 'SKIPPED'}")
    print(f"Downsample:   {args.downsample*100:.1f}%")
    print()
    
    # Initialize Rerun
    rr.init(f"mast3r_slam_{args.dataset}", spawn=True)
    
    # Load and log point cloud
    print("[1/3] Loading point cloud...")
    positions, colors = read_ply_point_cloud(ply_path, downsample=args.downsample)
    rr.log("world/point_cloud", rr.Points3D(positions=positions, colors=colors))
    print("  ✓ Point cloud logged")
    
    # Load camera parameters
    print("\n[2/3] Loading camera intrinsics...")
    camera_params = parse_colmap_cameras_txt(cameras_txt)
    print(f"  Camera: {camera_params['width']}x{camera_params['height']}")
    print(f"  Focal:  fx={camera_params['fx']:.1f}, fy={camera_params['fy']:.1f}")
    print(f"  Model:  {camera_params['model']}")
    
    # Load and log camera poses
    print("\n[3/3] Loading camera poses...")
    images_data = parse_colmap_images_txt(images_txt)
    
    load_images = not args.no_images
    if load_images and not images_dir.exists():
        print(f"  ⚠️  Images directory not found: {images_dir}")
        load_images = False
    
    for i, img_data in enumerate(images_data):
        # Set timeline for each frame
        rr.set_time("frame", sequence=i)
        
        # Log camera pose + intrinsics + optional image
        entity_path = f"world/cameras/frame_{i:03d}"
        log_camera_to_rerun(
            entity_path,
            img_data,
            camera_params,
            images_dir if load_images else None,
            load_images
        )
        
        if (i + 1) % 10 == 0 or i == len(images_data) - 1:
            print(f"  Logged {i + 1}/{len(images_data)} cameras...")
    
    print("\n" + "="*70)
    print("✅ SUCCESS: Visualization loaded in Rerun Viewer")
    print("="*70)
    print("\n📊 What to check:")
    print("  1. Camera frustums should be positioned around the point cloud")
    print("  2. Cameras should 'look into' the scene (not away from it)")
    print("  3. Scale should be consistent (no huge/tiny cameras)")
    print("  4. Use timeline slider to step through camera sequence")
    if load_images:
        print("  5. Click on camera frustums to see the actual keyframe images")
    print("\n💡 Navigation:")
    print("  - Left click + drag: Rotate view")
    print("  - Right click + drag: Pan view")
    print("  - Scroll: Zoom in/out")
    print("  - Use timeline at bottom to step through frames")
    print()


if __name__ == '__main__':
    main()
