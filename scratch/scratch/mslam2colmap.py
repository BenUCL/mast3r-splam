# ~/slam_to_colmap.py
import pycolmap
import open3d as o3d
import numpy as np
import argparse
from pathlib import Path

def parse_tum_poses(pose_file, image_list):
    """Reads a TUM-style pose file and matches poses to images."""
    poses = {}
    with open(pose_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            timestamp = parts[0]
            tvec = np.array(parts[1:4], dtype=np.float64)
            qvec = np.array(parts[5:8] + [parts[4]], dtype=np.float64) # COLMAP uses w,x,y,z
            poses[timestamp] = (qvec, tvec)

    # Match poses to the sorted image list
    sorted_images = sorted(image_list, key=lambda p: float(Path(p).stem))
    matched_poses = {}
    for i, img_path in enumerate(sorted_images):
        timestamp = Path(img_path).stem
        if timestamp in poses:
            matched_poses[Path(img_path).name] = poses[timestamp]
        else:
            print(f"Warning: No pose found for image timestamp {timestamp}")

    print(f"Successfully matched {len(matched_poses)} poses to images.")
    return matched_poses

def main(args):
    # Define paths
    colmap_intrinsics_dir = Path(args.colmap_intrinsics_dir)
    images_dir = Path(args.images_dir)
    slam_pose_file = Path(args.slam_pose_file)
    slam_ply_file = Path(args.slam_ply_file)
    output_dir = Path(args.output_dir)
    
    output_sparse_dir = output_dir / "sparse"
    output_images_dir = output_dir / "images"

    # Create output directories
    output_sparse_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    # 1. Load the intrinsics from the temporary COLMAP model
    print("Loading camera intrinsics from COLMAP run...")
    rec_intrinsics = pycolmap.Reconstruction(colmap_intrinsics_dir)
    if not rec_intrinsics.cameras:
        raise ValueError("Could not load camera intrinsics. Did the COLMAP feature extraction fail?")
    camera = list(rec_intrinsics.cameras.values())[0]
    print(f"Loaded camera model: {camera.model} with params {camera.params}")

    # 2. Create the new, final reconstruction object
    rec_final = pycolmap.Reconstruction()
    rec_final.add_camera(camera)

    # 3. Copy images and add them to the reconstruction
    print(f"Processing images from: {images_dir}")
    image_paths = list(images_dir.glob('*'))
    for img_path in image_paths:
        target_path = output_images_dir / img_path.name
        if not target_path.exists():
            import shutil
            shutil.copy(str(img_path), str(target_path))

    # 4. Load SLAM poses and add them to the reconstruction
    print(f"Loading SLAM poses from: {slam_pose_file}")
    poses = parse_tum_poses(slam_pose_file, [str(p) for p in image_paths])
    
    image_id_map = {}
    for i, img_path in enumerate(sorted(image_paths, key=lambda p: float(p.stem))):
        img_name = img_path.name
        if img_name in poses:
            qvec, tvec = poses[img_name]
            # Use dict constructor - more stable API
            img = pycolmap.Image({
                'name': img_name,
                'camera_id': camera.camera_id,
                'image_id': i + 1,
                'cam_from_world': pycolmap.Rigid3d(pycolmap.Rotation3d(qvec), tvec)
            })
            rec_final.add_image(img)
            image_id_map[img_name] = img.image_id

    # 5. Load SLAM point cloud and add it to the reconstruction
    print(f"Loading SLAM point cloud from: {slam_ply_file}")
    if slam_ply_file.exists():
        pcd = o3d.io.read_point_cloud(str(slam_ply_file))
        points = np.asarray(pcd.points)
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        for i in range(len(points)):
            rec_final.add_point_3D(xyz=points[i], track=pycolmap.Track(), color=colors[i])
        print(f"Added {len(points)} 3D points.")

    # 6. Write the final COLMAP model
    rec_final.write(output_sparse_dir)
    print("✅ Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MASt3R-SLAM output to COLMAP format for Gaussian Splatting.")
    parser.add_argument("--colmap_intrinsics_dir", required=True, help="Path to the 'sparse/0' directory from the temporary COLMAP run.")
    parser.add_argument("--images_dir", required=True, help="Path to the directory of keyframe images from MASt3R-SLAM.")
    parser.add_argument("--slam_pose_file", required=True, help="Path to the SLAM pose file (e.g., reef_soneva.txt).")
    parser.add_argument("--slam_ply_file", required=True, help="Path to the SLAM point cloud file (e.g., reef_soneva.ply).")
    parser.add_argument("--output_dir", required=True, help="Path to the final output directory for the COLMAP project.")
    args = parser.parse_args()
    main(args)