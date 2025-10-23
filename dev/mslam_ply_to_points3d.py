#!/usr/bin/env python3
"""
mslam_ply_to_points3d.py

Convert MASt3R-SLAM PLY point cloud to COLMAP points3D.bin format
for initialization in LichtFeld-Studio.

Uses wildflow.splat to convert simple point cloud (x,y,z,r,g,b) from
MASt3R-SLAM into COLMAP binary format that LichtFeld can use for
gaussian initialization instead of random points.

Usage:
    conda activate mast3r-slam
    python mslam_ply_to_points3d.py --dataset reef_soneva
    python mslam_ply_to_points3d.py --dataset truck_slam_splat --sample 5.0

Why sample_percentage?
    - MASt3R-SLAM outputs millions of points (5-10M+)
    - LichtFeld doesn't need that many for initialization
    - 10% gives ~500K-1M points which is plenty
    - Faster processing, less memory, good spatial coverage
"""

import argparse
import wildflow.splat as splat
from pathlib import Path

# Hardcoded paths
MSLAM_ROOT = Path('/home/bwilliams/encode/code/MASt3R-SLAM')
INTERMEDIATE_DATA_ROOT = Path('/home/bwilliams/encode/data/intermediate_data')


def main():
    parser = argparse.ArgumentParser(
        description="Convert MASt3R-SLAM PLY to COLMAP points3D.bin"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., reef_soneva, truck_slam_splat)'
    )
    parser.add_argument(
        '--sample',
        type=float,
        default=10.0,
        help='Sample percentage (default: 10.0). Lower = fewer points, faster. Higher = more detail.'
    )
    parser.add_argument(
        '--mslam_logs_dir',
        type=str,
        default=None,
        help='Path to MASt3R-SLAM logs directory (default: auto-detect from intermediate_data or fallback to MASt3R-SLAM/logs)'
    )
    
    args = parser.parse_args()
    
    # Determine MASt3R-SLAM logs location
    if args.mslam_logs_dir:
        mslam_logs = Path(args.mslam_logs_dir)
    else:
        # Try new location first (intermediate_data)
        new_location = INTERMEDIATE_DATA_ROOT / args.dataset / 'mslam_logs'
        if new_location.exists():
            mslam_logs = new_location
        else:
            # Fallback to old location (backward compatibility)
            mslam_logs = MSLAM_ROOT / 'logs'
    
    # Construct paths
    input_ply = mslam_logs / f'{args.dataset}.ply'
    output_dir = INTERMEDIATE_DATA_ROOT / args.dataset / 'for_splat' / 'sparse' / '0'
    output_bin = output_dir / 'points3D.bin'
    
    # Validate input
    if not input_ply.exists():
        raise FileNotFoundError(f"PLY file not found: {input_ply}")
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Converting PLY to COLMAP points3D.bin")
    print(f"{'='*70}")
    print(f"Dataset:        {args.dataset}")
    print(f"Input PLY:      {input_ply}")
    print(f"Output:         {output_bin}")
    print(f"Sample %:       {args.sample}%")
    print()
    
    # Convert using wildflow
    config = {
        "input_file": str(input_ply),
        "min_z": -float('inf'),
        "max_z": float('inf'),
        "sample_percentage": args.sample,
        "patches": [{
            "output_file": str(output_bin),
            "min_x": -float('inf'),
            "min_y": -float('inf'),
            "max_x": float('inf'),
            "max_y": float('inf')
        }]
    }
    
    result = splat.split_point_cloud(config)
    
    print()
    print(f"{'='*70}")
    print(f"✅ SUCCESS")
    print(f"{'='*70}")
    print(f"Points loaded:   {result['points_loaded']:,}")
    print(f"Points written:  {result['total_points_written']:,}")
    print(f"Output size:     {output_bin.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    print(f"Next steps:")
    print(f"  Run LichtFeld-Studio WITHOUT --random flag to use this point cloud")
    print(f"  for gaussian initialization instead of random points.")
    print()


if __name__ == '__main__':
    main()
