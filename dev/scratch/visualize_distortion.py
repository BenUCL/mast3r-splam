#!/usr/bin/env python3
"""
Visualize distortion effects on reef keyframes.
Shows original vs undistorted to understand impact of k1=-0.105, k2=0.106
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def visualize_distortion(dataset='reef_soneva'):
    """Create side-by-side comparison of distorted vs undistorted images"""
    
    # Load intrinsics from cameras.txt
    intermediate_data = Path('/home/bwilliams/encode/data/intermediate_data')
    cameras_txt = intermediate_data / dataset / 'colmap_outputs' / 'cameras.txt'
    
    # Parse cameras.txt to get intrinsics
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            # Format: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy k1 k2 p1 p2
            width = int(parts[2])
            height = int(parts[3])
            fx, fy, cx, cy = map(float, parts[4:8])
            k1, k2, p1, p2 = map(float, parts[8:12])
            break
    
    print(f"\n{'='*70}")
    print(f"Distortion Visualization: {dataset}")
    print(f"{'='*70}")
    print(f"Raw Resolution: {width}x{height}")
    print(f"Focal Length: fx={fx:.2f}, fy={fy:.2f}")
    print(f"Principal Point: cx={cx:.2f}, cy={cy:.2f}")
    print(f"Radial Distortion: k1={k1:.6f}, k2={k2:.6f}")
    print(f"Tangential Distortion: p1={p1:.6f}, p2={p2:.6f}")
    print(f"Total Radial Magnitude: {abs(k1) + abs(k2):.6f}")
    print()
    
    # Build camera matrix and distortion coefficients
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    
    # OpenCV expects dist = [k1, k2, p1, p2, k3, k4, ...] for 4-param OPENCV model
    dist = np.array([k1, k2, p1, p2], dtype=np.float32)
    
    # Get keyframe paths
    mslam_root = Path('/home/bwilliams/encode/code/MASt3R-SLAM')
    keyframes_dir = mslam_root / 'logs' / 'keyframes' / dataset
    
    if not keyframes_dir.exists():
        print(f"❌ Keyframes not found: {keyframes_dir}")
        return
    
    keyframes = sorted(keyframes_dir.glob('*.png'))
    print(f"Found {len(keyframes)} keyframes")
    
    # Process first, middle, and last keyframes
    indices = [0, len(keyframes) // 2, len(keyframes) - 1]
    
    output_dir = Path('/tmp/distortion_viz')
    output_dir.mkdir(exist_ok=True)
    
    for idx in indices:
        img_path = keyframes[idx]
        print(f"\nProcessing: {img_path.name}")
        
        # Read image (already resized to 512xH by MASt3R-SLAM)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Failed to load image")
            continue
        
        h, w = img.shape[:2]
        print(f"  Keyframe resolution: {w}x{h}")
        
        # Scale intrinsics to match keyframe resolution
        # (MASt3R-SLAM resizes from raw to 512xH, so we need to scale)
        scale_x = w / width
        scale_y = h / height
        
        K_scaled = K.copy()
        K_scaled[0, 0] *= scale_x  # fx
        K_scaled[1, 1] *= scale_y  # fy
        K_scaled[0, 2] *= scale_x  # cx
        K_scaled[1, 2] *= scale_y  # cy
        # distortion coefficients stay the same
        
        print(f"  Scaled intrinsics: fx={K_scaled[0,0]:.2f}, fy={K_scaled[1,1]:.2f}")
        
        # Undistort
        undistorted = cv2.undistort(img, K_scaled, dist)
        
        # Create side-by-side comparison
        comparison = np.hstack([img, undistorted])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original (Distorted)', (10, 30), 
                   font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(comparison, 'Undistorted', (w + 10, 30), 
                   font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw grid lines to visualize distortion
        for y in range(0, h, h // 8):
            cv2.line(comparison, (0, y), (w*2, y), (255, 0, 0), 1)
        for x in range(0, w, w // 8):
            cv2.line(comparison, (x, 0), (x, h), (255, 0, 0), 1)
            cv2.line(comparison, (w + x, 0), (w + x, h), (255, 0, 0), 1)
        
        # Save
        output_path = output_dir / f'{dataset}_frame_{idx:03d}_comparison.png'
        cv2.imwrite(str(output_path), comparison)
        print(f"  ✓ Saved: {output_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ Visualization complete!")
    print(f"{'='*70}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n💡 Open these images to see:")
    print(f"   - Left side: Original keyframe (as used by MASt3R-SLAM)")
    print(f"   - Right side: Undistorted version")
    print(f"   - Blue grid lines: Should be straight on right, curved on left")
    print(f"\n🔍 What to look for:")
    print(f"   - Barrel distortion (outward bulge) indicates strong k1")
    print(f"   - Pincushion distortion (inward pinch) indicates strong k2")
    print(f"   - If right side looks more 'correct', distortion IS the problem")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='reef_soneva', 
                       help='Dataset name (default: reef_soneva)')
    args = parser.parse_args()
    
    visualize_distortion(args.dataset)
