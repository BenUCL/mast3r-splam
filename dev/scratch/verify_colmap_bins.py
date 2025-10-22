#!/usr/bin/env python3
"""
Verify COLMAP binary files are properly formatted.
Reads cameras.bin and images.bin and checks for corruption.
"""

import struct
import sys
from pathlib import Path

def read_cameras_bin(path):
    """Read and validate cameras.bin"""
    print(f"\n{'='*60}")
    print(f"Reading: {path}")
    print(f"{'='*60}")
    
    with open(path, 'rb') as f:
        # Read number of cameras
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        print(f"Number of cameras: {num_cameras}")
        
        for i in range(num_cameras):
            print(f"\n--- Camera {i+1} ---")
            
            # Read camera_id (int32, not uint64!)
            camera_id = struct.unpack('<i', f.read(4))[0]
            print(f"  Camera ID: {camera_id}")
            
            # Read model_id (int32)
            model_id = struct.unpack('<i', f.read(4))[0]
            print(f"  Model ID: {model_id} (4=OPENCV)")
            
            # Read width, height (uint64)
            width = struct.unpack('<Q', f.read(8))[0]
            height = struct.unpack('<Q', f.read(8))[0]
            print(f"  Resolution: {width}x{height}")
            
            # OPENCV model has 8 params
            num_params = 8
            params = []
            for j in range(num_params):
                param = struct.unpack('<d', f.read(8))[0]
                params.append(param)
            
            print(f"  Params (fx,fy,cx,cy,k1,k2,p1,p2):")
            print(f"    fx={params[0]:.4f}, fy={params[1]:.4f}")
            print(f"    cx={params[2]:.4f}, cy={params[3]:.4f}")
            print(f"    k1={params[4]:.6f}, k2={params[5]:.6f}")
            print(f"    p1={params[6]:.6f}, p2={params[7]:.6f}")
        
        # Check if we've reached EOF
        remaining = f.read()
        if remaining:
            print(f"\n⚠️  WARNING: {len(remaining)} unexpected bytes at end of file!")
        else:
            print(f"\n✓ File properly terminated")
    
    return True


def read_images_bin(path):
    """Read and validate images.bin"""
    print(f"\n{'='*60}")
    print(f"Reading: {path}")
    print(f"{'='*60}")
    
    with open(path, 'rb') as f:
        # Read number of images
        num_images = struct.unpack('<Q', f.read(8))[0]
        print(f"Number of images: {num_images}")
        
        for i in range(num_images):
            try:
                # Read image_id (uint64)
                image_id = struct.unpack('<Q', f.read(8))[0]
                
                # Read quaternion (4 doubles: qw, qx, qy, qz)
                qw = struct.unpack('<d', f.read(8))[0]
                qx = struct.unpack('<d', f.read(8))[0]
                qy = struct.unpack('<d', f.read(8))[0]
                qz = struct.unpack('<d', f.read(8))[0]
                
                # Read translation (3 doubles)
                tx = struct.unpack('<d', f.read(8))[0]
                ty = struct.unpack('<d', f.read(8))[0]
                tz = struct.unpack('<d', f.read(8))[0]
                
                # Read camera_id (uint64)
                camera_id = struct.unpack('<Q', f.read(8))[0]
                
                # Read image name (null-terminated string)
                name_bytes = b''
                while True:
                    byte = f.read(1)
                    if not byte or byte == b'\x00':
                        break
                    name_bytes += byte
                name = name_bytes.decode('utf-8')
                
                # Read num_points2D (uint64)
                num_points2D = struct.unpack('<Q', f.read(8))[0]
                
                # Read points2D (should be 0 for us)
                for j in range(num_points2D):
                    x = struct.unpack('<d', f.read(8))[0]
                    y = struct.unpack('<d', f.read(8))[0]
                    point3d_id = struct.unpack('<Q', f.read(8))[0]
                
                # Print first 5 and last 5 images
                if i < 5 or i >= num_images - 5:
                    print(f"\nImage {image_id}: {name}")
                    print(f"  Quat: [{qw:.4f}, {qx:.4f}, {qy:.4f}, {qz:.4f}]")
                    print(f"  Trans: [{tx:.4f}, {ty:.4f}, {tz:.4f}]")
                    print(f"  Camera: {camera_id}, Points2D: {num_points2D}")
                elif i == 5:
                    print(f"\n... (skipping images 6-{num_images-5}) ...")
                    
            except Exception as e:
                print(f"\n❌ ERROR reading image {i+1}/{num_images}: {e}")
                print(f"   File position: {f.tell()}")
                return False
        
        # Check if we've reached EOF
        remaining = f.read()
        if remaining:
            print(f"\n⚠️  WARNING: {len(remaining)} unexpected bytes at end of file!")
            print(f"   First 50 bytes: {remaining[:50]}")
        else:
            print(f"\n✓ File properly terminated")
    
    return True


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python verify_colmap_bins.py <sparse_dir>")
        print("Example: python verify_colmap_bins.py /path/to/sparse/0/")
        sys.exit(1)
    
    sparse_dir = Path(sys.argv[1])
    cameras_bin = sparse_dir / 'cameras.bin'
    images_bin = sparse_dir / 'images.bin'
    
    if not cameras_bin.exists():
        print(f"❌ cameras.bin not found: {cameras_bin}")
        sys.exit(1)
    
    if not images_bin.exists():
        print(f"❌ images.bin not found: {images_bin}")
        sys.exit(1)
    
    # Verify cameras.bin
    try:
        read_cameras_bin(cameras_bin)
    except Exception as e:
        print(f"\n❌ FATAL ERROR in cameras.bin: {e}")
        sys.exit(1)
    
    # Verify images.bin
    try:
        read_images_bin(images_bin)
    except Exception as e:
        print(f"\n❌ FATAL ERROR in images.bin: {e}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"✅ Both binary files are valid!")
    print(f"{'='*60}\n")
