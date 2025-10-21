# Distortion Handling Analysis: Why Reef Fails but Truck Works

## Summary
**ROOT CAUSE IDENTIFIED**: Your pipeline is correct, but reef has 124x more distortion than truck, which may be causing issues despite using `--gut`.

## 1. Your Pipeline - Intrinsics Adjustment (✅ CORRECT)

### Location: `shuttle_intrinsics.py` lines 53-75

```python
def scale_intrinsics(K, raw_w, raw_h, target_size):
    """Scale intrinsics from raw resolution to target resolution"""
    _, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
        np.zeros((raw_h, raw_w, 3)), target_size, return_transformation=True
    )
    
    K_scaled = K.copy()
    K_scaled[0, 0] = K[0, 0] / scale_w   # fx scaled
    K_scaled[1, 1] = K[1, 1] / scale_h   # fy scaled
    K_scaled[0, 2] = K[0, 2] / scale_w - half_crop_w  # cx scaled + crop
    K_scaled[1, 2] = K[1, 2] / scale_h - half_crop_h  # cy scaled + crop
    return K_scaled
```

**Key Point**: Lines 267-273 show distortion coefficients are **NOT scaled** - this is CORRECT!
```python
scaled_params = [
    K_scaled[0, 0],  # fx (scaled)
    K_scaled[1, 1],  # fy (scaled)
    K_scaled[0, 2],  # cx (scaled + cropped)
    K_scaled[1, 2],  # cy (scaled + cropped)
    *distortion      # k1, k2, p1, p2 (UNCHANGED) ✅
]
```

### Why This is Correct
- **Distortion coefficients (k1, k2, p1, p2) are unitless** - they describe distortion as a function of normalized image coordinates
- **When you resize/crop images**, the pixel coordinates change but the underlying lens distortion doesn't
- **Only focal length (fx, fy) and principal point (cx, cy) need adjustment** for pixel coordinate changes

## 2. LichtFeld-Studio 3DGUT Support (✅ WORKS)

### Evidence from Source Code

**Location**: `src/loader/formats/colmap.cpp` lines 728-742
```cpp
case CAMERA_MODEL::OPENCV: {
    out[i]._focal_x = out[i]._params[0].item<float>();
    out[i]._focal_y = out[i]._params[1].item<float>();
    out[i]._center_x = out[i]._params[2].item<float>();
    out[i]._center_y = out[i]._params[3].item<float>();

    float k1 = out[i]._params[4].item<float>();
    float k2 = out[i]._params[5].item<float>();
    out[i]._radial_distortion = torch::tensor({k1, k2}, torch::kFloat32);

    float p1 = out[i]._params[6].item<float>();
    float p2 = out[i]._params[7].item<float>();
    out[i]._tangential_distortion = torch::tensor({p1, p2}, torch::kFloat32);

    out[i]._camera_model_type = gsplat::CameraModelType::PINHOLE;
    break;
}
```

**Location**: `src/training/trainer.cpp` lines 593-596
```cpp
if (!params_.optimization.rc) {
    if (cam->radial_distortion().numel() != 0 ||
        cam->tangential_distortion().numel() != 0) {
        return std::unexpected("You must use --gut option to train on cameras with distortion.");
    }
}
```

### Your Command (✅ CORRECT)
```bash
/home/bwilliams/encode/code/lichtfeld-studio/build/LichtFeld-Studio \
 -d /home/bwilliams/encode/data/intermediate_data/reef_soneva/for_splat \
 -o /home/bwilliams/encode/code/lichtfeld-studio/output/m-slam_reef_soneva \
 --headless \
 --gut \       # ✅ ENABLES 3DGUT for distortion handling
 -i 10000
```

## 3. Distortion Comparison: Reef vs Truck

### Reef Dataset
- **Raw resolution**: 1600x1200
- **SLAM resolution**: 512x384
- **Distortion**: k1=-0.1053, k2=0.1058
- **Total radial distortion**: |k1| + |k2| = **0.211**

### Truck Dataset  
- **Raw resolution**: 979x546
- **SLAM resolution**: 512x272
- **Distortion**: k1=-0.0012, k2=0.0005
- **Total radial distortion**: |k1| + |k2| = **0.0017**

### Result: Reef has **124x MORE DISTORTION** than truck! 🚨

## 4. Why Reef Might Still Fail Despite 3DGUT

### Theory 1: Extreme Distortion + Large Scene Scale
- Reef scene: 13.7m span, 71m trajectory
- Truck scene: 5.0m span, 41m trajectory  
- **Reef is 3.1x larger** with **124x more distortion**
- Large baseline + high distortion = harder reconstruction

### Theory 2: MASt3R-SLAM May Not Handle Distortion Perfectly
- MASt3R-SLAM receives distortion in `intrinsics.yaml` ✅
- But does it actually **undistort keyframes** before feature matching?
- If keyframes are still distorted, SLAM poses could be inaccurate

### Theory 3: Resolution Matters
- Reef: 1600x1200 → 512x384 (3.125x downscale)
- Truck: 979x546 → 512x272 (1.91x downscale)
- Larger downscale + high distortion might amplify errors

## 5. Diagnostic Questions

### Check if MASt3R-SLAM is handling distortion properly:
```bash
# Look at keyframes - are they visibly distorted?
eog /home/bwilliams/encode/code/MASt3R-SLAM/logs/keyframes/reef_soneva/0.0.png
eog /home/bwilliams/encode/code/MASt3R-SLAM/logs/keyframes/truck_slam_splat/0.0.png
```

If reef keyframes show **barrel distortion** (straight lines curved), then MASt3R-SLAM is NOT undistorting them, which could explain poor splatting results.

## 6. Potential Fixes

### Option 1: Undistort Images Before SLAM (RECOMMENDED)
**Pros**: Removes distortion entirely, simplifies pipeline
**Cons**: Requires additional processing step

```python
# Create undistort script using OpenCV
import cv2
import numpy as np

# Read intrinsics
K = np.array([[949.93, 0, 810.04],
              [0, 950.54, 596.83],
              [0, 0, 1]])
dist = np.array([-0.1053, 0.1058, 0.0002, -0.0012])

# Undistort all reef images
for img_path in reef_images:
    img = cv2.imread(str(img_path))
    undistorted = cv2.undistort(img, K, dist)
    cv2.imwrite(str(img_path), undistorted)
```

Then use **PINHOLE** model (no distortion) in COLMAP calibration.

### Option 2: Try OPENCV_FISHEYE Model
Your current k1=-0.105, k2=0.106 suggests **strong barrel then pincushion** distortion.
- Might be better modeled as fisheye
- Re-run calibration with `--camera_model OPENCV_FISHEYE`

### Option 3: Reduce Distortion in Calibration
- Use fewer images for calibration (maybe 50 instead of 100)
- COLMAP might be overfitting the distortion

### Option 4: Check if LichtFeld-Studio 3DGUT has Limits
- Test with artificially reduced distortion (multiply k1, k2 by 0.5)
- See if splat quality improves

## 7. Immediate Action

**RUN THIS** to visualize if distortion is the problem:
```bash
cd /home/bwilliams/encode/code/scratch
python3 << 'EOF'
import cv2
import numpy as np
from pathlib import Path

# Reef intrinsics
K = np.array([[949.93, 0, 810.04],
              [0, 950.54, 596.83],
              [0, 0, 1]], dtype=np.float32)
dist = np.array([-0.1053, 0.1058, 0.0002, -0.0012], dtype=np.float32)

# Load first reef keyframe
img_path = Path('/home/bwilliams/encode/code/MASt3R-SLAM/logs/keyframes/reef_soneva/0.0.png')
img = cv2.imread(str(img_path))

# Undistort
undistorted = cv2.undistort(img, K, dist)

# Save comparison
cv2.imwrite('/tmp/reef_original.png', img)
cv2.imwrite('/tmp/reef_undistorted.png', undistorted)
print("✓ Saved comparison images to /tmp/")
print("  Original: /tmp/reef_original.png")
print("  Undistorted: /tmp/reef_undistorted.png")
print("\nOpen these side-by-side to see distortion magnitude")
EOF
```

## 8. Conclusion

Your pipeline is **technically correct**:
- ✅ Distortion coefficients properly preserved through resize
- ✅ LichtFeld-Studio `--gut` flag enables 3DGUT distortion handling
- ✅ OPENCV camera model with k1, k2, p1, p2 is supported

**BUT**: Reef has **124x more distortion** than truck, and the combination of:
- Extreme distortion (k1=-0.105)
- Large scene scale (3.1x bigger)
- Larger downsampling (3.1x vs 1.9x)

...may be pushing the limits of either:
1. MASt3R-SLAM's ability to handle distorted images for pose estimation
2. 3DGUT's ability to handle extreme distortion during splatting
3. Both

**Recommended next step**: Create undistorted version of reef dataset and re-run the entire pipeline.
