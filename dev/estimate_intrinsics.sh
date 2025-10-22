#!/bin/bash
# estimate_intrinsics.sh
# Fast self-calibration: extract intrinsics from RAW images using COLMAP
# Uses a subset of images with sequential matcher for speed

# TODO: add fallback to OPENCV_FISHEYE if <20 images registered

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail

# --- Parse Arguments ---
IMAGES_PATH=""
OUT_DIR=""
DATASET_NAME=""  # Optional: auto-generate output dir name
NUM_IMAGES=100  # Default: use first 100 images
CAMERA_MODEL="OPENCV"  # Default for GoPro/wide-angle. Use OPENCV_FISHEYE for extreme fisheye
OVERWRITE=false  # Default: prompt before overwriting

usage() {
    echo "Usage: $0 --images_path <path> [--out_dir <path>] [--dataset_name <name>] [--num_images N] [--camera_model MODEL] [--overwrite]"
    echo ""
    echo "Options:"
    echo "  --images_path    Path to directory with RAW images (PNG/JPG)"
    echo "  --out_dir        Output directory where cameras.txt will be saved"
    echo "                   (if not specified, will be auto-generated from dataset_name)"
    echo "  --dataset_name   Dataset name (e.g., 'soneva_reef'). Auto-generates output dir:"
    echo "                   ./calibration_<dataset_name>"
    echo "  --num_images     Number of images to use for calibration (default: 100)"
    echo "  --camera_model   OPENCV (default) or OPENCV_FISHEYE (extreme fisheye)"
    echo "  --overwrite      Automatically overwrite existing output directory"
    echo ""
    echo "Examples:"
    echo "  # Auto-generate output dir from dataset name:"
    echo "  $0 --images_path /data/soneva/raw --dataset_name soneva_reef"
    echo ""
    echo "  # Specify custom output dir:"
    echo "  $0 --images_path /data/raw_images --out_dir /data/calibration"
    echo ""
    echo "  # Overwrite existing calibration without prompt:"
    echo "  $0 --images_path /data/raw --dataset_name soneva_reef --overwrite"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --images_path)
            IMAGES_PATH="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --num_images)
            NUM_IMAGES="$2"
            shift 2
            ;;
        --camera_model)
            CAMERA_MODEL="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            usage
            ;;
    esac
done

if [[ -z "$IMAGES_PATH" ]]; then
    echo "ERROR: --images_path is required"
    usage
fi

# Auto-generate output directory from dataset name if not explicitly provided
if [[ -z "$OUT_DIR" ]]; then
    if [[ -z "$DATASET_NAME" ]]; then
        echo "ERROR: Either --out_dir or --dataset_name must be specified"
        usage
    fi
    # Create output dir under intermediate_data: /home/bwilliams/encode/data/intermediate_data/<dataset>
    OUT_DIR="/home/bwilliams/encode/data/intermediate_data/${DATASET_NAME}"
fi

if [[ ! -d "$IMAGES_PATH" ]]; then
    echo "ERROR: Images path does not exist: $IMAGES_PATH"
    exit 1
fi

# Validate camera model
if [[ "$CAMERA_MODEL" != "OPENCV" && "$CAMERA_MODEL" != "OPENCV_FISHEYE" ]]; then
    echo "ERROR: camera_model must be OPENCV or OPENCV_FISHEYE"
    exit 1
fi

# Handle existing output directory
if [[ -d "$OUT_DIR" ]]; then
    if [[ "$OVERWRITE" == "true" ]]; then
        echo "⚠️  Removing existing calibration directory: ${OUT_DIR}"
        rm -rf "${OUT_DIR}"
    else
        echo ""
        echo "⚠️  WARNING: Output directory already exists: ${OUT_DIR}"
        echo ""
        echo "This directory contains previous calibration results."
        echo "To ensure clean results, it should be removed."
        echo ""
        read -p "Delete existing directory and continue? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing ${OUT_DIR}..."
            rm -rf "${OUT_DIR}"
        else
            echo "Aborted. Use --overwrite flag to skip this prompt."
            exit 1
        fi
    fi
fi

echo "================================="
echo "COLMAP Intrinsics Calibration"
echo "================================="
echo "Images path:   $IMAGES_PATH"
echo "Output dir:    $OUT_DIR"
if [[ -n "$DATASET_NAME" ]]; then
    echo "Dataset name:  $DATASET_NAME"
fi
echo "Num images:    $NUM_IMAGES"
echo "Camera model:  $CAMERA_MODEL"
echo "Overwrite:     $OVERWRITE"
echo ""

# --- Environment Setup ---
# Clean snap/conda variables that interfere with COLMAP
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "⚠️  Deactivating Conda environment..."
    conda deactivate 2>/dev/null || true
fi
unset XDG_DATA_DIRS XDG_CONFIG_DIRS GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR GTK_PATH LOCPATH 2>/dev/null || true

# --- Create Work Directory ---
WORK_DIR="${OUT_DIR}/colmap_outputs"
DB_PATH="${WORK_DIR}/database.db"
SPARSE_DIR="${WORK_DIR}/sparse"
IMAGE_SUBSET_DIR="${WORK_DIR}/images_subset"

mkdir -p "${WORK_DIR}"
mkdir -p "${SPARSE_DIR}"
mkdir -p "${IMAGE_SUBSET_DIR}"

echo "Working directory: ${WORK_DIR}"

# --- Step 1: Create Subset of Images ---
echo ""
echo "[1/5] Creating subset of $NUM_IMAGES images..."
# Find image files (more robust than glob expansion)
IMAGE_FILES=()
while IFS= read -r -d '' file; do
    IMAGE_FILES+=("$file")
done < <(find "${IMAGES_PATH}" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 | sort -z)

TOTAL_IMAGES=${#IMAGE_FILES[@]}

if [[ $TOTAL_IMAGES -eq 0 ]]; then
    echo "ERROR: No images found in $IMAGES_PATH"
    exit 1
fi

echo "Found $TOTAL_IMAGES total images"

# Take first N images (consecutive frames for good overlap)
if [[ $TOTAL_IMAGES -le $NUM_IMAGES ]]; then
    SELECTED_IMAGES=("${IMAGE_FILES[@]}")
    echo "Using all $TOTAL_IMAGES images (less than requested $NUM_IMAGES)"
else
    SELECTED_IMAGES=("${IMAGE_FILES[@]:0:$NUM_IMAGES}")
    echo "Selected first $NUM_IMAGES images (consecutive frames for overlap)"
fi

# Symlink selected images
for img in "${SELECTED_IMAGES[@]}"; do
    ln -sf "$img" "${IMAGE_SUBSET_DIR}/$(basename "$img")"
done

echo "✓ Image subset ready"

# --- Step 2: Create COLMAP Database ---
echo ""
echo "[2/5] Creating COLMAP database..."
colmap database_creator --database_path "${DB_PATH}"
echo "✓ Database created"

# --- Step 3: Feature Extraction ---
echo ""
echo "[3/5] Extracting features..."
colmap feature_extractor \
    --database_path "${DB_PATH}" \
    --image_path "${IMAGE_SUBSET_DIR}" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model "${CAMERA_MODEL}" \
    --SiftExtraction.use_gpu 1

echo "✓ Features extracted"

# --- Step 4: Sequential Matching ---
echo ""
echo "[4/5] Matching features (sequential matcher)..."
colmap sequential_matcher \
    --database_path "${DB_PATH}" \
    --SiftMatching.use_gpu 1

echo "✓ Matching complete"

# --- Step 5: Mapper (Bundle Adjustment) ---
echo ""
echo "[5/5] Running bundle adjustment to refine intrinsics..."
colmap mapper \
    --database_path "${DB_PATH}" \
    --image_path "${IMAGE_SUBSET_DIR}" \
    --output_path "${SPARSE_DIR}" \
    --Mapper.ba_refine_focal_length 1 \
    --Mapper.ba_refine_principal_point 1 \
    --Mapper.ba_refine_extra_params 1 \
    --Mapper.num_threads 8

if [[ ! -d "${SPARSE_DIR}/0" ]]; then
    echo "❌ ERROR: COLMAP mapper failed to create sparse/0"
    echo ""
    echo "Troubleshooting tips:"
    echo "  - Try increasing --num_images (e.g., 150 or 200)"
    echo "  - If GoPro/wide-angle, ensure --camera_model OPENCV_FISHEYE"
    echo "  - Check images have sufficient overlap and features"
    exit 1
fi

echo "✓ Bundle adjustment complete"

# --- Export to Text Format ---
echo ""
echo "Exporting to text format..."
colmap model_converter \
    --input_path "${SPARSE_DIR}/0" \
    --output_path "${WORK_DIR}" \
    --output_type TXT

if [[ ! -f "${WORK_DIR}/cameras.txt" ]]; then
    echo "❌ ERROR: Failed to export cameras.txt"
    exit 1
fi

# --- Generate Summary File ---
SUMMARY_FILE="${WORK_DIR}/calibration_summary.txt"
echo ""
echo "Generating calibration summary..."

# Extract camera parameters
CAMERA_LINE=$(grep -v "^#" "${WORK_DIR}/cameras.txt" | head -n1)
CAMERA_ID=$(echo "$CAMERA_LINE" | awk '{print $1}')
MODEL=$(echo "$CAMERA_LINE" | awk '{print $2}')
WIDTH=$(echo "$CAMERA_LINE" | awk '{print $3}')
HEIGHT=$(echo "$CAMERA_LINE" | awk '{print $4}')
PARAMS=$(echo "$CAMERA_LINE" | cut -d' ' -f5-)

# Count registered images (images.txt has 2 lines per image, so divide by 2)
NUM_REGISTERED=$(grep -v "^#" "${WORK_DIR}/images.txt" | wc -l)
NUM_REGISTERED=$(awk "BEGIN {print int(${NUM_REGISTERED}/2)}")

# Count total 3D points
NUM_POINTS=$(grep -v "^#" "${WORK_DIR}/points3D.txt" | grep -c "^[0-9]" || echo "0")

# Calculate registration percentage
REGISTRATION_PCT=$(awk "BEGIN {printf \"%.1f\", 100*${NUM_REGISTERED}/${NUM_IMAGES}}")

# Create summary file
cat > "${SUMMARY_FILE}" << EOF
================================================================================
COLMAP Camera Calibration Summary
================================================================================
Generated: $(date)
Dataset: ${DATASET_NAME:-N/A}

Camera Model:  ${CAMERA_MODEL}
Resolution:    ${WIDTH} x ${HEIGHT}
Images Used:   ${NUM_IMAGES}
Registered:    ${NUM_REGISTERED} (${REGISTRATION_PCT}%)
3D Points:     ${NUM_POINTS}

--------------------------------------------------------------------------------
Calibrated Intrinsics (cameras.txt)
--------------------------------------------------------------------------------
Camera ID: ${CAMERA_ID}
Model:     ${MODEL}
Parameters: ${PARAMS}

EOF

# Add model-specific parameter explanation
if [[ "$CAMERA_MODEL" == "OPENCV" ]]; then
    cat >> "${SUMMARY_FILE}" << EOF
Format: fx, fy, cx, cy, k1, k2, p1, p2
  fx, fy = focal length (pixels)
  cx, cy = principal point
  k1, k2 = radial distortion
  p1, p2 = tangential distortion

EOF
elif [[ "$CAMERA_MODEL" == "OPENCV_FISHEYE" ]]; then
    cat >> "${SUMMARY_FILE}" << EOF
Format: fx, fy, cx, cy, k1, k2, k3, k4
  fx, fy = focal length (pixels)
  cx, cy = principal point
  k1-k4  = fisheye distortion

EOF
fi

cat >> "${SUMMARY_FILE}" << EOF
NOTE: This calibration is only for extracting intrinsics from cameras.txt.
      The images.txt and points3D.txt files contain temporary calibration
      poses/points and should be ignored (will be replaced by SLAM data).

--------------------------------------------------------------------------------
Registered Images (${NUM_REGISTERED} total)
--------------------------------------------------------------------------------
EOF

# Add list of registered images (extract image names from images.txt)
grep -v "^#" "${WORK_DIR}/images.txt" | awk 'NR%2==1 {print "  " $NF}' | sort >> "${SUMMARY_FILE}"

cat >> "${SUMMARY_FILE}" << EOF

================================================================================
EOF

# --- Console Summary ---
echo ""
echo "================================="
echo "✅ SUCCESS: Calibration Complete"
echo "================================="
echo ""
echo "📊 Registration: ${NUM_REGISTERED}/${NUM_IMAGES} images (${REGISTRATION_PCT}%)"
echo "🔍 Model: ${CAMERA_MODEL}"
echo "📐 Resolution: ${WIDTH}x${HEIGHT}"
echo "🎯 3D Points: ${NUM_POINTS}"
echo ""
echo "📁 Output files:"
echo "   ${WORK_DIR}/cameras.txt"
echo "   ${WORK_DIR}/images.txt"
echo "   ${WORK_DIR}/points3D.txt"
echo "   ${WORK_DIR}/calibration_summary.txt  ⭐ (detailed report)"
echo ""
echo "📄 View full summary:"
echo "   cat ${SUMMARY_FILE}"
echo ""
