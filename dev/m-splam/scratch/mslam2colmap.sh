#!/bin/bash
# A script to convert MASt3R-SLAM output to COLMAP format

# Enable strict error handling
set -e  # Exit on any error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failures

# --- Configuration ---
DATASET_NAME="reef_soneva"
MAST3R_LOGS="/home/bwilliams/encode/code/MASt3R-SLAM/logs"
OUTPUT_PARENT_DIR="/home/bwilliams/encode/data/intermediate_data"
CONVERSION_SCRIPT="/home/bwilliams/encode/code/scratch/mslam2colmap.py"

# --- Derived Paths (usually no need to change) ---
KEYFRAME_IMAGES_DIR="${MAST3R_LOGS}/keyframes/${DATASET_NAME}"
SLAM_POSE_FILE="${MAST3R_LOGS}/${DATASET_NAME}.txt"
SLAM_PLY_FILE="${MAST3R_LOGS}/${DATASET_NAME}.ply"

FINAL_OUTPUT_DIR="${OUTPUT_PARENT_DIR}/${DATASET_NAME}_clmp"
COLMAP_TMP_DIR="${FINAL_OUTPUT_DIR}/colmap_tmp"


# --- Environment Cleanup ---
# Check for Conda conflicts
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "⚠️  WARNING: Conda environment detected (${CONDA_DEFAULT_ENV})"
    echo "⚠️  Conda can cause library conflicts with COLMAP. Deactivating conda..."
    conda deactivate
fi

# VS Code snap variables can cause library conflicts with COLMAP
# Unset all snap-related XDG variables that might interfere
echo "Cleaning snap environment variables..."
unset XDG_DATA_DIRS XDG_CONFIG_DIRS GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR GTK_PATH LOCPATH 2>/dev/null || true


# --- Activate UV Environment ---
echo "Activating uv environment..."
source /home/bwilliams/encode/code/ben-splat-env/bin/activate


# --- STEP A: Get Camera Intrinsics with COLMAP ---
echo "--- Starting Step A: Getting camera intrinsics with COLMAP ---"

# 1. Create a temporary database for COLMAP
mkdir -p "${CO_path "${COLMAP_TMP_DIR}/database.db" \
    --image_path "${KEYFRAME_IMAGES_DIR}" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV # OPENCV is robust for distortion

if [ $? -ne 0 ]; then
    echo "ERROR: Feature extraction failed"
    exit 1
fiLMAP_TMP_DIR}"
colmap database_creator --database_path "${COLMAP_TMP_DIR}/database.db"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create COLMAP database"
    exit 1
fi

# 2. Run feature extraction
# We tell COLM_path "${COLMAP_TMP_DIR}/database.db" \
    --image_path "${KEYFRAME_IMAGES_DIR}" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV # OPENCV is robust for distortion

if [ $? -ne 0 ]; then
    echo "ERROR: Feature extraction failed"
    exit 1
fiAP all images come from a single camera to improve accuracy
#TODO: this might not be the case in the future!
colmap feature_extractor \
    --database_path "${COLMAP_TMP_DIR}/database.db" \
    --image_path "${KEYFRAME_IMAGES_DIR}" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV # OPENCV is robust for distortion

if [ $? -ne 0 ]; then
    echo "ERROR: Feature extraction failed"
    exit 1
fi

# # 3. Run feature matching
# colmap exhaustive_matcher \
#     --database_path "${COLMAP_TMP_DIR}/database.db"

# if [ $? -ne 0 ]; then
#     echo "❌ ERROR: Feature matching failed"
#     exit 1
# fi

# # 4. Run a quick reconstruction just to generate the sparse model
# mkdir -p "${COLMAP_TMP_DIR}/sparse"
# colmap mapper \
#     --database_path "${COLMAP_TMP_DIR}/database.db" \
#     --image_path "${KEYFRAME_IMAGES_DIR}" \
#     --output_path "${COLMAP_TMP_DIR}/sparse"

# if [ $? -ne 0 ]; then
#     echo "❌ ERROR: COLMAP mapper failed"
#     exit 1
# fi

# # Verify the sparse model was created
# if [ ! -d "${COLMAP_TMP_DIR}/sparse/0" ]; then
#     echo "❌ ERROR: COLMAP failed to create sparse/0 directory"
#     exit 1
# fi

# echo "--- Step A complete. Intrinsics are in ${COLMAP_TMP_DIR}/sparse/0 ---"

# # --- STEP B: Convert SLAM Data using our Python Script ---
# echo "--- Starting Step B: Merging SLAM data into COLMAP format ---"

# /home/bwilliams/encode/code/ben-splat-env/bin/python "${CONVERSION_SCRIPT}" \
#     --colmap_intrinsics_dir "${COLMAP_TMP_DIR}/sparse/0" \
#     --images_dir "${KEYFRAME_IMAGES_DIR}" \
#     --slam_pose_file "${SLAM_POSE_FILE}" \
#     --slam_ply_file "${SLAM_PLY_FILE}" \
#     --output_dir "${FINAL_OUTPUT_DIR}"

# if [ $? -ne 0 ]; then
#     echo "❌ ERROR: Python conversion script failed"
#     exit 1
# fi

# echo "--- Workflow complete! Final data is ready for splatting. ---"

# # --- Final Check: Display the output structure ---
# echo "Final directory structure:"
# tree "${FINAL_OUTPUT_DIR}"