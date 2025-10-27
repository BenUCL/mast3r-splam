#!/bin/bash
# filepath: /home/bwilliams/encode/code/lichtfeld-studio/LichtFeld-Studio/dev/downsample_img.sh

SOURCE_DIR="/home/bwilliams/encode/data/sweet-coral_indo_tabuhan_p1_20250210/_indonesia_tabuhan_p1_20250210/corrected/images"
OUTPUT_DIR="/home/bwilliams/encode/data/sweet-coral_indo_tabuhan_p1_20250210/_indonesia_tabuhan_p1_20250210/corrected/downsampled"

# Check if ImageMagick is installed
if ! command -v convert &> /dev/null; then
    echo "Error: ImageMagick is not installed. Please run: sudo apt install imagemagick"
    exit 1
fi

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU parallel is not installed. Please run: sudo apt install parallel"
    exit 1
fi

# Check if source directory exists
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: Source directory $SOURCE_DIR does not exist!"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to process a single image
process_image() {
    local file="$1"
    local output_dir="$2"
    
    # Get filename without path
    local filename=$(basename "$file")
    # Remove extension and add .jpg
    local name="${filename%.*}"
    local output_file="$output_dir/${name}.jpg"
    
    # Skip if already processed
    if [[ -f "$output_file" ]]; then
        echo "SKIP: $filename (already exists)"
        return 0
    fi
    
    # Downsample with max side 1600px and 80% quality using ImageMagick
    # -resize 1600x1600> (> means only shrink, don't enlarge)
    # -quality 80 for 80% JPEG quality
    if convert "$file" -resize 1600x1600\> -quality 80 "$output_file" 2>/dev/null; then
        echo "✓ SUCCESS: $filename"
        return 0
    else
        echo "✗ FAILED: $filename"
        return 1
    fi
}

# Export the function and variables for parallel
export -f process_image
export OUTPUT_DIR

# Get list of image files
mapfile -t image_files < <(find "$SOURCE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tiff" -o -iname "*.tif" -o -iname "*.bmp" \))

total=${#image_files[@]}
echo "Found $total images to process..."

if [[ $total -eq 0 ]]; then
    echo "No images found in $SOURCE_DIR"
    exit 1
fi

# Calculate optimal number of jobs (CPU cores - 1, minimum 1, maximum 8)
num_cores=$(nproc)
max_jobs=$((num_cores > 1 ? num_cores - 1 : 1))
max_jobs=$((max_jobs > 8 ? 8 : max_jobs))

echo "Using $max_jobs parallel jobs (detected $num_cores CPU cores)"
echo "Starting parallel processing..."

# Process images in parallel with progress bar
printf '%s\n' "${image_files[@]}" | parallel -j "$max_jobs" --bar process_image {} '"$OUTPUT_DIR"'

# Count results
success_count=$(find "$OUTPUT_DIR" -name "*.jpg" | wc -l)
failed_count=$((total - success_count))

echo ""
echo "Done! Processed $success_count images successfully, $failed_count failed."
echo "Images saved to $OUTPUT_DIR"