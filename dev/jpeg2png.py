"""Convert all JPEG images in a directory to PNG format and save them in another directory."""
from PIL import Image
import glob, os

# Configuration
INPUT_DIR = "/home/bwilliams/encode/code/lichtfeld-studio/data/tandt/truck/images"
OUTPUT_DIR = "/home/bwilliams/encode/data/truck_slam_splat/raw_images_png"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all JPEG files (JPG, jpg, JPEG, jpeg)
jpeg_files = glob.glob(f"{INPUT_DIR}/*.[jJ][pP][gG]") + \
             glob.glob(f"{INPUT_DIR}/*.[jJ][pP][eE][gG]")
total = len(jpeg_files)

print(f"Found {total} JPEG images to convert...")

# Convert all JPEG files to PNG
count = 0
for jpg_file in jpeg_files:
    img = Image.open(jpg_file)
    base = os.path.basename(jpg_file)
    png_file = os.path.splitext(base)[0] + ".png"
    img.save(f"{OUTPUT_DIR}/{png_file}")
    # print progress
    count += 1
    print(f"Converted {count}/{total} images", end="\r")