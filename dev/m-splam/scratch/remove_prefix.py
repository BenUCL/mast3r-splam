#!/usr/bin/env python3
"""Quick script to remove first 5 characters from all filenames in a directory."""

from pathlib import Path

DIR = "/home/bwilliams/encode/data/sweet-coral_indo_tabuhan_p1_20250210/_indonesia_tabuhan_p1_20250210/corrected/combined_small"

dir_path = Path(DIR)
files = sorted(dir_path.glob("*"))

print(f"Found {len(files)} files in {DIR}")
print("\nPreview (first 5):")
for f in files[:5]:
    new_name = f.name[5:]
    print(f"  {f.name} -> {new_name}")

response = input(f"\nRename all {len(files)} files (remove first 5 chars)? [y/N]: ")
if response.lower() != 'y':
    print("Aborted.")
    exit(0)

for f in files:
    if f.is_file():
        new_name = f.parent / f.name[5:]
        f.rename(new_name)
        print(f"Renamed: {f.name} -> {new_name.name}")

print(f"\n✓ Done! Renamed {len(files)} files")
