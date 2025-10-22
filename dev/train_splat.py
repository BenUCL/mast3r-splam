#!/usr/bin/env python3
"""
train_splat.py

Wrapper to run LichtFeld-Studio and collect a concise run report.

Usage example:
  python train_splat.py --lichtfeld /home/bwilliams/encode/code/lichtfeld-studio/build/LichtFeld-Studio \
    -d /home/bwilliams/encode/data/intermediate_data/reef_soneva/for_splat \
    -o /home/bwilliams/encode/code/lichtfeld-studio/output/m-slam_reef_soneva_ply \
    -- --headless -i 25000 --max-cap 1000000

Notes:
- The script accepts a required `--lichtfeld` path to the binary and requires `-d` (dataset dir) and `-o` (output dir).
- Any arguments after `--` are forwarded directly to the LichtFeld binary.
- It streams stdout/stderr to both the terminal and a log file in the output dir.
- It writes `run_report.txt` into the output dir containing a header with the command, dataset metadata
  (image count, pose count, cameras paths), and the captured training progress appended at the bottom.

Output files created in the output dir:
- run.log           : full stdout+stderr of the LichtFeld run
- run_report.txt    : compact summary + appended training-step lines
- (LichtFeld outputs) : .ply files, checkpoints etc (unchanged)

"""

import argparse
import subprocess
import shlex
import sys
from pathlib import Path
import datetime
import re
import os
import json


PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s*\|\s*Loss:\s*([0-9.eE+-]+)\s*\|\s*Splats:\s*(\d+)")


def gather_metadata(dataset_dir: Path):
    """Try to collect useful metadata from the dataset folder.

    Expected layout (for_splat):
      - images/  or images_bin (depends on dataset)
      - sparse/0/cameras.bin or cameras.txt
      - sparse/0/images.bin or images.txt
      - sparse/0/points3D.bin
    """
    meta = {}
    # Count image files (common names)
    img_dirs = [dataset_dir / 'images', dataset_dir / 'rgb', dataset_dir / 'imgs']
    img_count = None
    for d in img_dirs:
        if d.exists() and d.is_dir():
            img_count = sum(1 for _ in d.glob('**/*') if _.is_file())
            meta['images_path'] = str(d)
            break
    if img_count is None:
        # fallback: count png/jpg in dataset root
        img_count = sum(1 for _ in dataset_dir.glob('**/*.png')) + sum(1 for _ in dataset_dir.glob('**/*.jpg'))
        meta['images_path'] = str(dataset_dir)
    meta['num_images'] = img_count

    # Sparse folder
    sparse0 = dataset_dir / 'sparse' / '0'
    if not sparse0.exists():
        # maybe user provided a for_splat dir which contains: sparse/0
        # try one level down
        for candidate in dataset_dir.iterdir():
            if candidate.is_dir() and (candidate / 'sparse').exists():
                sparse0 = candidate / 'sparse' / '0'
                break

    meta['sparse_folder'] = str(sparse0) if sparse0.exists() else ''

    # Cameras
    cams_bin = sparse0 / 'cameras.bin'
    cams_txt = sparse0 / 'cameras.txt'
    meta['cameras_bin'] = str(cams_bin) if cams_bin.exists() else ''
    meta['cameras_txt'] = str(cams_txt) if cams_txt.exists() else ''

    # Images/poses
    imgs_bin = sparse0 / 'images.bin'
    imgs_txt = sparse0 / 'images.txt'
    if imgs_bin.exists() or imgs_txt.exists():
        # quick count by file size or lines
        if imgs_txt.exists():
            try:
                meta['num_poses'] = sum(1 for _ in imgs_txt.open('r'))
            except Exception:
                meta['num_poses'] = ''
        else:
            # binary, can't easily parse here; leave empty
            meta['num_poses'] = ''
    else:
        meta['num_poses'] = ''

    return meta


def run_lichtfeld(cmd_list, out_dir: Path):
    """Run the LichtFeld binary (cmd_list) and stream output to a log file and capture progress lines.

    Returns a dict with:
      - all_lines: list of all lines captured
      - progress_lines: list of matched progress tuples (num, total, loss, splats)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'run.log'
    all_lines = []
    progress = []

    # Start process with unbuffered output
    with open(log_path, 'wb', buffering=0) as logfile:
        proc = subprocess.Popen(
            cmd_list, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            bufsize=1,  # line buffered
            universal_newlines=False
        )

        assert proc.stdout is not None
        buffer = b''
        while True:
            # Read character by character to handle carriage returns properly
            chunk = proc.stdout.read(1)
            if not chunk:
                # Process remaining buffer
                if buffer:
                    line = buffer.decode('utf-8', errors='replace')
                    sys.stdout.write(line + '\n')
                    sys.stdout.flush()
                    logfile.write(buffer + b'\n')
                    all_lines.append(line)
                    m = PROGRESS_RE.search(line)
                    if m:
                        progress.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))))
                break
            
            # Write raw byte to log immediately
            logfile.write(chunk)
            
            # Check for line endings or carriage return
            if chunk == b'\n':
                line = buffer.decode('utf-8', errors='replace')
                sys.stdout.write(line + '\n')
                sys.stdout.flush()
                all_lines.append(line)
                m = PROGRESS_RE.search(line)
                if m:
                    progress.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))))
                buffer = b''
            elif chunk == b'\r':
                # Carriage return - LichtFeld uses this for progress bars
                line = buffer.decode('utf-8', errors='replace')
                sys.stdout.write('\r' + line)
                sys.stdout.flush()
                # Check for progress pattern
                m = PROGRESS_RE.search(line)
                if m:
                    progress.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))))
                buffer = b''
            else:
                buffer += chunk
        
        proc.wait()
    return {'all_lines': all_lines, 'progress': progress, 'log': str(log_path)}


def write_report(out_dir: Path, cmd_str: str, meta: dict, run_result: dict):
    report_path = out_dir / 'run_report.txt'
    timestamp = datetime.datetime.now().isoformat()

    # Prepare header
    header = []
    header.append(f"Timestamp: {timestamp}")
    header.append("Command:")
    header.append(cmd_str)
    header.append("")
    header.append("Metadata:")
    header.append(json.dumps(meta, indent=2))
    header.append("")
    header.append("Generated files in output dir:")
    files = [p.name for p in out_dir.glob('*') if p.is_file()]
    header.append('\n'.join(sorted(files)))
    header.append("")
    header.append("--- Training progress (last lines appended below) ---")

    # Write header first, then append progress lines at the end (so header is at top)
    with report_path.open('w') as f:
        for line in header:
            f.write(line + '\n')
        f.write('\n')
        f.write('Progress lines (most recent last):\n')
        for tup in run_result['progress']:
            f.write(f"{tup[0]}/{tup[1]} | Loss: {tup[2]:.6f} | Splats: {tup[3]}\n")

    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description='Run LichtFeld and save a concise run report alongside outputs')
    parser.add_argument('--lichtfeld', required=True, help='Path to LichtFeld-Studio binary')
    parser.add_argument('-d', '--dataset', required=True, help='Dataset directory for LichtFeld -d')
    parser.add_argument('-o', '--output', required=True, help='Output directory for LichtFeld -o (this script will place logs/reports here)')
    parser.add_argument('extra', nargs=argparse.REMAINDER, help='Extra arguments forwarded to LichtFeld (prefix with --)')

    args = parser.parse_args()

    lf_bin = Path(args.lichtfeld).expanduser()
    dataset_dir = Path(args.dataset).expanduser()
    output_dir = Path(args.output).expanduser()

    if not lf_bin.exists():
        print(f"LichtFeld binary not found: {lf_bin}")
        sys.exit(2)

    # Build full command list
    cmd = [str(lf_bin), '-d', str(dataset_dir), '-o', str(output_dir)]
    extra = [e for e in args.extra]
    if extra:
        # If first extra is '--', strip it
        if extra[0] == '--':
            extra = extra[1:]
        cmd += extra

    cmd_str = ' '.join(shlex.quote(c) for c in cmd)

    print(f"Running: {cmd_str}")

    # Gather metadata early
    meta = gather_metadata(dataset_dir)

    # Run LichtFeld and capture output
    run_result = run_lichtfeld(cmd, output_dir)

    # Write report
    report_path = write_report(output_dir, cmd_str, meta, run_result)
    print(f"Report written to: {report_path}")
    print(f"Full log: {run_result['log']}")


if __name__ == '__main__':
    main()
