#!/usr/bin/env python3
"""
run_pipeline.py

Orchestrate the complete MASt3R-SLAM → Gaussian Splatting pipeline.

Runs the following steps:
1. COLMAP intrinsics estimation (estimate_intrinsics.py)
2. Intrinsics conversion (shuttle_intrinsics.py)
3. MASt3R-SLAM
4. Move MASt3R-SLAM outputs to run directory
5. Pose/keyframe conversion (cam_pose_keyframes_shuttle.py)
6. PLY to points3D conversion (mslam_ply_to_points3d.py)
7. Gaussian splatting training (train_splat.py)

Usage:
    python run_pipeline.py --config slam_splat_config.yaml
    python run_pipeline.py --config slam_splat_config.yaml --start-from 3  # Resume from step 3
    python run_pipeline.py --config slam_splat_config.yaml --only 1        # Run only step 1
"""

import argparse
import subprocess
import sys
import shutil
import yaml
from pathlib import Path
from datetime import datetime
import json
import os
import time

# TODO: Consider contributing a PR to MASt3R-SLAM to support custom output directories
# via command-line argument (--output_dir). This would eliminate the need to move files
# post-run and make the pipeline cleaner. For now, we move files after SLAM completion.


class PipelineRunner:
    def __init__(self, config_path):
        """Initialize pipeline with config file."""
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.run_name = self.config['run_name']
        self.paths = self.config['paths']
        
        # Auto-detect dataset name from images path (MASt3R-SLAM uses directory name)
        images_dir = Path(self.paths['images_path'])
        self.dataset_name = images_dir.name
        
        # Setup run directory
        self.run_dir = Path(self.paths['intermediate_data_root']) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config copy to run directory
        config_copy = self.run_dir / 'pipeline_config.yaml'
        shutil.copy(self.config_path, config_copy)
        print(f"📋 Config saved to: {config_copy}")
        
        # Setup logging and timing
        self.log_file = self.run_dir / 'pipeline.log'
        self.terminal_log_file = self.run_dir / 'terminal_output.log'
        self.start_time = datetime.now()
        self.pipeline_start_time = time.time()
        self.step_timings = {}  # Store timing for each step
        
        # Append to terminal output log (not overwrite)
        with open(self.terminal_log_file, 'a') as f:
            f.write(f"\n\n{'#'*70}\n")
            f.write(f"{'#'*70}\n")
            f.write(f"# NEW PIPELINE RUN\n")
            f.write(f"{'#'*70}\n")
            f.write(f"{'#'*70}\n")
            f.write(f"Pipeline started: {self.start_time}\n")
            f.write(f"Run name: {self.run_name}\n")
            f.write(f"Dataset name: {self.dataset_name}\n")
            f.write(f"Config: {self.config_path}\n")
            f.write(f"{'#'*70}\n\n")
        
        self.log(f"Pipeline started: {self.start_time}")
        self.log(f"Run name: {self.run_name}")
        self.log(f"Dataset name (from images path): {self.dataset_name}")
        self.log(f"Config: {self.config_path}")
    
    def log(self, message):
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def format_duration(self, seconds):
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {mins}m {secs}s"
    
    def log_timing(self, step_num, step_name, duration, skipped=False):
        """Log timing information for a step."""
        elapsed_total = time.time() - self.pipeline_start_time
        
        if skipped:
            # Step was skipped - show previous timing if available
            if step_num in self.step_timings:
                prev_duration = self.step_timings[step_num]
                self.log(f"⏭️  Step {step_num} skipped (previously took {self.format_duration(prev_duration)})")
            else:
                self.log(f"⏭️  Step {step_num} skipped")
        else:
            # Step completed - record timing
            self.step_timings[step_num] = duration
            self.log(f"⏱️  Step {step_num} completed in {self.format_duration(duration)}")
        
        self.log(f"⏱️  Total elapsed time: {self.format_duration(elapsed_total)}")
    
    def run_command(self, cmd, description, check=True):
        """Run a command with logging and terminal output capture."""
        self.log(f"\n{'='*70}")
        self.log(f"Step: {description}")
        self.log(f"Command: {' '.join(cmd)}")
        self.log(f"{'='*70}")
        
        # Also log to terminal output file with clear separator
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.terminal_log_file, 'a') as f:
            f.write(f"\n{'#'*70}\n")
            f.write(f"# Step: {description}\n")
            f.write(f"# Time: {timestamp}\n")
            f.write(f"# Command: {' '.join(cmd)}\n")
            f.write(f"{'#'*70}\n\n")
        
        try:
            # Use Popen to capture output while displaying it in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Stream output to both terminal and log file
            with open(self.terminal_log_file, 'a') as log_f:
                for line in process.stdout:
                    # Print to terminal
                    print(line, end='')
                    # Write to terminal log
                    log_f.write(line)
                    log_f.flush()  # Ensure immediate write
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code != 0:
                if check:
                    self.log(f"✗ {description} failed with exit code {return_code}")
                    raise subprocess.CalledProcessError(return_code, cmd)
                else:
                    self.log(f"✗ {description} failed with exit code {return_code}")
                    return False
            
            self.log(f"✓ {description} completed")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"✗ {description} failed with exit code {e.returncode}")
            if check:
                raise
            return False
    
    def step_1_intrinsics_estimation(self):
        """Step 1: Estimate camera intrinsics using COLMAP.
        
        Outputs:
            - {run_dir}/colmap_outputs/cameras.txt - Camera intrinsics in COLMAP format
            - {run_dir}/colmap_outputs/sparse/0/ - COLMAP reconstruction (cameras.bin, images.bin, points3D.bin)
            - {run_dir}/colmap_outputs/calibration_summary.txt - Summary of calibration results
        """
        #TODO: currently using 100 images which takes about 3min and could be overkill. Could even work with 3 images.
        # Would need to test how much the intrinsics change. OR consider taking small sets of image, distributed across
        # the dataset and getting a mean of intrinsics. As currently this takes the first 100 images.
        # TODO: should be able to take anyb colmap camera model, not just OPENCV and OPENCV_FISHEYE
        step_name = "1. COLMAP Intrinsics Estimation"
        step_num = 1
        self.log(f"\n{'#'*70}")
        self.log(f"# {step_name}")
        self.log(f"{'#'*70}")
        
        step_start = time.time()
        
        # Check if already done
        output_cameras = self.run_dir / 'colmap_outputs' / 'cameras.txt'
        if self.config['pipeline'].get('skip_existing', True) and output_cameras.exists():
            self.log(f"⏭️  Skipping - output exists: {output_cameras}")
            self.log_timing(step_num, step_name, 0, skipped=True)
            return True
        
        cfg = self.config['intrinsics_estimation']
        cmd = [
            'python',
            str(Path(__file__).parent / 'estimate_intrinsics.py'),
            '--images_path', self.paths['images_path'],
            '--dataset', self.run_name,
            '--num_images', str(cfg['num_images']),
            '--camera_model', cfg['camera_model']
        ]
        
        if cfg.get('overwrite', False):
            cmd.append('--overwrite')
        
        result = self.run_command(cmd, step_name)
        step_duration = time.time() - step_start
        self.log_timing(step_num, step_name, step_duration, skipped=False)
        return result
    
    def step_2_intrinsics_conversion(self):
        """Step 2: Convert intrinsics for MASt3R-SLAM and LichtFeld.
        
        Outputs:
            - {run_dir}/intrinsics.yaml - Intrinsics for MASt3R-SLAM (OPENCV format with distortion)
            - {run_dir}/for_splat/sparse/0/cameras.bin - PINHOLE camera model for LichtFeld
            - {run_dir}/for_splat/sparse/0/cameras.txt - PINHOLE camera model (text format)
        """
        step_name = "2. Intrinsics Conversion"
        step_num = 2
        self.log(f"\n{'#'*70}")
        self.log(f"# {step_name}")
        self.log(f"{'#'*70}")
        
        step_start = time.time()
        
        # Check if already done
        output_yaml = self.run_dir / 'intrinsics.yaml'
        if self.config['pipeline'].get('skip_existing', True) and output_yaml.exists():
            self.log(f"⏭️  Skipping - output exists: {output_yaml}")
            self.log_timing(step_num, step_name, 0, skipped=True)
            return True
        
        cfg = self.config['intrinsics_conversion']
        cmd = [
            'python',
            str(Path(__file__).parent / 'shuttle_intrinsics.py'),
            '--dataset', self.run_name
        ]
        
        if cfg.get('keep_original', False):
            cmd.append('--keep-original')
        
        result = self.run_command(cmd, step_name)
        step_duration = time.time() - step_start
        self.log_timing(step_num, step_name, step_duration, skipped=False)
        return result
    
    def step_3_mast3r_slam(self):
        """Step 3: Run MASt3R-SLAM.
        
        Outputs (initially in MASt3R-SLAM/logs/, moved in step 4):
            - {dataset_name}.ply - Dense 3D point cloud from SLAM
            - {dataset_name}.txt - Camera poses in TUM format (timestamp tx ty tz qx qy qz qw)
            - keyframes/{dataset_name}/ - Undistorted keyframe images
        
        Note: MASt3R-SLAM names outputs based on the images directory name, not run_name.
        """
        step_name = "3. MASt3R-SLAM"
        step_num = 3
        self.log(f"\n{'#'*70}")
        self.log(f"# {step_name}")
        self.log(f"{'#'*70}")
        
        step_start = time.time()

        # Check if already done - look in mslam_logs for the renamed files (run_name)
        # or the original dataset_name files, or in old MASt3R-SLAM location
        mslam_logs = self.run_dir / 'mslam_logs'
        
        # Check for renamed files (run_name) in mslam_logs
        renamed_ply = mslam_logs / f'{self.run_name}.ply'
        renamed_txt = mslam_logs / f'{self.run_name}.txt'
        
        # Check for original files (dataset_name) in mslam_logs
        dataset_ply = mslam_logs / f'{self.dataset_name}.ply'
        dataset_txt = mslam_logs / f'{self.dataset_name}.txt'
        
        # Check old location with dataset_name
        old_ply = Path(self.paths['mast3r_slam_root']) / 'logs' / f'{self.dataset_name}.ply'

        if self.config['pipeline'].get('skip_existing', True):
            if renamed_ply.exists() or dataset_ply.exists() or old_ply.exists():
                existing = renamed_ply if renamed_ply.exists() else (dataset_ply if dataset_ply.exists() else old_ply)
                self.log(f"⏭️  Skipping - output exists: {existing}")
                self.log_timing(step_num, step_name, 0, skipped=True)
                return True

        cfg = self.config['mast3r_slam']
        mslam_root = Path(self.paths['mast3r_slam_root'])

        # Save original working directory
        original_cwd = os.getcwd()

        # Change working directory to MASt3R-SLAM root so model checkpoints are found
        # Otherwise it gives error that model ckpts can't be found
        os.chdir(mslam_root)

        # Build config path
        config_path = cfg['config']
        if not Path(config_path).is_absolute():
            config_path = mslam_root / config_path

        cmd = [
            'python',
            str(mslam_root / 'main.py'),
            '--dataset', self.paths['images_path'],
            '--config', str(config_path),
            '--calib', str(self.run_dir / 'intrinsics.yaml')
        ]
        
        # Disable visualization if requested (allows automated pipeline execution)
        if not cfg.get('enable_visualization', False):
            cmd.append('--no-viz')

        # Add extra args
        cmd.extend(cfg.get('extra_args', []))

        result = self.run_command(cmd, step_name)
        
        # Restore original working directory
        os.chdir(original_cwd)
        
        step_duration = time.time() - step_start
        self.log_timing(step_num, step_name, step_duration, skipped=False)
        return result
    
    def step_4_move_mslam_outputs(self):
        """Step 4: Move and rename MASt3R-SLAM outputs from logs/ to run directory.
        
        Moves outputs from MASt3R-SLAM/logs/ and renames them to use run_name:
            - {run_dir}/mslam_logs/{run_name}.ply - Point cloud (renamed from {dataset_name}.ply)
            - {run_dir}/mslam_logs/{run_name}.txt - Poses (renamed from {dataset_name}.txt)
            - {run_dir}/mslam_logs/keyframes/ - Keyframe images
        
        Note: MASt3R-SLAM names outputs based on images directory name, so we rename for consistency.
        """
        step_name = "4. Move MASt3R-SLAM Outputs"
        step_num = 4
        self.log(f"\n{'#'*70}")
        self.log(f"# {step_name}")
        self.log(f"{'#'*70}")
        
        step_start = time.time()
        
        mslam_root = Path(self.paths['mast3r_slam_root'])
        src_logs = mslam_root / 'logs'
        target_mslam = self.run_dir / 'mslam_logs'
        target_mslam.mkdir(parents=True, exist_ok=True)
        
        moved_any = False
        
        # Auto-detect dataset name from images path (MASt3R-SLAM uses directory name)
        images_dir = Path(self.paths['images_path'])
        dataset_name = images_dir.name
        self.log(f"Detected MASt3R-SLAM dataset name: {dataset_name}")
        
        # Move keyframes directory (if in source location)
        src_keyframes = src_logs / 'keyframes' / dataset_name
        target_keyframes = target_mslam / 'keyframes'
        
        if src_keyframes.exists():
            if target_keyframes.exists():
                shutil.rmtree(target_keyframes)
            shutil.move(str(src_keyframes), str(target_keyframes))
            self.log(f"✓ Moved keyframes: {src_keyframes} → {target_keyframes}")
            moved_any = True
        elif target_keyframes.exists():
            self.log(f"✓ Keyframes already in target: {target_keyframes}")
            moved_any = True
        else:
            self.log(f"⚠️  Keyframes not found: {src_keyframes}")
        
        # Move TUM poses and PLY (using detected dataset name)
        for ext in ['.txt', '.ply']:
            src_file = src_logs / f'{dataset_name}{ext}'
            target_file = target_mslam / f'{dataset_name}{ext}'
            
            if src_file.exists():
                if target_file.exists():
                    target_file.unlink()
                shutil.move(str(src_file), str(target_file))
                self.log(f"✓ Moved {ext[1:]}: {src_file} → {target_file}")
                moved_any = True
            elif target_file.exists():
                self.log(f"✓ File already in target: {target_file.name}")
                moved_any = True
            else:
                self.log(f"⚠️  File not found: {src_file}")
        
        # Rename files from dataset_name to run_name for consistency
        if dataset_name != self.run_name:
            self.log(f"\nRenaming files from '{dataset_name}' to '{self.run_name}'...")
            for ext in ['.txt', '.ply']:
                old_name = target_mslam / f'{dataset_name}{ext}'
                new_name = target_mslam / f'{self.run_name}{ext}'
                
                if old_name.exists():
                    if new_name.exists():
                        new_name.unlink()
                    old_name.rename(new_name)
                    self.log(f"✓ Renamed: {old_name.name} → {new_name.name}")
        
        step_duration = time.time() - step_start
        
        if moved_any:
            self.log(f"\n✓ MASt3R-SLAM outputs moved to: {target_mslam}")
            self.log_timing(step_num, step_name, step_duration, skipped=False)
            return True
        else:
            self.log("⚠️  No MASt3R-SLAM outputs found to move")
            self.log_timing(step_num, step_name, step_duration, skipped=False)
            return False
    
    def step_5_pose_conversion(self):
        """Step 5: Convert SLAM poses to COLMAP format.
        
        Outputs:
            - {run_dir}/for_splat/images/ - Keyframe images (copied or symlinked)
            - {run_dir}/for_splat/sparse/0/images.bin - Camera poses in COLMAP binary format
            - {run_dir}/for_splat/sparse/0/images.txt - Camera poses in COLMAP text format
        """
        step_name = "5. Pose/Keyframe Conversion"
        step_num = 5
        self.log(f"\n{'#'*70}")
        self.log(f"# {step_name}")
        self.log(f"{'#'*70}")
        
        step_start = time.time()
        
        # Check if already done
        output_images_bin = self.run_dir / 'for_splat' / 'sparse' / '0' / 'images.bin'
        if self.config['pipeline'].get('skip_existing', True) and output_images_bin.exists():
            self.log(f"⏭️  Skipping - output exists: {output_images_bin}")
            self.log_timing(step_num, step_name, 0, skipped=True)
            return True
        
        cfg = self.config['pose_conversion']
        mslam_logs = self.run_dir / 'mslam_logs'
        
        # Use run_name since step 4 renames the files from dataset_name to run_name
        cmd = [
            'python',
            str(Path(__file__).parent / 'cam_pose_keyframes_shuttle.py'),
            '--dataset', self.run_name,
            '--mslam_logs_dir', str(mslam_logs)
        ]
        
        if cfg.get('link_images', False):
            cmd.append('--link')
        
        if cfg.get('camera_id') is not None:
            cmd.extend(['--camera_id', str(cfg['camera_id'])])
        
        result = self.run_command(cmd, step_name)
        step_duration = time.time() - step_start
        self.log_timing(step_num, step_name, step_duration, skipped=False)
        return result
    
    def step_6_ply_conversion(self):
        """Step 6: Convert PLY to COLMAP points3D.bin.
        
        Outputs:
            - {run_dir}/for_splat/sparse/0/points3D.bin - 3D points in COLMAP format
        """
        step_name = "6. PLY to points3D Conversion"
        step_num = 6
        self.log(f"\n{'#'*70}")
        self.log(f"# {step_name}")
        self.log(f"{'#'*70}")
        
        step_start = time.time()
        
        # Check if already done
        output_points3d = self.run_dir / 'for_splat' / 'sparse' / '0' / 'points3D.bin'
        if self.config['pipeline'].get('skip_existing', True) and output_points3d.exists():
            self.log(f"⏭️  Skipping - output exists: {output_points3d}")
            self.log_timing(step_num, step_name, 0, skipped=True)
            return True
        
        cfg = self.config['ply_conversion']
        mslam_logs = self.run_dir / 'mslam_logs'
        
        # Use run_name since step 4 renames the files from dataset_name to run_name
        cmd = [
            'python',
            str(Path(__file__).parent / 'mslam_ply_to_points3d.py'),
            '--dataset', self.run_name,
            '--mslam_logs_dir', str(mslam_logs),
            '--sample', str(cfg.get('sample_percentage', 10.0))
        ]
        
        result = self.run_command(cmd, step_name)
        step_duration = time.time() - step_start
        self.log_timing(step_num, step_name, step_duration, skipped=False)
        return result
    
    def step_7_gaussian_splatting(self):
        """Step 7: Train Gaussian Splatting model.
        
        Outputs:
            - {run_dir}/splats/splat_*.ply - Gaussian splat PLY files at various iterations
            - {run_dir}/splats/run.log - Full training log
            - {run_dir}/splats/run_report.txt - Concise training summary with progress
        """
        step_name = "7. Gaussian Splatting Training"
        step_num = 7
        self.log(f"\n{'#'*70}")
        self.log(f"# {step_name}")
        self.log(f"{'#'*70}")
        
        step_start = time.time()
        
        # Check if already done
        output_dir = self.run_dir / 'splats'
        final_ply = output_dir / f'splat_{self.config["gaussian_splatting"]["iterations"]}.ply'
        if self.config['pipeline'].get('skip_existing', True) and final_ply.exists():
            self.log(f"⏭️  Skipping - output exists: {final_ply}")
            self.log_timing(step_num, step_name, 0, skipped=True)
            return True
        
        cfg = self.config['gaussian_splatting']
        dataset_dir = self.run_dir / 'for_splat'
        
        cmd = [
            'python',
            str(Path(__file__).parent / 'train_splat.py'),
            '--lichtfeld', self.paths['lichtfeld_binary'],
            '-d', str(dataset_dir),
            '-o', str(output_dir),
            '--'
        ]
        
        # Add LichtFeld arguments
        if cfg.get('headless', True):
            cmd.append('--headless')
        
        cmd.extend(['-i', str(cfg['iterations'])])
        cmd.extend(['--max-cap', str(cfg['max_cap'])])
        
        # Add extra args
        cmd.extend(cfg.get('extra_args', []))
        
        result = self.run_command(cmd, step_name)
        step_duration = time.time() - step_start
        self.log_timing(step_num, step_name, step_duration, skipped=False)
        return result
    
    def run(self, start_from=1, only=None):
        """Run the pipeline."""
        steps = [
            (1, "Intrinsics Estimation", self.step_1_intrinsics_estimation),
            (2, "Intrinsics Conversion", self.step_2_intrinsics_conversion),
            (3, "MASt3R-SLAM", self.step_3_mast3r_slam),
            (4, "Move MASt3R-SLAM Outputs", self.step_4_move_mslam_outputs),
            (5, "Pose Conversion", self.step_5_pose_conversion),
            (6, "PLY Conversion", self.step_6_ply_conversion),
            (7, "Gaussian Splatting", self.step_7_gaussian_splatting)
        ]
        
        if only is not None:
            steps = [(num, name, func) for num, name, func in steps if num == only]
        else:
            steps = [(num, name, func) for num, name, func in steps if num >= start_from]
        
        self.log(f"\n{'='*70}")
        self.log(f"PIPELINE EXECUTION PLAN")
        self.log(f"{'='*70}")
        self.log(f"Run name: {self.run_name}")
        self.log(f"Run directory: {self.run_dir}")
        self.log(f"Steps to execute: {[f'{num}. {name}' for num, name, _ in steps]}")
        self.log(f"{'='*70}\n")
        
        # Run steps
        for step_num, step_name, step_func in steps:
            interactive = self.config['pipeline'].get('interactive', False)
            if interactive:
                response = input(f"\n▶️  Run Step {step_num}: {step_name}? [Y/n]: ")
                if response.lower() == 'n':
                    self.log(f"⏭️  Skipped by user: Step {step_num}")
                    continue
            
            try:
                success = step_func()
                if not success:
                    self.log(f"\n❌ Pipeline failed at step {step_num}: {step_name}")
                    return False
            except Exception as e:
                self.log(f"\n❌ Exception in step {step_num}: {step_name}")
                self.log(f"Error: {str(e)}")
                import traceback
                self.log(traceback.format_exc())
                return False
        
        # Summary
        elapsed = datetime.now() - self.start_time
        self.log(f"\n{'='*70}")
        self.log(f"✅ PIPELINE COMPLETED SUCCESSFULLY")
        self.log(f"{'='*70}")
        self.log(f"Run name: {self.run_name}")
        self.log(f"Total time: {elapsed}")
        self.log(f"Output directory: {self.run_dir}")
        self.log(f"Log file: {self.log_file}")
        self.log(f"Terminal output: {self.terminal_log_file}")
        self.log(f"{'='*70}\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete MASt3R-SLAM → Gaussian Splatting pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to pipeline configuration YAML file'
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=1,
        choices=range(1, 8),
        help='Start from specific step (1-7, default: 1)'
    )
    parser.add_argument(
        '--only',
        type=int,
        default=None,
        choices=range(1, 8),
        help='Run only a specific step (1-7)'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = PipelineRunner(args.config)
    success = pipeline.run(start_from=args.start_from, only=args.only)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
