#!/usr/bin/env python3
"""
Download Sweet-Corals Q7 images from Hugging Face reliably.

Features:
- Auth token (env or .env)
- Raised timeouts to avoid read timeouts
- Folder filtering (Q7_Left / Q7_Right / both)
- Resume support (snapshot_download skips existing files)
- Retries with exponential backoff on transient failures
- Debug logs to a file
- Simple JPEG validation

Usage:
  python3 download_sweet.py --side left
  python3 download_sweet.py --side right
  python3 download_sweet.py --side both
  python3 download_sweet.py --side both --max-workers 4

Token:
  - Create a Read token at https://huggingface.co/settings/tokens
  - export HUGGINGFACE_HUB_TOKEN="hf_XXXXXXXXXXXXXXXX"
"""

from __future__ import annotations

from typing import List, Sequence
import argparse
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import logging as hf_logging

try:
  from dotenv import load_dotenv
except ImportError:
  load_dotenv = None  # Optional dependency

# =========================
# Global configuration
# =========================
DATASET_ID: str = "wildflow/sweet-corals"
OUT_DIR: str = "/home/bwilliams/encode/data/sweet-corals_tabuhan_p1_20250210_raw"
Q7_BASE: str = "_indonesia_tabuhan_p1_20250210/raw"
LEFT_DIRNAME: str = "Q7_Left"
RIGHT_DIRNAME: str = "Q7_Right"

# Defaults (CLI can override some)
DEFAULT_MAX_WORKERS: int = 3           # 2–6 is usually fine
DEFAULT_ETAG_TIMEOUT_S: int = 180      # metadata timeout
DEFAULT_DL_TIMEOUT_S: int = 180        # read timeout via env var
SAMPLES_TO_CHECK: int = 10             # validate N random images per folder


def ensure_env_timeouts() -> None:
  """Set HF Hub timeouts in env if not already set."""
  os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(DEFAULT_ETAG_TIMEOUT_S))
  os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(DEFAULT_DL_TIMEOUT_S))


def enable_hf_debug_logging(log_path: Path) -> None:
  """Enable Hugging Face hub debug logs to a file (and INFO to console)."""
  # HF internal verbosity
  os.environ.setdefault("HF_HUB_VERBOSITY", "debug")
  hf_logging.set_verbosity_debug()

  # Python logging: DEBUG to file, INFO to console
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  # File handler (avoid duplicates if re-run in same interpreter)
  if not any(isinstance(h, logging.FileHandler) and getattr(h, "_hf_log", False) for h in logger.handlers):
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    fh._hf_log = True  # type: ignore[attr-defined]
    logger.addHandler(fh)

  # Console handler
  if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_hf_console", False) for h in logger.handlers):
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    ch._hf_console = True  # type: ignore[attr-defined]
    logger.addHandler(ch)


def load_token() -> str:
  """Load token from env/.env; exit with guidance if missing."""
  if load_dotenv:
    # Try current dir .env, then parent .env, then repo-ish root .env (your earlier pattern).
    for probe in (Path(".env"), Path("..") / ".env", Path(__file__).resolve().parent.parent.parent / ".env"):
      if probe.exists():
        load_dotenv(probe)  # type: ignore[arg-type]
        break

  token = os.getenv("HUGGINGFACE_HUB_TOKEN", "")
  if not token:
    print(
      "ERROR: HUGGINGFACE_HUB_TOKEN not set.\n"
      "Create a Read token at https://huggingface.co/settings/tokens and export it, e.g.\n"
      '  export HUGGINGFACE_HUB_TOKEN="hf_XXXXXXXXXXXXXXXX"\n',
      file=sys.stderr,
    )
    sys.exit(2)
  return token


def is_jpeg(path: Path) -> bool:
  """Return True if file starts with JPEG magic bytes FF D8 FF."""
  try:
    with path.open("rb") as f:
      return f.read(3) == b"\xff\xd8\xff"
  except OSError:
    return False


def pick_samples(paths: Sequence[Path], k: int) -> List[Path]:
  """Pick up to k JPGs for validation."""
  jpgs = [p for p in paths if p.suffix.lower() in (".jpg", ".jpeg")]
  if len(jpgs) <= k:
    return list(jpgs)
  random.shuffle(jpgs)
  return jpgs[:k]


def list_folder_files(folder: Path) -> List[Path]:
  """List all files under a folder (recursively)."""
  if not folder.exists():
    return []
  return [p for p in folder.rglob("*") if p.is_file()]


def download_side(side: str, token: str, max_workers: int, etag_timeout: int) -> Path:
  """Download one side (left/right) using snapshot_download with filters."""
  if side not in ("left", "right"):
    raise ValueError("side must be 'left' or 'right'")

  subdir = LEFT_DIRNAME if side == "left" else RIGHT_DIRNAME
  allow_patterns = [f"{Q7_BASE}/{subdir}/**"]

  print(f"→ Downloading {side.upper()} with allow_patterns={allow_patterns}")
  _ = snapshot_download(
    repo_id=DATASET_ID,
    repo_type="dataset",
    allow_patterns=allow_patterns,
    local_dir=OUT_DIR,
    max_workers=max_workers,
    token=token,
    etag_timeout=etag_timeout,
  )

  # Return the concrete folder we expect images in.
  return Path(OUT_DIR) / "_indonesia_tabuhan_p1_20250210" / "raw" / subdir


def safe_snapshot_download(side: str, token: str, max_workers: int, etag_timeout: int) -> Path:
  """Run snapshot_download with retries/backoff for transient errors."""
  backoff_sec = 60  # constant wait between retries
  attempt = 1
  while True:
    try:
      return download_side(side, token, max_workers, etag_timeout)
    except Exception as exc:  # broad catch to retry timeouts/network hiccups
      print(f"\nDownload attempt {attempt} failed: {exc}")
      traceback.print_exc(limit=2)
      print(f"⏳ Waiting {backoff_sec}s before retry {attempt + 1} ...")
      print("(Download will resume from where it left off - Ctrl+C to stop)")
      time.sleep(backoff_sec)
      attempt += 1


def validate_folder(folder: Path) -> None:
  """Quick sanity check: confirm sample images look like real JPEGs."""
  files = list_folder_files(folder)
  if not files:
    print(f"WARNING: No files found under {folder} (unexpected).")
    return

  samples = pick_samples(files, SAMPLES_TO_CHECK)
  bad = [p for p in samples if not is_jpeg(p)]
  if bad:
    print("Validation FAILED: these sample files are not JPEGs:")
    for p in bad:
      print(f"  - {p}")
    print("Tip: if they start with 'version https://git-lfs.github.com/spec/v1', "
          "they are Git-LFS pointers. Ensure you're using snapshot_download (as in this script), "
          "not raw git clone on LFS paths.")
  else:
    print(f"Validation OK ✅  ({len(samples)} sample files look like valid JPEGs)")


def main() -> int:
  """CLI entrypoint."""
  parser = argparse.ArgumentParser(
    description="Download Sweet-Corals Q7 images from Hugging Face reliably."
  )
  parser.add_argument(
    "--side",
    choices=("left", "right", "both"),
    required=True,
    help="Which folder to download.",
  )
  parser.add_argument(
    "--max-workers",
    type=int,
    default=DEFAULT_MAX_WORKERS,
    help="Concurrent workers (2–6 usually fine).",
  )
  parser.add_argument(
    "--etag-timeout",
    type=int,
    default=DEFAULT_ETAG_TIMEOUT_S,
    help="ETag (metadata) timeout seconds.",
  )
  args = parser.parse_args()

  ensure_env_timeouts()
  token = load_token()

  log_name = f"hf_{args.side}_debug.log"
  enable_hf_debug_logging(Path("/home/bwilliams/encode/data") / log_name)

  try:
    if args.side in ("left", "both"):
      left_dir = safe_snapshot_download("left", token, args.max_workers, args.etag_timeout)
      print(f"Saved under: {left_dir}")
      validate_folder(left_dir)

    if args.side in ("right", "both"):
      right_dir = safe_snapshot_download("right", token, args.max_workers, args.etag_timeout)
      print(f"Saved under: {right_dir}")
      validate_folder(right_dir)

  except Exception as exc:  # pylint: disable=broad-except
    print(
      f"\nERROR: {exc}\n"
      "If this was a ReadTimeout, try reducing --max-workers or increasing --etag-timeout.\n"
      "Re-run the same command to resume safely.",
      file=sys.stderr,
    )
    return 1

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
