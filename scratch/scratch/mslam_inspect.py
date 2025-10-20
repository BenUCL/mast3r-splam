#!/usr/bin/env python3
"""
Inspect MASt3R-SLAM (TUM-style) outputs and summarise what we can convert to COLMAP.

Writes a human-readable report covering:
- Keyframe image list and sizes.
- Any intrinsics found in text/yaml/json logs.
- Pose blocks (4x4) discovered and heuristics about T_cw vs T_wc.
- Basic trajectory stats under both conventions.

This DOES NOT write COLMAP files; it only helps you decide how to proceed.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image  # pillow

# Regexes we’ll use to hunt intrinsics and 4x4 matrices.
INTRINSICS_HINTS = [
 "fx", "fy", "cx", "cy", "camera_matrix", "intrinsics", "K",
 "width", "height", "image_width", "image_height",
]
MAT4_ROW_RE = re.compile(r"""
^\s*
(?P<a>-?\d+(\.\d+)?([eE][+-]?\d+)?)\s+
(?P<b>-?\d+(\.\d+)?([eE][+-]?\d+)?)\s+
(?P<c>-?\d+(\.\d+)?([eE][+-]?\d+)?)\s+
(?P<d>-?\d+(\.\d+)?([eE][+-]?\d+)?)\s*
$
""", re.VERBOSE)

@dataclass
class PoseBlock:
 """One 4x4 pose matrix."""
 mat: List[List[float]]  # 4x4 row-major


def parse_args() -> argparse.Namespace:
 parser = argparse.ArgumentParser(description="Inspect MASt3R-SLAM logs.")
 parser.add_argument("--logs", type=Path, required=True, help="Path to logs/ directory")
 parser.add_argument("--out", type=Path, required=True, help="Summary output .txt")
 return parser.parse_args()


def find_keyframes(logs_dir: Path) -> List[Path]:
 """Find keyframe images (common MASt3R layout)."""
 candidates = []
 for p in logs_dir.rglob("*"):
  if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and "keyframe" in str(p.parent).lower():
   candidates.append(p)
 return sorted(candidates)


def image_size(path: Path) -> Optional[Tuple[int, int]]:
 """Return (width, height) or None."""
 try:
  with Image.open(path) as im:
   return im.size
 except Exception:
  return None


def scan_text_for_intrinsics(text: str) -> List[str]:
 """Return lines that look like they contain intrinsics."""
 hits: List[str] = []
 lc = text.lower()
 if any(k in lc for k in INTRINSICS_HINTS):
  for line in text.splitlines():
   ll = line.lower()
   if any(k in ll for k in INTRINSICS_HINTS):
    hits.append(line.strip())
 return hits


def read_text_like(path: Path) -> Optional[str]:
 """Read text from .txt/.yaml/.yml/.json if possible."""
 try:
  if path.suffix.lower() in {".txt", ".yaml", ".yml", ".json"}:
   return path.read_text(errors="ignore")
 except Exception:
  return None
 return None


def parse_mat4_blocks(text: str) -> List[PoseBlock]:
 """Parse 4x4 matrices expressed as 4 numeric rows."""
 lines = [ln for ln in text.splitlines() if ln.strip()]
 blocks: List[PoseBlock] = []
 i = 0
 while i + 3 < len(lines):
  rows: List[List[float]] = []
  ok = True
  for k in range(4):
   m = MAT4_ROW_RE.match(lines[i + k])
   if not m:
    ok = False
    break
   rows.append([float(m.group(f)) for f in ["a", "b", "c", "d"]])
  if ok:
   blocks.append(PoseBlock(rows))
   i += 4
  else:
   i += 1
 return blocks


def mat_R_t(block: PoseBlock) -> Tuple[List[List[float]], List[float]]:
 """Split 4x4 into (R, t)."""
 R = [row[:3] for row in block.mat[:3]]
 t = [row[3] for row in block.mat[:3]]
 return R, t


def det3(R: List[List[float]]) -> float:
 """Determinant of 3x3."""
 a,b,c = R[0]
 d,e,f = R[1]
 g,h,i = R[2]
 return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)


def matT(R: List[List[float]]) -> List[List[float]]:
 """Transpose 3x3."""
 return [[R[r][c] for r in range(3)] for c in range(3)]


def matvec(R: List[List[float]], v: List[float]) -> List[float]:
 """R @ v."""
 return [sum(R[r][c]*v[c] for c in range(3)) for r in range(3)]


def vec_add(a: List[float], b: List[float]) -> List[float]:
 return [a[i]+b[i] for i in range(3)]


def vec_sub(a: List[float], b: List[float]) -> List[float]:
 return [a[i]-b[i] for i in range(3)]


def norm(v: List[float]) -> float:
 return math.sqrt(sum(x*x for x in v))


def camera_center_from_Tcw(R: List[List[float]], t: List[float]) -> List[float]:
 """C = -R^T t."""
 Rt = matT(R)
 Rt_t = matvec(Rt, t)
 return [-x for x in Rt_t]


def stats_for_interpretation(blocks: List[PoseBlock], assume: str) -> Tuple[float, float, float]:
 """
 Compute simple diagnostics:
  - mean |det(R)-1|
  - trajectory length (sum of segment lengths)
  - bbox diagonal of camera centres
 """
 centres: List[List[float]] = []
 det_errs: List[float] = []
 prev: Optional[List[float]] = None

 for blk in blocks:
  R, t = mat_R_t(blk)
  det_errs.append(abs(det3(R) - 1.0))

  if assume == "T_cw":
   C = camera_center_from_Tcw(R, t)
  elif assume == "T_wc":
   # If given T_wc, camera centre in world is translation.
   C = t
  else:
   raise ValueError("assume must be T_cw or T_wc")
  centres.append(C)

 traj = 0.0
 for i in range(1, len(centres)):
  traj += norm(vec_sub(centres[i], centres[i-1]))

 if centres:
  mins = [min(c[j] for c in centres) for j in range(3)]
  maxs = [max(c[j] for c in centres) for j in range(3)]
  diag = norm(vec_sub(maxs, mins))
 else:
  diag = 0.0

 mean_det = sum(det_errs)/len(det_errs) if det_errs else 0.0
 return mean_det, traj, diag


def main() -> None:
 args = parse_args()
 logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

 logs_dir: Path = args.logs
 out_path: Path = args.out
 out_path.parent.mkdir(parents=True, exist_ok=True)

 keyframes = find_keyframes(logs_dir)
 sizes = [(p, image_size(p)) for p in keyframes]

 text_hits: List[Tuple[Path, List[str]]] = []
 pose_blocks: List[PoseBlock] = []

 for p in logs_dir.rglob("*"):
  txt = read_text_like(p)
  if not txt:
   continue
  intr = scan_text_for_intrinsics(txt)
  if intr:
   text_hits.append((p, intr))
  # Heuristic: also try to parse 4x4 blocks
  mats = parse_mat4_blocks(txt)
  if mats:
   pose_blocks.extend(mats)

 # Compute pose stats
 pose_stats = ""
 if pose_blocks:
  m1, t1, d1 = stats_for_interpretation(pose_blocks, "T_cw")
  m2, t2, d2 = stats_for_interpretation(pose_blocks, "T_wc")
  likely = "T_cw" if (t1 > t2 and m1 <= m2*1.5) else "T_wc"
  pose_stats = (
   f"Pose blocks found: {len(pose_blocks)}\n"
   f"Assuming T_cw: mean|det(R)-1|={m1:.3e}, traj={t1:.3f}, bbox_diag={d1:.3f}\n"
   f"Assuming T_wc: mean|det(R)-1|={m2:.3e}, traj={t2:.3f}, bbox_diag={d2:.3f}\n"
   f"Likely convention (heuristic): {likely}\n"
  )
 else:
  pose_stats = "No 4x4 pose blocks detected.\n"

 # Write summary
 lines: List[str] = []
 lines.append("# MASt3R-SLAM inspection summary\n")
 lines.append(f"Logs dir: {logs_dir}\n")
 lines.append("## Keyframes\n")
 lines.append(f"Count: {len(keyframes)}\n")
 for p, sz in sizes[:10]:
  lines.append(f"- {p.name}: {sz[0]}x{sz[1]}" if sz else f"- {p.name}: (size unknown)")
 if len(sizes) > 10:
  lines.append(f"... (+{len(sizes)-10} more)\n")

 lines.append("\n## Intrinsics hints in text-like files\n")
 if text_hits:
  for p, intr in text_hits:
   lines.append(f"- {p} :")
   for ln in intr[:10]:
    lines.append(f"    {ln}")
   if len(intr) > 10:
    lines.append("    ...")
 else:
  lines.append("None found (fx/fy/cx/cy/K not detected).\n")

 lines.append("\n## Pose diagnostics\n")
 lines.append(pose_stats)

 lines.append("\n## Next steps (recommended)\n")
 lines.append("1) Decide intrinsics strategy:")
 lines.append("   - Preferred: calibrate in-housing and UNDISTORT → use PINHOLE in cameras.txt")
 lines.append("   - If no calibration: approximate fx (DUSt3R focal), set cx,cy to image centre; expect artefacts")
 lines.append("2) Confirm pose convention above before writing images.txt")
 lines.append("3) Optionally seed points3D from your .ply; otherwise leave empty\n")

 out_path.write_text("\n".join(lines))
 logging.info("Wrote summary to %s", out_path)


if __name__ == "__main__":
 main()
