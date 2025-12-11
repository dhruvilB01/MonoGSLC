#!/usr/bin/env python3
"""
Evaluate loop-detection summaries against TUM ground-truth poses.

Example:
python scripts/eval_loop_detections.py \
    --summary results/loop_eval/sift/neighborhood_1_train_rgb/summary.json \
    --dataset_type tum \
    --distance_thresh 1.0 \
    --angle_thresh 15 \
    --index_tolerance 5
"""
import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from utils.dataset import TUMParser
except ImportError as e:  # pragma: no cover - informative error for missing deps
    raise SystemExit(
        "Failed to import TUMParser from utils.dataset. "
        "Make sure MonoGS is on the PYTHONPATH."
    ) from e


IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def list_images(frames_dir: str) -> List[str]:
    files: List[str] = []
    for pattern in IMAGE_GLOBS:
        files.extend(glob.glob(os.path.join(frames_dir, pattern)))
    files.sort()
    return files


def rotation_error_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    ra = pose_a[:3, :3]
    rb = pose_b[:3, :3]
    rel = ra @ rb.T
    trace = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def translation_error(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    ta = pose_a[:3, 3]
    tb = pose_b[:3, 3]
    return float(np.linalg.norm(ta - tb))


def load_tum_poses(
    dataset_root: str, frames: Sequence[str]
) -> Tuple[List[Optional[np.ndarray]], List[str]]:
    parser = TUMParser(dataset_root)
    name_to_idx: Dict[str, int] = {}
    idx_to_pose: List[Optional[np.ndarray]] = [None] * len(frames)

    for idx, path in enumerate(frames):
        name = os.path.basename(path)
        if name in name_to_idx:
            raise ValueError(f"Duplicate frame name detected: {name}")
        name_to_idx[name] = idx

    matched = 0
    for path, pose in zip(parser.color_paths, parser.poses):
        key = os.path.basename(path)
        idx = name_to_idx.get(key)
        if idx is None:
            continue
        idx_to_pose[idx] = np.asarray(pose, dtype=np.float64)
        matched += 1

    missing = [frames[i] for i, pose in enumerate(idx_to_pose) if pose is None]
    return idx_to_pose, missing


def compute_gt_loops(
    poses: Sequence[Optional[np.ndarray]],
    min_gap: int,
    dist_thresh: float,
    angle_thresh: float,
) -> List[Tuple[int, int, float, float]]:
    gt_pairs: List[Tuple[int, int, float, float]] = []
    n = len(poses)
    for i in range(n):
        pose_i = poses[i]
        if pose_i is None:
            continue
        for j in range(0, i - min_gap):
            pose_j = poses[j]
            if pose_j is None:
                continue
            dist = translation_error(pose_i, pose_j)
            if dist > dist_thresh:
                continue
            angle = rotation_error_deg(pose_i, pose_j)
            if angle > angle_thresh:
                continue
            gt_pairs.append((j, i, dist, angle))
    return gt_pairs


def mark_gt_coverage(
    det_i: int,
    det_j: int,
    gt_pairs: Sequence[Tuple[int, int, float, float]],
    covered: List[bool],
    idx_tolerance: int,
) -> None:
    for idx, (gt_j, gt_i, _, _) in enumerate(gt_pairs):
        if covered[idx]:
            continue
        if abs(det_i - gt_i) <= idx_tolerance and abs(det_j - gt_j) <= idx_tolerance:
            covered[idx] = True
            return


def evaluate_summary(args: argparse.Namespace) -> Dict:
    with open(args.summary, "r") as f:
        summary = json.load(f)
    detections = summary.get("detections")
    if not detections:
        raise SystemExit(f"No 'detections' key in {args.summary}. Re-run the detector with the latest code.")

    frames_dir = args.frames_dir or summary.get("frames_dir")
    if not frames_dir:
        raise SystemExit("frames_dir missing from summary; please provide --frames_dir.")
    stride = args.stride or summary.get("stride", 1)
    frames = list_images(frames_dir)
    if stride > 1:
        frames = frames[:: int(stride)]
    if not frames:
        raise SystemExit(f"No frames found under {frames_dir}")

    if args.dataset_type != "tum":
        raise SystemExit(f"Unsupported dataset_type '{args.dataset_type}'. Currently only 'tum' is implemented.")
    dataset_root = args.dataset_path or os.path.dirname(frames_dir)
    gt_poses, missing_frames = load_tum_poses(dataset_root, frames)
    if missing_frames:
        print(f"[warn] Missing GT poses for {len(missing_frames)} frames (will skip those detections)")

    min_gap = args.min_gap if args.min_gap is not None else summary.get("min_gap", 0)
    gt_pairs = compute_gt_loops(
        gt_poses,
        min_gap=min_gap,
        dist_thresh=args.distance_thresh,
        angle_thresh=args.angle_thresh,
    )

    evaluated = 0
    true_pos = 0
    false_pos = 0
    skipped = 0
    per_detection: List[Dict] = []

    for det in detections:
        i = int(det["i"])
        j = int(det["j"])
        if i >= len(gt_poses) or j >= len(gt_poses) or i <= j:
            skipped += 1
            continue
        pose_i = gt_poses[i]
        pose_j = gt_poses[j]
        if pose_i is None or pose_j is None:
            skipped += 1
            continue
        dist = translation_error(pose_i, pose_j)
        angle = rotation_error_deg(pose_i, pose_j)
        is_tp = dist <= args.distance_thresh and angle <= args.angle_thresh
        evaluated += 1
        if is_tp:
            true_pos += 1
        else:
            false_pos += 1
        per_det_entry = {
            "i": i,
            "j": j,
            "distance_m": dist,
            "angle_deg": angle,
            "is_tp": is_tp,
        }
        for extra_key in ("score", "score_inliers", "inlier_ratio", "good_matches"):
            if extra_key in det:
                per_det_entry[extra_key] = det[extra_key]
        per_detection.append(per_det_entry)

    gt_covered = [False] * len(gt_pairs)
    for det in per_detection:
        if not det["is_tp"]:
            continue
        mark_gt_coverage(det["i"], det["j"], gt_pairs, gt_covered, args.index_tolerance)

    precision = true_pos / evaluated if evaluated else 0.0
    recall = (sum(gt_covered) / len(gt_pairs)) if gt_pairs else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    report = {
        "summary_path": os.path.abspath(args.summary),
        "frames_dir": os.path.abspath(frames_dir),
        "dataset_root": os.path.abspath(dataset_root),
        "dataset_type": args.dataset_type,
        "num_frames": len(frames),
        "stride": stride,
        "min_gap": min_gap,
        "distance_thresh": args.distance_thresh,
        "angle_thresh": args.angle_thresh,
        "index_tolerance": args.index_tolerance,
        "detections_total": len(detections),
        "detections_evaluated": evaluated,
        "detections_skipped": skipped,
        "true_positives": true_pos,
        "false_positives": false_pos,
        "precision": precision,
        "gt_loops": len(gt_pairs),
        "gt_loops_matched": int(sum(gt_covered)),
        "recall": recall,
        "f1": f1,
        "per_detection": per_detection,
    }
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote evaluation report to {args.out_json}")
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare loop detections with GT poses")
    ap.add_argument("--summary", required=True, help="Path to summary.json produced by a detector script")
    ap.add_argument("--dataset_type", default="tum", choices=["tum"], help="Dataset flavor (currently only TUM)")
    ap.add_argument("--dataset_path", type=str, default=None, help="Root dataset directory (defaults to parent of frames_dir)")
    ap.add_argument("--frames_dir", type=str, default=None, help="Override frames_dir from summary")
    ap.add_argument("--stride", type=int, default=None, help="Override stride when listing frames (for sampled detectors)")
    ap.add_argument("--distance_thresh", type=float, default=1.0, help="Meters threshold to count a TP")
    ap.add_argument("--angle_thresh", type=float, default=15.0, help="Degrees threshold to count a TP")
    ap.add_argument("--min_gap", type=int, default=None, help="Override min_gap used when enumerating GT loops")
    ap.add_argument("--index_tolerance", type=int, default=5, help="Allowable idx difference when matching GT loops")
    ap.add_argument("--out_json", type=str, default=None, help="Optional path to dump the detailed report")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_summary(args)
    print(json.dumps(
        {
            "precision": report["precision"],
            "recall": report["recall"],
            "f1": report["f1"],
            "tp": report["true_positives"],
            "fp": report["false_positives"],
            "gt_loops": report["gt_loops"],
            "gt_loops_matched": report["gt_loops_matched"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
