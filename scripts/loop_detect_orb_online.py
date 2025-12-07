#!/usr/bin/env python3
import argparse
import os
import glob
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2  # type: ignore
except Exception as e:
    raise SystemExit("OpenCV (cv2) is required. Install with: pip install opencv-python")


# ----------------------------- IO utils -----------------------------

def list_images(frames_dir: str, exts=("*.png", "*.jpg", "*.jpeg", "*.bmp")) -> List[str]:
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(frames_dir, e)))
    files = sorted(files)
    return files


def imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------- Feature + matching -----------------------------

@dataclass
class FrameFeatures:
    idx: int
    path: str
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray  # (N, 32)


def create_orb(nfeatures: int = 2000) -> cv2.ORB:
    return cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=15,
    )


def extract_orb(orb: cv2.ORB, img_gray: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    kps, desc = orb.detectAndCompute(img_gray, None)
    if desc is None:
        desc = np.zeros((0, 32), dtype=np.uint8)
    return kps, desc


def create_matcher() -> cv2.DescriptorMatcher:
    # FLANN with LSH for ORB (binary) descriptors
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                        table_number=12,
                        key_size=20,
                        multi_probe_level=2)
    search_params = dict(checks=64)
    return cv2.FlannBasedMatcher(index_params, search_params)


@dataclass
class MatchResult:
    i: int
    j: int
    score_inliers: int
    inlier_ratio: float
    good_matches: int


# ----------------------------- Collage helpers -----------------------------

def letterbox(img_pil: Image.Image, size: Tuple[int, int], fill=(240, 240, 240)) -> Image.Image:
    w, h = img_pil.size
    W, H = size
    scale = min(W / max(1, w), H / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img_pil.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (W, H), fill)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    canvas.paste(img_resized, (ox, oy))
    return canvas


def draw_label(img: Image.Image, text: str, height: int = 22) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    bar_h = min(height, max(16, H // 14))
    draw.rectangle([0, 0, W, bar_h], fill=(0, 0, 0, 150))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(10, bar_h - 6))
    except Exception:
        font = ImageFont.load_default()
    # Truncate long labels
    max_chars = max(10, int(W / 8))
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    draw.text((4, 2), text, fill=(255, 255, 255, 255), font=font)


def save_pair_collage(path_a: str, path_b: str, out_path: str, tile: int = 320, title: Optional[str] = None) -> None:
    a = Image.open(path_a).convert("RGB")
    b = Image.open(path_b).convert("RGB")
    a = letterbox(a, (tile, tile))
    b = letterbox(b, (tile, tile))
    draw_label(a, os.path.basename(path_a))
    draw_label(b, os.path.basename(path_b))

    margin_top = 28 if title else 0
    canvas = Image.new("RGB", (tile * 2, tile + margin_top), (230, 230, 230))
    if title:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=18)
        except Exception:
            font = ImageFont.load_default()
        draw.text((6, 4), title, fill=(20, 20, 20), font=font)
    canvas.paste(a, (0, margin_top))
    canvas.paste(b, (tile, margin_top))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)


def save_group_collage(paths: List[str], out_path: str, cols: int = 6, tile: int = 192, title: Optional[str] = None) -> None:
    if not paths:
        return
    rows = int(np.ceil(len(paths) / cols))
    margin_top = 28 if title else 0
    canvas = Image.new("RGB", (cols * tile, rows * tile + margin_top), (235, 235, 235))
    if title:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=18)
        except Exception:
            font = ImageFont.load_default()
        draw.text((6, 4), title, fill=(20, 20, 20), font=font)
    for k, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        tile_img = letterbox(img, (tile, tile))
        draw_label(tile_img, os.path.basename(p))
        r, c = divmod(k, cols)
        canvas.paste(tile_img, (c * tile, margin_top + r * tile))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)


# ----------------------------- Online loop detection -----------------------------

def knn_ratio_match(matcher: cv2.DescriptorMatcher, desc_q: np.ndarray, desc_t: np.ndarray, ratio: float) -> List[cv2.DMatch]:
    if len(desc_q) == 0 or len(desc_t) == 0:
        return []
    try:
        knn = matcher.knnMatch(desc_q, desc_t, k=2)
    except cv2.error:
        # Fallback to BF when FLANN errors on small sets
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(desc_q, desc_t, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    good.sort(key=lambda x: x.distance)
    return good


def geometric_verification(kp_q, kp_t, matches: List[cv2.DMatch], ransac_thresh: float = 3.0) -> Tuple[int, float]:
    if len(matches) < 8:
        return 0, 0.0
    src = np.float32([kp_q[m.queryIdx].pt for m in matches])
    dst = np.float32([kp_t[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if mask is None:
        return 0, 0.0
    inliers = int(mask.sum())
    ratio = inliers / max(1, len(matches))
    return inliers, ratio


def detect_loops_online(
    frames: List[str],
    out_dir: str,
    min_gap: int = 50,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 3.0,
    min_inliers: int = 40,
    min_inlier_ratio: float = 0.5,
    search_stride: int = 1,
    max_db: Optional[int] = None,
) -> Tuple[List[MatchResult], Dict[int, List[int]]]:
    os.makedirs(out_dir, exist_ok=True)

    orb = create_orb()
    matcher = create_matcher()

    database: List[FrameFeatures] = []
    matches_found: List[MatchResult] = []
    groups: Dict[int, List[int]] = {}  # anchor_j -> list of i that matched

    for i, path in enumerate(frames):
        img = imread_rgb(path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp_i, desc_i = extract_orb(orb, gray)

        best: Optional[MatchResult] = None
        # Search previous frames honoring min_gap and stride
        j_candidates = list(range(0, max(0, i - min_gap), search_stride))
        for j in j_candidates:
            ref = database[j]
            good = knn_ratio_match(matcher, desc_i, ref.descriptors, ratio_thresh)
            if len(good) < min_inliers:  # quick reject
                continue
            inl, inl_ratio = geometric_verification(kp_i, ref.keypoints, good, ransac_thresh)
            if inl < min_inliers or inl_ratio < min_inlier_ratio:
                continue
            mr = MatchResult(i=i, j=j, score_inliers=inl, inlier_ratio=inl_ratio, good_matches=len(good))
            if best is None or (mr.score_inliers, mr.inlier_ratio) > (best.score_inliers, best.inlier_ratio):
                best = mr

        # Record and export evidence if we found a confident loop
        if best is not None:
            matches_found.append(best)
            # Save a side-by-side collage
            title = f"Loop i={best.i} ↔ j={best.j}  inliers={best.score_inliers}  ratio={best.inlier_ratio:.2f}"
            out_path = os.path.join(out_dir, f"loop_i{best.i:05d}_j{best.j:05d}_inl{best.score_inliers}.png")
            save_pair_collage(frames[best.j], frames[best.i], out_path, tile=320, title=title)
            # Group by anchor j
            groups.setdefault(best.j, []).append(best.i)

        # Append current to database
        database.append(FrameFeatures(idx=i, path=path, keypoints=kp_i, descriptors=desc_i))
        # Optional memory cap
        if max_db is not None and len(database) > max_db:
            # Drop oldest to keep window; adjust groups keys/indices unaffected since we keep absolute indices
            database.pop(0)

    # Export group collages (anchor + all matches)
    for anchor_j, idx_list in groups.items():
        paths = [frames[anchor_j]] + [frames[k] for k in sorted(idx_list)]
        title = f"Anchor j={anchor_j} with {len(idx_list)} revisits"
        out_path = os.path.join(out_dir, f"group_anchor_{anchor_j:05d}_N{len(paths)}.png")
        save_group_collage(paths, out_path, cols=6, tile=192, title=title)

    return matches_found, groups


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Online feature-based loop detection over a frames directory")
    ap.add_argument("--frames_dir", required=True, help="Directory with sequential frames (images)")
    ap.add_argument("--out_dir", required=True, help="Directory to save loop evidence and collages")
    ap.add_argument("--min_gap", type=int, default=50, help="Minimum index gap between matches")
    ap.add_argument("--ratio_thresh", type=float, default=0.75, help="Lowe ratio test threshold")
    ap.add_argument("--ransac_thresh", type=float, default=3.0, help="RANSAC reprojection threshold (pixels)")
    ap.add_argument("--min_inliers", type=int, default=40, help="Minimum RANSAC inliers to accept a loop")
    ap.add_argument("--min_inlier_ratio", type=float, default=0.5, help="Minimum inliers/ good-matches ratio")
    ap.add_argument("--search_stride", type=int, default=1, help="Check every k-th previous frame for speed")
    ap.add_argument("--max_db", type=int, default=None, help="Optional cap on database size (sliding window)")
    ap.add_argument("--save_summary", type=str, default=None, help="Optional JSON path to write summary results")
    args = ap.parse_args()

    frames = list_images(args.frames_dir)
    if not frames:
        raise SystemExit(f"No images found under {args.frames_dir}")

    matches, groups = detect_loops_online(
        frames,
        out_dir=args.out_dir,
        min_gap=args.min_gap,
        ratio_thresh=args.ratio_thresh,
        ransac_thresh=args.ransac_thresh,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        search_stride=args.search_stride,
        max_db=args.max_db,
    )

    # Estimate number of loops: count anchors with >= 1 revisit
    num_loops = sum(1 for _, lst in groups.items() if len(lst) >= 1)

    summary = {
        "frames_dir": os.path.abspath(args.frames_dir),
        "num_frames": len(frames),
        "min_gap": args.min_gap,
        "ratio_thresh": args.ratio_thresh,
        "ransac_thresh": args.ransac_thresh,
        "min_inliers": args.min_inliers,
        "min_inlier_ratio": args.min_inlier_ratio,
        "search_stride": args.search_stride,
        "num_candidates": len(matches),
        "num_loop_groups": num_loops,
        "groups": {int(k): [int(x) for x in v] for k, v in groups.items()},
        "detections": [
            {
                "i": int(m.i),
                "j": int(m.j),
                "score_inliers": int(m.score_inliers),
                "inlier_ratio": float(m.inlier_ratio),
                "good_matches": int(m.good_matches),
            }
            for m in matches
        ],
    }

    print(json.dumps({k: summary[k] for k in ("num_frames", "num_candidates", "num_loop_groups")}, indent=2))

    if args.save_summary:
        os.makedirs(os.path.dirname(args.save_summary) or ".", exist_ok=True)
        with open(args.save_summary, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary: {args.save_summary}")


if __name__ == "__main__":
    main()
