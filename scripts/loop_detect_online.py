import argparse
import json
import math
import os
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def l2_normalize(F: np.ndarray) -> np.ndarray:
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)


def load_index(index_json: str) -> List[str]:
    with open(index_json, "r") as f:
        meta = json.load(f)
    paths = meta.get("paths", [])
    return [p if os.path.isabs(p) else os.path.abspath(p) for p in paths]


def draw_label(img: Image.Image, text: str, height: int = 20) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    overlay_h = min(height, max(16, H // 12))
    draw.rectangle([0, 0, W, overlay_h], fill=(0, 0, 0, 140))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(10, overlay_h - 6))
    except Exception:
        font = ImageFont.load_default()
    max_chars = max(8, int(W / 7))
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    draw.text((4, 2), text, fill=(255, 255, 255, 255), font=font)


def letterbox(img: Image.Image, size: Tuple[int, int], fill=(255, 255, 255)) -> Image.Image:
    w, h = img.size
    W, H = size
    scale = min(W / w, H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (W, H), fill)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    canvas.paste(img_resized, (ox, oy))
    return canvas


def make_pairs_collage(pairs: List[Tuple[str, str, int, int, float]], out_path: str, tile: int = 192) -> None:
    # Pairs: list of (path_j, path_i, j, i, score)
    rows = len(pairs)
    cols = 2
    margin_top = 24
    canvas = Image.new("RGB", (cols * tile, rows * tile + margin_top), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 4), f"Pairs: {rows} (left=past, right=current)", fill=(20, 20, 20), font=font)

    for r, (pj, pi, j, i, s) in enumerate(pairs):
        for c, pth in enumerate([pj, pi]):
            try:
                im = Image.open(pth).convert("RGB")
            except Exception:
                im = Image.new("RGB", (tile, tile), (200, 200, 200))
            tile_img = letterbox(im, (tile, tile))
            name = os.path.basename(pth)
            lbl = f"idx={j if c==0 else i}  s={s:.2f}  {name}"
            draw_label(tile_img, lbl)
            canvas.paste(tile_img, (c * tile, margin_top + r * tile))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)


def online_loop_detect(F: np.ndarray, min_gap: int, threshold: float, cooldown: int) -> List[Dict]:
    # Assumes F is L2-normalized
    N = F.shape[0]
    events = []  # list of dicts: {i, j, score}
    last_fire = -1e9
    for i in range(N):
        if i - min_gap <= 0:
            continue
        # cooldown to avoid duplicates in immediate vicinity
        if i - last_fire < cooldown:
            continue
        past = F[: i - min_gap]  # [M, D]
        if past.shape[0] == 0:
            continue
        sims = past @ F[i]  # [M]
        j = int(np.argmax(sims))
        s = float(sims[j])
        if s >= threshold:
            events.append({"i": i, "j": j, "score": s})
            last_fire = i
    return events


def cluster_events(events: List[Dict], window: int) -> List[List[Dict]]:
    if not events:
        return []
    groups = []
    cur = [events[0]]
    for e in events[1:]:
        if e["i"] - cur[-1]["i"] <= window:
            cur.append(e)
        else:
            groups.append(cur)
            cur = [e]
    groups.append(cur)
    return groups


def pick_representatives(group: List[Dict], max_pairs: int) -> List[Dict]:
    # Take top-by-score, then ensure some temporal spread in i
    group_sorted = sorted(group, key=lambda x: (-x["score"], x["i"]))
    picked = []
    used_i = []
    for e in group_sorted:
        if not used_i or min(abs(e["i"] - u) for u in used_i) > 5:
            picked.append(e)
            used_i.append(e["i"])
        if len(picked) >= max_pairs:
            break
    return picked


def main():
    ap = argparse.ArgumentParser(description="Online loop detection over feature stream; outputs collages per loop episode")
    ap.add_argument("--features", required=True, help="Path to features .npy (N,D)")
    ap.add_argument("--index", required=True, help="Path to index .json with frame 'paths'")
    ap.add_argument("--out_dir", required=True, help="Output directory for loop collages and summary")
    ap.add_argument("--min_gap", type=int, default=50, help="Minimum frame gap before considering as loop match")
    ap.add_argument("--threshold", type=float, default=0.82, help="Cosine similarity threshold")
    ap.add_argument("--cooldown", type=int, default=25, help="Frames to suppress after a detection")
    ap.add_argument("--cluster_window", type=int, default=100, help="Group detections within this window as one loop episode")
    ap.add_argument("--max_pairs_per_loop", type=int, default=10, help="Max pairs to render per loop collage")
    ap.add_argument("--tile", type=int, default=192, help="Tile size for collage")
    ap.add_argument("--expected_loops", type=int, default=None, help="If provided, compare with detected loops")
    args = ap.parse_args()

    F = np.load(args.features)
    paths = load_index(args.index)
    assert F.shape[0] == len(paths), f"Feature count {F.shape[0]} != index paths {len(paths)}"

    F = l2_normalize(F.astype(np.float32))

    events = online_loop_detect(F, min_gap=args.min_gap, threshold=args.threshold, cooldown=args.cooldown)
    groups = cluster_events(events, window=args.cluster_window)

    os.makedirs(args.out_dir, exist_ok=True)

    # Save collages for each loop episode
    summary = []
    for k, g in enumerate(groups, 1):
        reps = pick_representatives(g, max_pairs=args.max_pairs_per_loop)
        pairs = []
        for e in reps:
            i, j, s = e["i"], e["j"], e["score"]
            pairs.append((paths[j], paths[i], j, i, s))
        if not pairs:
            continue
        span = g[-1]["i"] - g[0]["i"]
        out_png = os.path.join(args.out_dir, f"loop_{k:02d}_pairs{len(pairs)}_span{span}.png")
        make_pairs_collage(pairs, out_png, tile=args.tile)
        summary.append({
            "loop_id": k,
            "num_pairs": len(pairs),
            "first_i": int(g[0]["i"]),
            "last_i": int(g[-1]["i"]),
            "span": int(span),
            "max_score": float(max(e["score"] for e in g)),
            "image": out_png,
        })

    # Save summary JSON and print a short report
    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump({
            "features": os.path.abspath(args.features),
            "index": os.path.abspath(args.index),
            "min_gap": args.min_gap,
            "threshold": args.threshold,
            "cooldown": args.cooldown,
            "cluster_window": args.cluster_window,
            "detected_loops": len(summary),
            "loops": summary,
        }, f, indent=2)

    print(f"Detected loops: {len(summary)} -> {out_json}")
    if args.expected_loops is not None:
        ok = (len(summary) == args.expected_loops)
        print(f"Expected {args.expected_loops}, detected {len(summary)} -> {'MATCH' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
