import argparse
import json
import math
import os
from typing import List, Tuple, Dict, Set

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_paths(index_json: str) -> List[str]:
    with open(index_json, "r") as f:
        meta = json.load(f)
    paths = meta.get("paths", [])
    # Normalize to absolute paths for robustness
    abs_paths = [p if os.path.isabs(p) else os.path.abspath(p) for p in paths]
    return abs_paths


def compute_similarity(F: np.ndarray) -> np.ndarray:
    F = F.astype(np.float32)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    return F @ F.T  # [N, N]


def build_graph(S: np.ndarray, min_gap: int, threshold: float, topk: int) -> Dict[int, Set[int]]:
    N = S.shape[0]
    adj: Dict[int, Set[int]] = {i: set() for i in range(N)}
    for i in range(N):
        j_start = i + min_gap + 1
        if j_start >= N:
            continue
        sims = S[i, j_start:]
        # Candidate indices in global frame space
        cand_idx = np.argwhere(sims >= threshold).flatten()
        if cand_idx.size == 0:
            continue
        # Sort by similarity descending
        order = np.argsort(-sims[cand_idx])
        cand_idx = cand_idx[order][:topk]
        for off in cand_idx.tolist():
            j = j_start + off
            adj[i].add(j)
            adj[j].add(i)
    return adj


def connected_components(adj: Dict[int, Set[int]]) -> List[List[int]]:
    N = len(adj)
    visited = [False] * N
    comps: List[List[int]] = []
    for i in range(N):
        if visited[i]:
            continue
        if len(adj[i]) == 0:
            visited[i] = True
            continue
        q = [i]
        visited[i] = True
        comp = []
        while q:
            u = q.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        comps.append(sorted(comp))
    return comps


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


def draw_label(img: Image.Image, text: str, height: int = 20) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    overlay_h = min(height, max(16, H // 12))
    # semi-transparent black bar at top
    draw.rectangle([0, 0, W, overlay_h], fill=(0, 0, 0, 140))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(10, overlay_h - 6))
    except Exception:
        font = ImageFont.load_default()
    # clip overly long text
    max_chars = max(8, int(W / 8))
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    draw.text((4, 2), text, fill=(255, 255, 255, 255), font=font)


def make_collage(paths: List[str], out_path: str, cols: int = 6, tile: int = 192, title: str = None) -> None:
    if len(paths) == 0:
        return
    n = len(paths)
    rows = math.ceil(n / cols)
    margin_top = 0
    title_h = 28
    if title:
        margin_top = title_h + 6
    canvas = Image.new("RGB", (cols * tile, rows * tile + margin_top), (240, 240, 240))

    if title:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=16)
        except Exception:
            font = ImageFont.load_default()
        draw.text((6, 4), title, fill=(20, 20, 20), font=font)

    for idx, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            # placeholder for unreadable image
            img = Image.new("RGB", (tile, tile), (200, 200, 200))
        tile_img = letterbox(img, (tile, tile))
        draw_label(tile_img, os.path.basename(p))
        r = idx // cols
        c = idx % cols
        canvas.paste(tile_img, (c * tile, margin_top + r * tile))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser(description="Find groups of similar non-consecutive frames and save collages")
    ap.add_argument("--features", required=True, help="Path to features .npy (N, D)")
    ap.add_argument("--index", required=True, help="Path to index JSON with 'paths'")
    ap.add_argument("--output_dir", required=True, help="Directory to save collages")
    ap.add_argument("--min_gap", type=int, default=50, help="Minimum index gap to consider (non-consecutive)")
    ap.add_argument("--threshold", type=float, default=0.8, help="Cosine similarity threshold")
    ap.add_argument("--topk", type=int, default=5, help="For each frame, keep top-K later matches")
    ap.add_argument("--max_groups", type=int, default=20, help="Maximum number of groups to export")
    ap.add_argument("--max_per_group", type=int, default=30, help="Limit images per collage to avoid huge grids")
    ap.add_argument("--cols", type=int, default=6, help="Number of columns in collage")
    ap.add_argument("--tile", type=int, default=192, help="Tile size (pixels)")
    args = ap.parse_args()

    F = np.load(args.features)
    paths = load_paths(args.index)
    if F.shape[0] != len(paths):
        raise SystemExit(f"Mismatch: features N={F.shape[0]} vs paths={len(paths)}")

    S = compute_similarity(F)
    adj = build_graph(S, min_gap=args.min_gap, threshold=args.threshold, topk=args.topk)
    comps = connected_components(adj)

    # Score and sort components: by size desc, then temporal span desc, then average internal similarity
    scored: List[Tuple[float, List[int]]] = []
    for comp in comps:
        if len(comp) < 2:
            continue
        span = comp[-1] - comp[0]
        # compute mean pairwise sim within comp (upper triangle)
        sub = S[np.ix_(comp, comp)]
        tri = sub[np.triu_indices(len(comp), k=1)]
        mean_sim = float(tri.mean()) if tri.size else 0.0
        score = (len(comp), span, mean_sim)
        # Pack into a single scalar for sorting tuple-wise
        scored.append((score, comp))

    scored.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]), reverse=True)

    os.makedirs(args.output_dir, exist_ok=True)

    exported = 0
    for rank, (score, comp) in enumerate(scored):
        if exported >= args.max_groups:
            break
        # Subsample if too large
        if len(comp) > args.max_per_group:
            step = max(1, len(comp) // args.max_per_group)
            comp = comp[::step][: args.max_per_group]
        comp_paths = [paths[i] for i in comp]
        title = f"Group {exported+1}: N={len(comp_paths)} span={comp[-1]-comp[0]} indices={comp[0]}..{comp[-1]}"
        out_name = f"group_{exported+1:02d}_N{len(comp_paths)}_span{comp[-1]-comp[0]}.png"
        out_path = os.path.join(args.output_dir, out_name)
        make_collage(comp_paths, out_path, cols=args.cols, tile=args.tile, title=title)
        exported += 1

    print(f"Exported {exported} group collages to {args.output_dir}")


if __name__ == "__main__":
    main()
