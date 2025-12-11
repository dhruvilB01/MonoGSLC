import argparse
import glob
import json
import os
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from netvlad_encoder import get_netvlad_model  # noqa: E402


def list_images(frames_dir: str, exts=("*.png", "*.jpg", "*.jpeg", "*.bmp")) -> List[str]:
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(frames_dir, e)))
    files = sorted(files)
    return files


def l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / n


def draw_label(img: Image.Image, text: str, height: int = 20) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    overlay_h = min(height, max(16, H // 12))
    draw.rectangle([0, 0, W, overlay_h], fill=(0, 0, 0, 140))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(10, overlay_h - 6))
    except Exception:
        font = ImageFont.load_default()
    max_chars = max(8, int(W / 9))
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


def online_loop_detect(
    encoder,
    transform,
    device: str,
    paths: List[str],
    min_gap: int,
    threshold: float,
    cooldown: int,
) -> List[Dict]:
    events = []
    last_fire = -10**9
    feats: List[np.ndarray] = []
    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        with torch.no_grad():
            x = transform(img).unsqueeze(0).to(device)
            f = encoder(x).cpu().numpy().squeeze(0)
            f = l2n(f)
        if i - min_gap <= 0:
            feats.append(f)
            continue
        if i - last_fire < cooldown:
            feats.append(f)
            continue
        past = np.stack(feats[: i - min_gap], axis=0) if (i - min_gap) > 0 else None
        if past is None or past.shape[0] == 0:
            feats.append(f)
            continue
        sims = past @ f  # [M]
        j = int(np.argmax(sims))
        s = float(sims[j])
        if s >= threshold:
            events.append({"i": i, "j": j, "score": s})
            last_fire = i
        feats.append(f)
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
    ap = argparse.ArgumentParser(description="Online loop detection with NetVLAD global descriptors")
    ap.add_argument("--frames_dir", required=True, help="Directory of frames to stream")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out_dir", required=True, help="Output directory for collages and summary")
    ap.add_argument("--min_gap", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.82)
    ap.add_argument("--cooldown", type=int, default=25)
    ap.add_argument("--cluster_window", type=int, default=100)
    ap.add_argument("--max_pairs_per_loop", type=int, default=10)
    ap.add_argument("--tile", type=int, default=192)
    ap.add_argument("--num_clusters", type=int, default=64)
    ap.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--no_pretrained_backbone", action="store_true", help="Do not load ImageNet weights for backbone")
    args = ap.parse_args()

    paths = list_images(args.frames_dir)
    if args.stride > 1:
        paths = paths[:: args.stride]
    if not paths:
        raise SystemExit(f"No images in {args.frames_dir}")

    encoder, transform, device, dim = get_netvlad_model(
        num_clusters=args.num_clusters,
        backbone_name=args.backbone,
        pretrained_backbone=not args.no_pretrained_backbone,
    )

    events = online_loop_detect(
        encoder=encoder,
        transform=transform,
        device=device,
        paths=paths,
        min_gap=args.min_gap,
        threshold=args.threshold,
        cooldown=args.cooldown,
    )
    groups = cluster_events(events, window=args.cluster_window)

    os.makedirs(args.out_dir, exist_ok=True)

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
        summary.append(
            {
                "loop_id": k,
                "num_pairs": len(pairs),
                "first_i": int(g[0]["i"]),
                "last_i": int(g[-1]["i"]),
                "span": int(span),
                "max_score": float(max(e["score"] for e in g)),
                "image": out_png,
            }
        )

    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "frames_dir": os.path.abspath(args.frames_dir),
                "stride": args.stride,
                "model": f"netvlad-{args.backbone}-{args.num_clusters}",
                "min_gap": args.min_gap,
                "threshold": args.threshold,
                "cooldown": args.cooldown,
                "cluster_window": args.cluster_window,
                "detected_loops": len(summary),
                "loops": summary,
                "embed_dim": dim,
                "detections": [
                    {"i": int(e["i"]), "j": int(e["j"]), "score": float(e["score"])}
                    for e in events
                ],
            },
            f,
            indent=2,
        )
    print(f"Detected loops: {len(summary)} -> {out_json}")


if __name__ == "__main__":
    main()
