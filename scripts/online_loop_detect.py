import argparse
import glob
import json
import os
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Allow importing helper in same scripts folder
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from feature_encoder import get_model_and_transform  # noqa: E402


def list_images(frames_dir: str, exts=("*.png", "*.jpg", "*.jpeg", "*.bmp")) -> List[str]:
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(frames_dir, e)))
    files = sorted(files)
    return files


def l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / n


def powerlaw(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    # Signed power-law normalization; improves robustness
    return np.sign(x) * (np.abs(x) ** alpha)


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


def make_pairs_collage(pairs: List[Tuple[str, str, int, int, float]], out_path: str, tile: int = 192, title: Optional[str] = None) -> None:
    rows = len(pairs)
    cols = 2
    margin_top = 0
    if title:
        margin_top = 26
    canvas = Image.new("RGB", (cols * tile, rows * tile + margin_top), (240, 240, 240))
    if title:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=16)
        except Exception:
            font = ImageFont.load_default()
        draw.text((6, 4), title, fill=(20, 20, 20), font=font)

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


class OnlinePlaceRec:
    def __init__(
        self,
        model_names: List[str],
        weights: List[float],
        device: Optional[str] = None,
        use_powerlaw: bool = True,
        tta_flip: bool = True,
    ) -> None:
        assert len(model_names) == len(weights)
        self.models = []
        self.transforms = []
        self.embed_dims = []
        self.device = device
        self.use_powerlaw = use_powerlaw
        self.tta_flip = tta_flip
        # Normalize weights to sum to 1
        w = np.array(weights, dtype=np.float32)
        w = w / (w.sum() + 1e-8)
        self.weights = w
        for name in model_names:
            m, tfm, dev, dim = get_model_and_transform(name, device=device)
            self.models.append(m)
            self.transforms.append(tfm)
            self.embed_dims.append(dim)
            self.device = dev
        # Memory of features per model: list of np arrays [N_k, D_k]
        self.memory: List[List[np.ndarray]] = [[] for _ in model_names]

    @staticmethod
    def _to_numpy(t):
        return t.detach().cpu().numpy()

    def encode(self, img: Image.Image) -> List[np.ndarray]:
        import torch
        outs = []
        for m, tfm in zip(self.models, self.transforms):
            im = img.convert("RGB")
            x = tfm(im).unsqueeze(0).to(self.device)
            if self.tta_flip:
                im_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
                x2 = tfm(im_flip).unsqueeze(0).to(self.device)
                x = torch.cat([x, x2], dim=0)
            with torch.no_grad():
                f = m(x)  # [B, D]
            f = f.mean(dim=0, keepdim=True)  # TTA average
            f = self._to_numpy(f)
            if self.use_powerlaw:
                f = powerlaw(f)
            f = l2n(f)
            outs.append(f.squeeze(0))  # [D]
        return outs

    def add_to_memory(self, feats: List[np.ndarray]) -> None:
        for k, f in enumerate(feats):
            self.memory[k].append(f.astype(np.float32))

    def best_match(self, feats: List[np.ndarray], max_j: int) -> Tuple[int, float]:
        # Compute weighted fused similarity over models
        best_j = -1
        best_s = -1.0
        for k, (f, bank) in enumerate(zip(feats, self.memory)):
            if max_j <= 0:
                s_k = None
            else:
                M = np.stack(bank[:max_j], axis=0)  # [M, D]
                s_k = (M @ f)  # [M]
            if k == 0:
                if s_k is None:
                    S = None
                else:
                    S = self.weights[k] * s_k
            else:
                if s_k is not None:
                    S += self.weights[k] * s_k
        if S is None or S.size == 0:
            return -1, -1.0
        j = int(np.argmax(S))
        s = float(S[j])
        return j, s


def run_online(
    frames_dir: str,
    stride: int,
    models: List[str],
    weights: List[float],
    out_dir: str,
    min_gap: int,
    threshold: float,
    cooldown: int,
    cluster_window: int,
    tile: int,
) -> Dict:
    paths = list_images(frames_dir)
    if stride > 1:
        paths = paths[::stride]
    if not paths:
        raise SystemExit(f"No images in {frames_dir}")

    opr = OnlinePlaceRec(models, weights)

    last_fire = -10**9
    events: List[Dict] = []

    # Online processing
    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        feats = opr.encode(img)
        # Search only in memory before current frame respecting min_gap
        max_j = max(0, i - min_gap)
        if (i - last_fire) >= cooldown and max_j > 0:
            j, s = opr.best_match(feats, max_j=max_j)
            if j >= 0 and s >= threshold:
                events.append({"i": i, "j": j, "score": s})
                last_fire = i
        # Add current frame to memory after querying
        opr.add_to_memory(feats)

    # Cluster events online-style (single pass grouping by proximity in i)
    groups: List[List[Dict]] = []
    if events:
        cur = [events[0]]
        for e in events[1:]:
            if e["i"] - cur[-1]["i"] <= cluster_window:
                cur.append(e)
            else:
                groups.append(cur)
                cur = [e]
        groups.append(cur)

    os.makedirs(out_dir, exist_ok=True)

    # Render collages and summary
    loops_summary = []
    for k, g in enumerate(groups, 1):
        # pick up to 12 top-scoring, spaced by >=5 frames on i
        g_sorted = sorted(g, key=lambda x: (-x["score"], x["i"]))
        picked = []
        used_i = []
        for e in g_sorted:
            if not used_i or min(abs(e["i"] - u) for u in used_i) > 5:
                picked.append(e)
                used_i.append(e["i"])
            if len(picked) >= 12:
                break
        pairs = [(paths[e["j"]], paths[e["i"]], e["j"], e["i"], e["score"]) for e in picked]
        span = g[-1]["i"] - g[0]["i"]
        title = f"Loop {k}: N={len(pairs)} span={span} i:[{g[0]['i']}..{g[-1]['i']}]"
        out_png = os.path.join(out_dir, f"loop_{k:02d}_pairs{len(pairs)}_span{span}.png")
        if pairs:
            make_pairs_collage(pairs, out_png, tile=tile, title=title)
        loops_summary.append({
            "loop_id": k,
            "num_pairs": len(pairs),
            "first_i": int(g[0]["i"]),
            "last_i": int(g[-1]["i"]),
            "span": int(span),
            "max_score": float(max(e["score"] for e in g)),
            "image": out_png if pairs else None,
        })

    summary = {
        "frames_dir": os.path.abspath(frames_dir),
        "stride": stride,
        "models": models,
        "weights": [float(w) for w in weights],
        "min_gap": min_gap,
        "threshold": threshold,
        "cooldown": cooldown,
        "cluster_window": cluster_window,
        "detected_loops": len(loops_summary),
        "loops": loops_summary,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Truly online loop detection: encode frames on-the-fly, query memory, output loop collages")
    ap.add_argument("--frames_dir", required=True, help="Directory of frames to stream")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=["dinov2"], help="One or more encoders: e.g., dinov2 clip-b32")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="Weights for each model (same length as models)")
    ap.add_argument("--out_dir", required=True, help="Output directory for collages and summary")
    ap.add_argument("--min_gap", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.86)
    ap.add_argument("--cooldown", type=int, default=30)
    ap.add_argument("--cluster_window", type=int, default=120)
    ap.add_argument("--tile", type=int, default=192)
    args = ap.parse_args()

    if args.weights is None:
        args.weights = [1.0 / len(args.models)] * len(args.models)
    else:
        assert len(args.weights) == len(args.models), "weights must match models"

    summary = run_online(
        frames_dir=args.frames_dir,
        stride=args.stride,
        models=args.models,
        weights=args.weights,
        out_dir=args.out_dir,
        min_gap=args.min_gap,
        threshold=args.threshold,
        cooldown=args.cooldown,
        cluster_window=args.cluster_window,
        tile=args.tile,
    )
    print(f"Detected loops: {summary['detected_loops']} -> {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
