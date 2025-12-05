import argparse
import glob
import json
import os
from collections import deque
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2 as cv

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
    return np.sign(x) * (np.abs(x) ** alpha)


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
    draw.rectangle([0, 0, W, overlay_h], fill=(0, 0, 0, 140))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(10, overlay_h - 6))
    except Exception:
        font = ImageFont.load_default()
    max_chars = max(8, int(W / 9))
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    draw.text((4, 2), text, fill=(255, 255, 255, 255), font=font)


def make_pairs_collage(pairs: List[Tuple[str, str, int, int, float, int, float]], out_path: str, tile: int = 192, title: Optional[str] = None) -> None:
    # pairs: (path_j, path_i, j, i, sim, inliers, inlier_ratio)
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

    for r, (pj, pi, j, i, s, ninl, rinl) in enumerate(pairs):
        for c, pth in enumerate([pj, pi]):
            try:
                im = Image.open(pth).convert("RGB")
            except Exception:
                im = Image.new("RGB", (tile, tile), (200, 200, 200))
            tile_img = letterbox(im, (tile, tile))
            name = os.path.basename(pth)
            lbl = f"idx={j if c==0 else i}  s={s:.2f}  inl={ninl} r={rinl:.2f}  {name}"
            draw_label(tile_img, lbl)
            canvas.paste(tile_img, (c * tile, margin_top + r * tile))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)


class FusionEncoder:
    def __init__(self, model_names: List[str], weights: List[float], device: Optional[str] = None, tta_flip=True, use_powerlaw=True):
        assert len(model_names) == len(weights)
        self.weights = np.array(weights, dtype=np.float32)
        self.weights /= (self.weights.sum() + 1e-8)
        self.models = []
        self.transforms = []
        self.device = device
        self.tta_flip = tta_flip
        self.use_powerlaw = use_powerlaw
        for name in model_names:
            m, tfm, dev, dim = get_model_and_transform(name, device=device)
            self.models.append(m)
            self.transforms.append(tfm)
            self.device = dev

    def encode(self, img: Image.Image) -> List[np.ndarray]:
        import torch
        out = []
        for m, tfm in zip(self.models, self.transforms):
            x = tfm(img).unsqueeze(0).to(self.device)
            if self.tta_flip:
                xf = tfm(img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(self.device)
                x = torch.cat([x, xf], dim=0)
            with torch.no_grad():
                f = m(x)
            f = f.mean(dim=0, keepdim=True)
            f = f.detach().cpu().numpy()
            if self.use_powerlaw:
                f = powerlaw(f)
            f = l2n(f)
            out.append(f.squeeze(0).astype(np.float32))
        return out

    def fused_sim(self, feats: List[np.ndarray], bank: List[List[np.ndarray]], max_j: int) -> Tuple[int, float]:
        S = None
        for w, f, rows in zip(self.weights, feats, bank):
            if max_j <= 0:
                continue
            if max_j > len(rows):
                max_j = len(rows)
            if max_j == 0:
                continue
            M = np.stack(rows[:max_j], axis=0)
            s = M @ f
            S = (w * s) if S is None else (S + w * s)
        if S is None or S.size == 0:
            return -1, -1.0
        j = int(np.argmax(S))
        return j, float(S[j])


class ORBVerifier:
    def __init__(self, nfeatures=2000):
        self.orb = cv.ORB_create(nfeatures=nfeatures)
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    def verify(self, img_path_a: str, img_path_b: str, min_inliers=30, ratio=0.75, ransac_thresh=3.0) -> Tuple[bool, int, float]:
        """
        Return (ok, inliers, inlier_ratio).
        - ok is False if we cannot compute a homography robustly or not enough matches.
        - Always require at least 4 correspondences before calling findHomography, regardless of min_inliers.
        """
        img_a = cv.imread(img_path_a, cv.IMREAD_GRAYSCALE)
        img_b = cv.imread(img_path_b, cv.IMREAD_GRAYSCALE)
        if img_a is None or img_b is None:
            return False, 0, 0.0
        kpa, da = self.orb.detectAndCompute(img_a, None)
        kpb, db = self.orb.detectAndCompute(img_b, None)
        if da is None or db is None or len(kpa) < 20 or len(kpb) < 20:
            return False, 0, 0.0
        matches = self.bf.knnMatch(da, db, k=2)
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        # Need at least 4 matches for homography estimation
        min_for_H = 4
        if len(good) < max(min_inliers, min_for_H):
            return False, len(good), 0.0
        pts_a = np.float32([kpa[m.queryIdx].pt for m in good])
        pts_b = np.float32([kpb[m.trainIdx].pt for m in good])
        try:
            H, mask = cv.findHomography(pts_a, pts_b, cv.RANSAC, ransac_thresh)
        except cv.error:
            return False, len(good), 0.0
        if mask is None:
            return False, len(good), 0.0
        inliers = int(mask.sum())
        ratio_inl = inliers / max(1, len(good))
        return inliers >= max(min_inliers, min_for_H) and ratio_inl >= 0.3, inliers, ratio_inl


def run_true_online(
    frames_dir: str,
    out_dir: str,
    stride: int,
    models: List[str],
    weights: List[float],
    min_gap: int,
    threshold: float,
    cooldown: int,
    cluster_window: int,
    tile: int,
    k_of_m: Tuple[int, int],
    mnn_window: int,
    verify: bool,
    min_inliers: int,
) -> Dict:
    paths = list_images(frames_dir)
    if stride > 1:
        paths = paths[::stride]
    if not paths:
        raise SystemExit(f"No images found in {frames_dir}")

    enc = FusionEncoder(models, weights)
    verifier = ORBVerifier() if verify else None

    # memory of features per model
    bank: List[List[np.ndarray]] = [[] for _ in models]

    events: List[Dict] = []
    last_fire = -10**9

    # queue for temporal consistency (keep recent events)
    M = k_of_m[1]
    K = k_of_m[0]
    recent = deque(maxlen=M)

    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        feats = enc.encode(img)
        max_j = max(0, i - min_gap)

        confirmed = None
        if (i - last_fire) >= cooldown and max_j > 0:
            j, s = enc.fused_sim(feats, bank, max_j=max_j)
            if j >= 0 and s >= threshold:
                # Mutual NN check within window around i
                # compute j's best match among [i-mnn_window, i]
                j_feats = [bank[k][j] for k in range(len(models))]
                lo = max(0, i - mnn_window)
                cand_feats = [np.stack(rows[lo:i], axis=0) if i - lo > 0 else None for rows in bank]
                if i - lo > 0:
                    S = None
                    for w, jf, rows in zip(enc.weights, j_feats, cand_feats):
                        svec = rows @ jf
                        S = w * svec if S is None else S + w * svec
                    # best in that window should be the last element (i-1) or near it; allow tolerance of 3 frames
                    best_rel = int(np.argmax(S))
                    best_i = lo + best_rel
                    mnn_ok = abs(best_i - i) <= 3
                else:
                    mnn_ok = True

                if mnn_ok:
                    # local geometric verification if requested
                    ok_geo, ninl, rinl = (True, 0, 0.0)
                    if verifier is not None:
                        ok_geo, ninl, rinl = verifier.verify(paths[j], paths[i], min_inliers=min_inliers)
                    if ok_geo:
                        candidate = {"i": i, "j": j, "score": float(s), "ninl": int(ninl), "rinl": float(rinl)}
                        recent.append(candidate)
                        # K-of-M confirmation: if at least K events in last M frames refer to similar j (within +/- mnn_window)
                        js = [e["j"] for e in recent]
                        if sum(abs(x - j) <= mnn_window for x in js) >= K:
                            confirmed = candidate
                            last_fire = i
                            recent.clear()
        # append feats to memory after query
        for k, f in enumerate(feats):
            bank[k].append(f)

        if confirmed is not None:
            events.append(confirmed)

    # cluster events by proximity in i
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

    # render collages
    loops = []
    for k, g in enumerate(groups, 1):
        # pick up to 12 top by score then inliers
        g_sorted = sorted(g, key=lambda x: (-x["score"], -x.get("ninl", 0)))
        pick = g_sorted[:12]
        pairs = [(paths[e["j"]], paths[e["i"]], e["j"], e["i"], e["score"], e.get("ninl", 0), e.get("rinl", 0.0)) for e in pick]
        span = g[-1]["i"] - g[0]["i"]
        out_png = os.path.join(out_dir, f"loop_{k:02d}_pairs{len(pairs)}_span{span}.png")
        if pairs:
            make_pairs_collage(pairs, out_png, tile=192, title=f"Loop {k}: K-of-M, MNN, ORB-RANSAC")
        loops.append({
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
        "k_of_m": list(k_of_m),
        "mnn_window": mnn_window,
        "verify": verify,
        "min_inliers": min_inliers,
        "detected_loops": len(loops),
        "loops": loops,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Strict online loop detection with DINOv2/CLIP fusion + temporal consistency + MNN + ORB-RANSAC verification")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=["dinov2", "clip-b32"])
    ap.add_argument("--weights", nargs="+", type=float, default=[0.7, 0.3])
    ap.add_argument("--min_gap", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.86)
    ap.add_argument("--cooldown", type=int, default=30)
    ap.add_argument("--cluster_window", type=int, default=120)
    ap.add_argument("--k_of_m", nargs=2, type=int, default=[2, 5], help="Require K detections in last M frames to confirm")
    ap.add_argument("--mnn_window", type=int, default=10, help="Mutual-NN temporal tolerance (frames)")
    ap.add_argument("--verify", action="store_true", help="Enable local ORB geometric verification")
    ap.add_argument("--min_inliers", type=int, default=30)
    args = ap.parse_args()

    summary = run_true_online(
        frames_dir=args.frames_dir,
        out_dir=args.out_dir,
        stride=args.stride,
        models=args.models,
        weights=args.weights,
        min_gap=args.min_gap,
        threshold=args.threshold,
        cooldown=args.cooldown,
        cluster_window=args.cluster_window,
        tile=192,
        k_of_m=tuple(args.k_of_m),
        mnn_window=args.mnn_window,
        verify=args.verify,
        min_inliers=args.min_inliers,
    )
    print(f"Detected loops: {summary['detected_loops']} -> {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
