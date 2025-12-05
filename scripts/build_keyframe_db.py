import argparse
import os
import glob
import json
import numpy as np
from typing import List, Tuple

from feature_encoder import encode_paths


def list_images(frames_dir: str, exts=("*.png", "*.jpg", "*.jpeg", "*.bmp")) -> List[str]:
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(frames_dir, e)))
    files = sorted(files)
    return files


def main():
    ap = argparse.ArgumentParser(description="Build keyframe feature database from frames directory")
    ap.add_argument("--frames_dir", type=str, required=True, help="Directory containing frames (images)")
    ap.add_argument("--stride", type=int, default=5, help="Take every Nth frame (default: 5)")
    ap.add_argument(
        "--model",
        type=str,
        default="dinov2",
        help="Encoder: 'dinov2', 'clip-b32', or any timm model id compatible with num_classes=0",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output", type=str, default="keyframes_features.npy", help="Path to save features .npy")
    ap.add_argument("--index_out", type=str, default="keyframes_index.json", help="Path to save frame index JSON")
    args = ap.parse_args()

    imgs = list_images(args.frames_dir)
    if args.stride > 1:
        imgs = imgs[:: args.stride]

    if not imgs:
        raise SystemExit(f"No images found in {args.frames_dir}")

    feats, dim = encode_paths(imgs, model_name=args.model, batch_size=args.batch_size)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, feats)

    index = {
        "frames_dir": os.path.abspath(args.frames_dir),
        "stride": int(args.stride),
        "model": args.model,
        "embed_dim": int(dim),
        "num_frames": int(len(imgs)),
        "paths": imgs,
    }
    with open(args.index_out, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Saved features: {feats.shape} -> {args.output}")
    print(f"Saved index: {args.index_out}")


if __name__ == "__main__":
    main()
