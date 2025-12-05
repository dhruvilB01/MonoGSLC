import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="Plot cosine similarity heatmap from feature matrix")
    ap.add_argument("--features", type=str, required=True, help="Path to features .npy of shape (N, D)")
    ap.add_argument("--output", type=str, default=None, help="Output image path (PNG)")
    ap.add_argument("--title", type=str, default=None, help="Optional plot title")
    ap.add_argument("--show", action="store_true", help="Show the plot interactively")
    args = ap.parse_args()

    F = np.load(args.features)
    if F.ndim != 2:
        raise SystemExit(f"Expected 2D array, got shape {F.shape}")

    # L2-normalize rows, then cosine similarity = F_norm @ F_norm^T
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    S = F @ F.T

    plt.figure(figsize=(6, 5), dpi=150)
    im = plt.imshow(S, cmap="magma", vmin=-1, vmax=1, origin="upper")
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label("Cosine similarity")
    plt.xlabel("Frame index")
    plt.ylabel("Frame index")
    if args.title:
        plt.title(args.title)
    else:
        plt.title(f"Similarity matrix (N={F.shape[0]})")
    plt.tight_layout()

    out = args.output or os.path.splitext(args.features)[0] + "_similarity.png"
    plt.savefig(out)
    if args.show:
        plt.show()
    print(f"Saved heatmap: {out}")


if __name__ == "__main__":
    main()
