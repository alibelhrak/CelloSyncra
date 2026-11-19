"""idr0109: how do the masks (e.g. 217 x 217) sit on the single-cell crops (256 x 256)?

For a few cells: searches scale and shift that best put each mask on the cell in its crop, prints the result,
and saves a picture (crop | mask | overlay with the best fit) to check by eye.

  python idr_maskcheck.py --data idr0109 --n 12 --out maskcheck.png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import idr_build as IB  # noqa: E402


def cellness(img):
    """where the cell is in a phase-contrast crop: local texture (std) and edges, smoothed."""
    from scipy.ndimage import gaussian_filter, sobel, uniform_filter
    x = img.astype(np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    m = uniform_filter(x, 7)
    sd = np.sqrt(np.clip(uniform_filter(x * x, 7) - m * m, 0, None))
    g = np.hypot(sobel(x, 0), sobel(x, 1))
    c = gaussian_filter(sd + 0.5 * g, 2)
    return (c - c.mean()) / (c.std() + 1e-9)


def best_fit(img, m, scales):
    """search scale (mask resized by s) and shift: the placement whose outline lies on the strongest image edges
    (mean edge strength along the outline, via FFT correlation, so short and long outlines compare fairly)."""
    from scipy.ndimage import sobel, gaussian_filter
    from skimage.transform import resize
    from skimage.segmentation import find_boundaries
    x = img.astype(np.float64)
    G = gaussian_filter(np.hypot(sobel(x, 0), sobel(x, 1)), 1.0)
    G = G / (G.mean() + 1e-9)
    H, W = img.shape
    best = None
    for s in scales:
        h, w = max(4, int(round(m.shape[0] * s))), max(4, int(round(m.shape[1] * s)))
        ms = resize(m.astype(np.float64), (h, w), order=0, preserve_range=True, anti_aliasing=False) > 0.5
        B = find_boundaries(ms, mode="inner").astype(np.float64)
        if B.sum() < 8:
            continue
        P = np.zeros((H + h, W + w)); P[:h, :w] = B
        Gp = np.zeros_like(P); Gp[:H, :W] = G
        corr = np.real(np.fft.ifft2(np.fft.fft2(Gp) * np.conj(np.fft.fft2(P)))) / B.sum()
        # only placements where the whole outline stays inside the crop (no credit for the padding)
        ys = np.arange(P.shape[0]); xs = np.arange(P.shape[1])
        yy = np.where(ys < H, ys, ys - (H + h)); xx = np.where(xs < W, xs, xs - (W + w))
        by, bx = np.nonzero(B)
        okY = (yy + by.min() >= 0) & (yy + by.max() < H)
        okX = (xx + bx.min() >= 0) & (xx + bx.max() < W)
        corr = np.where(okY[:, None] & okX[None, :], corr, -np.inf)
        idx = np.unravel_index(np.argmax(corr), corr.shape)
        sc = float(corr[idx])
        if best is None or sc > best[0]:
            best = (sc, s, int(yy[idx[0]]), int(xx[idx[1]]), (h, w))
    return best


def place(m, shape, s, y0, x0):
    from skimage.transform import resize
    h, w = max(4, int(round(m.shape[0] * s))), max(4, int(round(m.shape[1] * s)))
    ms = resize(m.astype(np.float64), (h, w), order=0, preserve_range=True, anti_aliasing=False) > 0.5
    out = np.zeros(shape, bool)
    H, W = shape
    sy0, sx0 = max(0, -y0), max(0, -x0)
    dy0, dx0 = max(0, y0), max(0, x0)
    hh, ww = min(h - sy0, H - dy0), min(w - sx0, W - dx0)
    if hh > 0 and ww > 0:
        out[dy0:dy0 + hh, dx0:dx0 + ww] = ms[sy0:sy0 + hh, sx0:sx0 + ww]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the idr0109 folder (with 20210408-ftp, 20210416-ftp, unpacked)")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="maskcheck.png")
    a = ap.parse_args()
    D = Path(a.data)
    roi = pd.read_csv(D / "20210408-ftp/ROI/ImageFileTable_ROI.txt", sep="\t", nrows=400000,
                      usecols=["cellType", "expStr", "frame", "xs", "ys", "imageFilePathROI"], dtype={"expStr": str})
    mk = pd.read_csv(D / "20210416-ftp/ImageFileTable_roiMask_15APR2021.txt", sep="\t", nrows=400000,
                     usecols=["expStr", "frame", "xs", "ys", "imageFilePathMask"], dtype={"expStr": str})
    roi["track"] = roi["imageFilePathROI"].map(lambda p: os.path.basename(os.path.dirname(p)))
    mk["track"] = mk["imageFilePathMask"].map(lambda p: os.path.basename(os.path.dirname(p)))
    roi["key"] = roi["expStr"] + "|" + roi["track"].map(IB._tail_key) + "|" + roi["frame"].astype(str)
    mk["key"] = mk["expStr"] + "|" + mk["track"].map(IB._tail_key) + "|" + mk["frame"].astype(str)
    j = roi.merge(mk[["key", "xs", "ys", "imageFilePathMask"]], on="key", suffixes=("", "_m"))
    print(f"rows read: crops {len(roi):,}, masks {len(mk):,}, matched {len(j):,}")
    print("same centre in both tables (xs, ys):", bool(np.allclose(j["xs"], j["xs_m"]) and np.allclose(j["ys"], j["ys_m"])))
    rr = IB.Resolver(D / "unpacked/roi", j["imageFilePathROI"].head(50))
    mr = IB.Resolver(D / "unpacked/mask", j["imageFilePathMask"].head(50))
    pick = j.groupby("track", sort=False).head(1).sample(min(a.n, j["track"].nunique()), random_state=0)
    scales = [256 / 217, 1.0, 0.9, 1.1, 1.25, 0.8, 1.4, 0.7, 0.5, 2.0]
    rows, panels = [], []
    for _, r in pick.iterrows():
        img, m = IB._read(rr(r["imageFilePathROI"])), IB._read(mr(r["imageFilePathMask"])) > 0
        sc, s, y0, x0, hw = best_fit(img, m, scales)
        fit = place(m, img.shape, s, y0, x0)
        cy, cx = (img.shape[0] - hw[0]) / 2, (img.shape[1] - hw[1]) / 2
        rows.append(dict(cell=r["cellType"], crop=img.shape, mask=m.shape, mask_fill=round(float(m.mean()), 3),
                         scale=round(s, 3), dy_from_centred=round(y0 - cy, 1), dx_from_centred=round(x0 - cx, 1),
                         edge_score_fit=round(IB.edge_score(img, fit), 2),
                         edge_score_center=round(IB.edge_score(img, IB.fit_mask(m, img.shape, "center")), 2),
                         edge_score_resize=round(IB.edge_score(img, IB.fit_mask(m, img.shape, "resize")), 2)))
        panels.append((img, m, fit, r["cellType"]))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print("\nmedian best scale", df["scale"].median(), "| median shift from centred (dy, dx):",
          df["dy_from_centred"].median(), df["dx_from_centred"].median())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    fig, ax = plt.subplots(len(panels), 3, figsize=(7.5, 2.5 * len(panels)), squeeze=False)
    for i, (img, m, fit, ct) in enumerate(panels):
        lo, hi = np.percentile(img, [1, 99])
        v = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
        ax[i, 0].imshow(v, cmap="gray"); ax[i, 0].set_ylabel(ct, fontsize=8)
        ax[i, 1].imshow(m, cmap="gray")
        rgb = np.dstack([v] * 3); rgb[find_boundaries(fit, mode="inner")] = (1, 0.2, 0.2)
        rgb[find_boundaries(IB.fit_mask(m, img.shape, "center"), mode="inner")] = (0.2, 0.8, 1)
        ax[i, 2].imshow(rgb)
        for k in range(3):
            ax[i, k].set_xticks([]); ax[i, k].set_yticks([])
        if i == 0:
            ax[i, 0].set_title("crop", fontsize=9); ax[i, 1].set_title("mask", fontsize=9)
            ax[i, 2].set_title("best fit (red), centred (blue)", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=90)
    print("picture:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
