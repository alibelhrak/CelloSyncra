"""idr0109 (Zaritsky et al. 2021, melanoma metastatic efficiency): one feature row per tracked cell.

Input (from the IDR FTP, unpacked):
  ImageFileTable_ROI.txt            one row per cell per frame: cellType, metEff, date, expStr, locationStr, frame, ts, xs, ys,
                                    imageFilePathROI (single-cell crop, png)
  ImageFileTable_roiMask_*.txt      same rows for the cell masks (tif)
  imageROIs.tar.bz2 / masks.tgz     the crops and masks themselves

Per frame: shape (from the mask), brightness and texture inside the cell (from the crop), position (from the table).
Per cell (track): mean / spread / frame-to-frame change of every per-frame value, plus movement (speed,
straightness, MSD exponent). Time-averaged descriptors are what the original paper found most predictive.

Output: <out>/tracks.csv.gz  (one row per cell track) and <out>/build_log.json

  python idr_build.py --roi-table .../ImageFileTable_ROI.txt --mask-table .../ImageFileTable_roiMask_15APR2021.txt \
      --roi-root <folder where imageROIs.tar.bz2 was unpacked> --mask-root <folder where masks.tgz was unpacked> \
      --out out/idr_features --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

FRAME_MIN = 1.0          # minutes between frames (paper: one frame per minute)
META = ["track", "cellType", "metEff", "date", "expStr", "locationStr", "n_frames", "t_start", "n_masked"]


# ----------------------------------------------------------------------------- paths
def _tail_key(folder: str) -> str:
    """'140604_mv3_s01_t184_x1112_y1399_t308' and '22-Feb-2018_mv3_s01_t184_x1112_y1399_t308' -> 'mv3_s01_t184_x1112_y1399_t308'"""
    return folder.split("_", 1)[1] if "_" in folder else folder


def _frame_no(name: str) -> int:
    m = re.search(r"_f(\d+)\.\w+$", name)
    return int(m.group(1)) if m else -1


class Resolver:
    """the tables hold paths like data/ROI/png/...; the archive may unpack with or without the leading folders.
    Finds once which leading part to drop (or add) so that root/<path> exists."""

    def __init__(self, root: Path, sample_paths):
        self.root, self.drop = Path(root), None
        for p in sample_paths:
            parts = Path(p).parts
            for k in range(len(parts)):
                if (self.root.joinpath(*parts[k:])).exists():
                    self.drop = k
                    return
        self.found = False

    def __call__(self, p):
        if self.drop is None:
            return None
        return self.root.joinpath(*Path(p).parts[self.drop:])


# ----------------------------------------------------------------------------- per frame
def _read(p):
    from skimage.io import imread
    a = imread(str(p))
    if a.ndim == 3:
        a = a[..., :3].mean(-1) if a.shape[-1] in (3, 4) else a[0]
    return a.astype(np.float32)


def frame_features(img, mask):
    from skimage.measure import label, regionprops
    from scipy.ndimage import laplace, sobel
    m = mask > 0
    if m.shape != img.shape or m.sum() < 10:
        return None
    lab = label(m)
    if lab.max() > 1:                                    # keep the component under the crop centre, else the largest
        c = lab[lab.shape[0] // 2, lab.shape[1] // 2]
        if c == 0:
            c = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        m = lab == c
    rp = regionprops(m.astype(np.uint8), intensity_image=img)[0]
    ax_maj = float(getattr(rp, "axis_major_length", None) or rp.major_axis_length)
    ax_min = float(getattr(rp, "axis_minor_length", None) or rp.minor_axis_length)
    inside, outside = img[m], img[~m]
    bg_mu = float(outside.mean()) if outside.size else 0.0
    bg_sd = float(outside.std()) if outside.size > 1 else 1.0
    gm = np.hypot(sobel(img, 0), sobel(img, 1))
    lap = laplace(img)
    per = float(rp.perimeter) or 1.0
    cimg = getattr(rp, "image_convex", None)
    cimg = rp.convex_image if cimg is None else cimg
    cper = float(regionprops(cimg.astype(np.uint8))[0].perimeter) or per
    return dict(
        area=float(rp.area), perimeter=per, major=ax_maj, minor=ax_min,
        aspect=ax_maj / max(ax_min, 1e-6), eccentricity=float(rp.eccentricity),
        solidity=float(rp.solidity), extent=float(rp.extent), circularity=float(4 * np.pi * rp.area / per ** 2),
        roughness=float(per / cper),
        int_mean=float(inside.mean()), int_std=float(inside.std()), int_p10=float(np.percentile(inside, 10)),
        int_p90=float(np.percentile(inside, 90)), contrast=float((inside.mean() - bg_mu) / (bg_sd + 1e-6)),
        int_cv=float(inside.std() / (abs(inside.mean()) + 1e-6)),
        grad_mean=float(gm[m].mean()), grad_edge_ratio=float(gm[m].mean() / (gm[~m].mean() + 1e-6)) if (~m).any() else 0.0,
        lap_var=float(lap[m].var()),
    )


# ----------------------------------------------------------------------------- per track
def motion(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    out = dict(speed_mean=np.nan, speed_std=np.nan, speed_max=np.nan, straightness=np.nan, net_disp=np.nan,
               msd_alpha=np.nan, turn_mean=np.nan)
    if len(xs) < 3:
        return out
    d = np.hypot(np.diff(xs), np.diff(ys)) / FRAME_MIN
    path = d.sum()
    net = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    out.update(speed_mean=float(d.mean()), speed_std=float(d.std()), speed_max=float(d.max()), net_disp=net,
               straightness=float(net / path) if path > 0 else 0.0)
    lags = np.arange(1, min(11, len(xs) // 2))
    if len(lags) >= 3:
        msd = np.array([np.mean((xs[l:] - xs[:-l]) ** 2 + (ys[l:] - ys[:-l]) ** 2) for l in lags])
        ok = msd > 0
        if ok.sum() >= 3:
            out["msd_alpha"] = float(np.polyfit(np.log(lags[ok]), np.log(msd[ok]), 1)[0])
    ang = np.arctan2(np.diff(ys), np.diff(xs))
    if len(ang) > 1:
        da = np.angle(np.exp(1j * np.diff(ang)))
        out["turn_mean"] = float(np.mean(np.abs(da)))
    return out


def track_features(job):
    track, meta, rows, roi_res, mask_res, step, mode = job
    per = []
    for r in rows[::step]:
        q = r.get("mask_path")
        ip, mp = roi_res(r["imageFilePathROI"]), (mask_res(q) if isinstance(q, str) and q else None)
        if ip is None or mp is None or not ip.exists() or not mp.exists():
            continue
        try:
            img = _read(ip)
            f = frame_features(img, fit_mask(_read(mp), img.shape, mode))
        except Exception:
            f = None
        if f:
            per.append(f)
    out = dict(meta, track=track, n_masked=len(per))
    if len(per) >= 3:
        df = pd.DataFrame(per)
        for c in df.columns:
            v = df[c].to_numpy()
            out[f"{c}_mean"] = float(np.mean(v))
            out[f"{c}_std"] = float(np.std(v))
            out[f"{c}_dt"] = float(np.mean(np.abs(np.diff(v)))) if len(v) > 1 else 0.0   # frame-to-frame fluctuation
    xy = sorted((r["frame"], r["xs"], r["ys"]) for r in rows)
    out.update(motion([p[1] for p in xy], [p[2] for p in xy]))
    return out


MASK_MODE = "same"     # how a mask fits its crop; set by preflight(): same | resize | center


def fit_mask(m, shape, mode=None):
    """put a mask onto a crop of `shape`. The idr0109 masks can be smaller than the 256 x 256 crops: either the
    same field of view at another size ('resize') or a smaller box around the same centre ('center')."""
    mode = mode or MASK_MODE
    m = np.asarray(m) > 0
    if mode.endswith("-inv"):
        m = ~m
        mode = mode[:-4]
    if m.shape == tuple(shape):
        return m
    if mode == "resize":
        from skimage.transform import resize
        return resize(m.astype(np.float32), shape, order=0, preserve_range=True, anti_aliasing=False) > 0.5
    out = np.zeros(shape, bool)                             # center: paste / crop around the middle
    H, W = shape
    h, w = m.shape
    y0, x0 = (H - h) // 2, (W - w) // 2
    sy0, sx0 = max(0, -y0), max(0, -x0)
    dy0, dx0 = max(0, y0), max(0, x0)
    hh, ww = min(h - sy0, H - dy0), min(w - sx0, W - dx0)
    out[dy0:dy0 + hh, dx0:dx0 + ww] = m[sy0:sy0 + hh, sx0:sx0 + ww]
    return out


def edge_score(img, m):
    """how well a mask's outline sits on edges of the image (1 = no better than anywhere else)."""
    from scipy.ndimage import sobel
    from skimage.segmentation import find_boundaries
    g = np.hypot(sobel(img, 0), sobel(img, 1))
    b = find_boundaries(m, mode="inner")
    return float(g[b].mean() / (g.mean() + 1e-9)) if b.any() and g.mean() > 0 else 0.0


def preflight(roi, roi_res, mask_res, n=30):
    """read a few image/mask pairs before starting: stop at once with the real error instead of silently
    skipping every cell (e.g. zip-compressed TIFF masks need the imagecodecs package)."""
    ok = roi.dropna(subset=["mask_path"])
    sample = ok.groupby("track", sort=False).head(1).head(n) if "track" in ok else ok.head(n)
    bad = []
    for p, q in sample[["imageFilePathROI", "mask_path"]].itertuples(index=False):
        for f in (roi_res(p), mask_res(q)):
            try:
                a = _read(f)
                if a.size == 0:
                    raise ValueError("empty image")
            except Exception as e:
                bad.append(f"{f}: {type(e).__name__}: {e}")
    if bad:
        hint = (" -> install it: <venv>/bin/pip install imagecodecs" if any("imagecodecs" in b.lower() or "compress" in b.lower()
                                                                          for b in bad) else "")
        sys.exit("cannot read the image files:\n  " + "\n  ".join(bad[:4]) + hint)
    global MASK_MODE
    pairs = [(_read(roi_res(p)), _read(mask_res(q))) for p, q in sample[["imageFilePathROI", "mask_path"]].itertuples(index=False)]
    shapes = sorted({(a.shape, b.shape) for a, b in pairs})
    print("mask on-share", round(float(np.mean([(np.asarray(b) > 0).mean() for _, b in pairs])), 2), flush=True)
    modes = ("same", "same-inv") if all(a.shape == b.shape for a, b in pairs) else ("resize", "center", "resize-inv", "center-inv")
    sc = {md: float(np.mean([edge_score(a, fit_mask(b, a.shape, md)) for a, b in pairs])) for md in modes}
    MASK_MODE = max(sc, key=sc.get)
    print(f"preflight: crop/mask sizes {shapes[:4]}{' ...' if len(shapes) > 4 else ''}; mask outline on image edges: "
          + ", ".join(f"{k} {v:.2f}" for k, v in sc.items()) + f" -> using '{MASK_MODE}'", flush=True)
    if sc[MASK_MODE] < 1.2:
        sys.exit("neither way of fitting the masks onto the crops matches the image edges: send this output to check")
    return MASK_MODE


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roi-table", required=True)
    ap.add_argument("--mask-table", required=True)
    ap.add_argument("--roi-root", required=True)
    ap.add_argument("--mask-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--frame-step", type=int, default=1, help="use every n-th frame for the image features (speed)")
    ap.add_argument("--limit", type=int, default=0, help="only the first n tracks (quick test)")
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["cellType", "metEff", "date", "expStr", "locationStr", "frame", "ts", "xs", "ys", "imageFilePathROI"]
    roi = pd.read_csv(a.roi_table, sep="\t", usecols=cols, dtype={"date": str, "locationStr": str, "metEff": str})
    roi["track"] = roi["imageFilePathROI"].map(lambda p: os.path.basename(os.path.dirname(p)))
    print(f"ROI table: {len(roi):,} frames, {roi['track'].nunique():,} tracks", flush=True)

    mk = pd.read_csv(a.mask_table, sep="\t", usecols=["expStr", "frame", "imageFilePathMask"], dtype=str)
    mk["key"] = (mk["expStr"] + "|" + mk["imageFilePathMask"].map(lambda p: _tail_key(os.path.basename(os.path.dirname(p))))
                 + "|" + mk["frame"].astype(str))
    mdict = dict(zip(mk["key"], mk["imageFilePathMask"]))
    roi["mask_path"] = (roi["expStr"] + "|" + roi["track"].map(_tail_key) + "|" + roi["frame"].astype(str)).map(mdict)
    print(f"frames with a matching mask: {roi['mask_path'].notna().mean():.1%}", flush=True)
    del mk, mdict

    roi_res = Resolver(Path(a.roi_root), roi["imageFilePathROI"].head(50))
    mask_res = Resolver(Path(a.mask_root), roi["mask_path"].dropna().head(50))
    if roi_res.drop is None or mask_res.drop is None:
        sys.exit(f"could not find the image files under --roi-root ({roi_res.drop is not None}) / --mask-root "
                 f"({mask_res.drop is not None}); check that the archives are unpacked there.\n"
                 f"example table path: {roi['imageFilePathROI'].iloc[0]}")

    preflight(roi, roi_res, mask_res)
    jobs = []
    for tr, g in roi.groupby("track", sort=False):
        g = g.sort_values("frame")
        r0 = g.iloc[0]
        meta = dict(cellType=r0["cellType"], metEff=r0["metEff"], date=r0["date"], expStr=r0["expStr"],
                    locationStr=r0["locationStr"], n_frames=len(g), t_start=int(r0["ts"]))
        jobs.append((tr, meta, g[["frame", "xs", "ys", "imageFilePathROI", "mask_path"]].to_dict("records"),
                     roi_res, mask_res, a.frame_step, MASK_MODE))
        if a.limit and len(jobs) >= a.limit:
            break
    del roi
    print(f"{len(jobs):,} tracks -> {a.workers} workers", flush=True)

    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(track_features, jobs, chunksize=8), 1):
            rows.append(r)
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i:,}/{len(jobs):,} tracks  ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df = df[[c for c in META if c in df.columns] + [c for c in df.columns if c not in META]]
    df.to_csv(out / "tracks.csv.gz", index=False)
    log = dict(tracks=len(df), tracks_with_image_features=int((df["n_masked"] >= 3).sum()), mask_mode=MASK_MODE,
               by_cellType=df.groupby(["cellType", "metEff"]).size().rename("n").reset_index().to_dict("records"),
               frame_step=a.frame_step, seconds=round(time.time() - t0))
    (out / "build_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != "by_cellType"}), flush=True)


if __name__ == "__main__":
    main()