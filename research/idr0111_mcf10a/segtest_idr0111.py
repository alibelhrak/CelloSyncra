"""idr0111: can Cellpose-SAM separate the single cells in these dense sheets? Test on 3 videos before building anything.

For each test video: reads the OME-TIFF (prints its shape and pixel size), takes 3 frames (start, middle, end),
segments them with the general Cellpose-SAM model ('cpsam') and with our fine-tuned model, and saves
  segtest/<video>_<model>.png   crop of the frame | same crop with the cell outlines (crop = centre 512 x 512 px)
  segtest/summary.json          cells per frame, median cell diameter (px), share of the image covered by cells
  python segtest_idr0111.py --data data --out segtest --ft-model <cpsam_ft_joint>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

VIDEOS = {  # same recording day for the two MCF10A videos; one MDA-MB-231 video
    "MCF10A_normal": "2019-08-07/2019-08-07_10Amutants_Migration2019-08-07_10Amutants_3minFrames_XY0001.ome.tiff",
    "MCF10A_KRas_PTEN": "2019-08-07/2019-08-07_10Amutants_Migration2019-08-07_10Amutants_3minFrames_XY0007.ome.tiff",
    "MDA-MB-231": "2019-12-10/2019-12-10_10A_231_Migration2019-12-10_10A_231_3minFrames_XY0007.ome.tiff",
}


def read_video(p):
    import tifffile
    with tifffile.TiffFile(p) as tf:
        s = tf.series[0]
        arr = s.asarray()
        axes = s.axes
        px = None
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(tf.ome_metadata)
            pix = root.find(".//{*}Pixels")
            px = (pix.get("PhysicalSizeX"), pix.get("PhysicalSizeXUnit"), pix.get("TimeIncrement"), pix.get("TimeIncrementUnit"))
        except Exception:
            pass
    info = dict(shape=list(arr.shape), axes=axes, dtype=str(arr.dtype), pixel_size_and_time=px)
    # -> (T, Y, X), first channel / plane
    a = arr
    order = list(axes)
    for ax in ("C", "Z", "S"):
        if ax in order:
            a = np.take(a, 0, axis=order.index(ax)); order.remove(ax)
    if "T" not in order:
        a = a[None]; order = ["T"] + order
    a = np.moveaxis(a, order.index("T"), 0)
    return a, info


def overlay(img, lab, crop=512):
    from skimage.segmentation import find_boundaries
    H, W = img.shape
    y0, x0 = max(0, (H - crop) // 2), max(0, (W - crop) // 2)
    im = img[y0:y0 + crop, x0:x0 + crop].astype(np.float64)
    lb = lab[y0:y0 + crop, x0:x0 + crop]
    lo, hi = np.percentile(im, [1, 99])
    v = np.clip((im - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = np.dstack([v] * 3)
    rgb[find_boundaries(lb, mode="inner")] = (1.0, 0.85, 0.0)
    left = np.dstack([v] * 3)
    gap = np.ones((left.shape[0], 6, 3))
    return (np.hstack([left, gap, rgb]) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="segtest")
    ap.add_argument("--ft-model", default="", help="our fine-tuned whole-cell model (optional)")
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    from cellpose import models
    from PIL import Image
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mods = {"general": models.CellposeModel(gpu=bool(a.gpu), pretrained_model="cpsam")}
    if a.ft_model and Path(a.ft_model).exists():
        mods["finetuned"] = models.CellposeModel(gpu=bool(a.gpu), pretrained_model=a.ft_model)
    summ = {}
    for name, rel in VIDEOS.items():
        p = Path(a.data) / rel
        if not p.exists():
            print("missing:", p); continue
        vid, info = read_video(p)
        T = vid.shape[0]
        frames = sorted({0, T // 2, T - 1})
        print(f"{name}: {info}", flush=True)
        summ[name] = dict(file=rel, **info, frames=frames, models={})
        for mn, m in mods.items():
            rows, pics = [], []
            for t in frames:
                t0 = time.time()
                lab = m.eval(vid[t].astype(np.float32), diameter=None, channel_axis=None)[0]
                ids, areas = np.unique(lab[lab > 0], return_counts=True)
                d = 2 * np.sqrt(areas / np.pi) if len(areas) else np.array([0.0])
                rows.append(dict(frame=int(t), cells=int(len(ids)), median_diameter_px=round(float(np.median(d)), 1),
                                 covered_percent=round(100 * float((lab > 0).mean()), 1), seconds=round(time.time() - t0, 1)))
                pics.append(overlay(vid[t], lab))
                print(f"  {mn} frame {t}: {rows[-1]}", flush=True)
            Image.fromarray(np.vstack([np.vstack([p_, np.full((6, p_.shape[1], 3), 255, np.uint8)]) for p_ in pics])).save(out / f"{name}_{mn}.png")
            summ[name]["models"][mn] = rows
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print("pictures and summary in", out.resolve())


if __name__ == "__main__":
    main()
