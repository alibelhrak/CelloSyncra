"""idr0109: deep image features per tracked cell from the Cellpose-SAM encoder (no training on the labels).

For every track, up to --frames evenly spaced frames are passed through the encoder of the website's whole-cell
model (the same encoder and the same 'average the 256 features under the cell mask' as the cell-state model,
cellstate_build.Encoder). Per track we keep
  emb_mean  (256)  the average over the sampled frames  -> what the cell looks like
  emb_std   (256)  the spread over the frames            -> how much its look varies
  emb_step  (1)    mean distance between consecutive sampled frames -> how fast its look changes
The encoder is never trained here, so it cannot memorise which tumour a cell comes from; the PCA and the
classifier are fitted later inside each leave-one-tumour-out fold (idr_train.py).

  python idr_embed.py --roi-table ... --mask-table ... --roi-root ... --mask-root ... \
      --model <cpsam_ft_joint> --cellstate-dir <folder with cellstate_build.py> --out out/embed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import idr_build as IB  # noqa: E402  (table parsing, path resolver, image reading)

TILE = 256


def prep(img, mask, enc_scale):
    """crop -> Cellpose normalisation -> rescale so the cell is ~60 px wide -> padded 256 x 256 tile.
    returns the tile and the cell mask on the 1/8 feature grid (32 x 32)."""
    from cellpose import transforms
    from skimage.transform import resize
    x = transforms.normalize_img(transforms.convert_image(img, channel_axis=None, z_axis=None, do_3D=False))
    x = np.asarray(x, np.float32)[..., 0]
    m = mask > 0
    if enc_scale != 1:
        h, w = max(8, int(round(x.shape[0] * enc_scale))), max(8, int(round(x.shape[1] * enc_scale)))
        x = resize(x, (h, w), order=1, preserve_range=True, anti_aliasing=enc_scale < 1).astype(np.float32)
        m = resize(m.astype(np.uint8), (h, w), order=0, preserve_range=True, anti_aliasing=False) > 0
    h, w = min(x.shape[0], TILE), min(x.shape[1], TILE)
    y0, x0 = (x.shape[0] - h) // 2, (x.shape[1] - w) // 2           # centre crop if bigger than a tile
    tile = np.zeros((TILE, TILE), np.float32)
    mt = np.zeros((TILE, TILE), bool)
    oy, ox = (TILE - h) // 2, (TILE - w) // 2                         # centre in the tile
    tile[oy:oy + h, ox:ox + w] = x[y0:y0 + h, x0:x0 + w]
    mt[oy:oy + h, ox:ox + w] = m[y0:y0 + h, x0:x0 + w]
    g = TILE // 8
    small = mt.reshape(g, 8, g, 8).mean((1, 3)) >= 0.5
    if not small.any():
        yy, xx = np.nonzero(mt)
        if len(yy):
            small[int(yy.mean()) // 8, int(xx.mean()) // 8] = True
        else:
            small[g // 2, g // 2] = True
    return tile, small


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roi-table", required=True)
    ap.add_argument("--mask-table", required=True)
    ap.add_argument("--roi-root", required=True)
    ap.add_argument("--mask-root", required=True)
    ap.add_argument("--model", required=False, default=os.environ.get("CELL_MODEL"), help="whole-cell Cellpose-SAM model")
    ap.add_argument("--cellstate-dir", default=os.environ.get("CELLSTATE_DIR"), help="folder with cellstate_build.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=12, help="frames sampled per track")
    ap.add_argument("--cell-px", type=float, default=60.0, help="cell width in the encoder input (cell-state model: 2 x 30)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="tiny random encoder on CPU (tests only)")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cellstate_dir:
        sys.path.insert(0, a.cellstate_dir)
    try:
        import cellstate_build as B
    except ImportError:
        sys.exit("cellstate_build.py not found: pass --cellstate-dir (the CellState code folder) or set CELLSTATE_DIR")
    if not a.debug and (not a.model or not Path(a.model).exists()):
        sys.exit(f"whole-cell model not found: {a.model!r} (pass --model or set CELL_MODEL)")
    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- tracks and their frames (same parsing as idr_build)
    roi = pd.read_csv(a.roi_table, sep="\t", usecols=["expStr", "frame", "imageFilePathROI"], dtype={"expStr": str})
    roi["track"] = roi["imageFilePathROI"].map(lambda p: os.path.basename(os.path.dirname(p)))
    mk = pd.read_csv(a.mask_table, sep="\t", usecols=["expStr", "frame", "imageFilePathMask"], dtype=str)
    mk["key"] = (mk["expStr"] + "|" + mk["imageFilePathMask"].map(lambda p: IB._tail_key(os.path.basename(os.path.dirname(p))))
                 + "|" + mk["frame"].astype(str))
    mdict = dict(zip(mk["key"], mk["imageFilePathMask"]))
    roi["mask_path"] = (roi["expStr"] + "|" + roi["track"].map(IB._tail_key) + "|" + roi["frame"].astype(str)).map(mdict)
    del mk, mdict
    roi = roi.dropna(subset=["mask_path"])
    rres = IB.Resolver(Path(a.roi_root), roi["imageFilePathROI"].head(50))
    mres = IB.Resolver(Path(a.mask_root), roi["mask_path"].head(50))
    if rres.drop is None or mres.drop is None:
        sys.exit("image files not found under --roi-root / --mask-root")
    mode = IB.preflight(roi, rres, mres)
    jobs = []
    for tr, g in roi.groupby("track", sort=False):
        g = g.sort_values("frame")
        idx = np.unique(np.linspace(0, len(g) - 1, min(a.frames, len(g))).round().astype(int))
        jobs.append((tr, [(rres(p), mres(q)) for p, q in g.iloc[idx][["imageFilePathROI", "mask_path"]].itertuples(index=False)]))
        if a.limit and len(jobs) >= a.limit:
            break
    del roi
    print(f"{len(jobs):,} tracks, up to {a.frames} frames each", flush=True)

    # ---- encoder
    segm = B.Segmenter(a.model, debug=a.debug)
    enc = B.Encoder(segm, stub=a.debug)

    fails = {}

    def load(pair):
        ip, mp = pair
        try:
            img = IB._read(ip)
            m = IB.fit_mask(IB._read(mp), img.shape, mode)
            if (m > 0).sum() < 10:
                fails["mask smaller than 10 px"] = fails.get("mask smaller than 10 px", 0) + 1
                return None
            d = 2 * np.sqrt((m > 0).sum() / np.pi)                 # equivalent diameter of this cell
            return prep(img, m, a.cell_px / max(d, 1.0))
        except Exception as e:
            k = f"{type(e).__name__}: {str(e)[:120]}"
            fails[k] = fails.get(k, 0) + 1
            return None

    names, means, stds, steps, nused = [], [], [], [], []
    pool = ThreadPoolExecutor(8)
    buf_t, buf_m, owner = [], [], []
    per_track = {}

    def flush():
        if not buf_t:
            return
        x = torch.from_numpy(np.stack(buf_t)[:, None]).to(enc.dev, enc.dtype)
        with torch.no_grad():
            f = enc._forward(x).float().cpu().numpy()             # (B, 256, 32, 32)
        for fi, mi, (k, j) in zip(f, buf_m, owner):
            per_track.setdefault(k, {})[j] = fi[:, mi].mean(1)
        buf_t.clear(); buf_m.clear(); owner.clear()

    def finish(k):
        tr = jobs[k][0]
        e = per_track.pop(k, {})
        if len(e) < 2:
            return
        E = np.stack([e[j] for j in sorted(e)])
        names.append(tr)
        means.append(E.mean(0))
        stds.append(E.std(0))
        steps.append(float(np.linalg.norm(np.diff(E, axis=0), axis=1).mean()))
        nused.append(len(E))

    for k, (tr, pairs) in enumerate(jobs):
        for j, r in enumerate(pool.map(load, pairs)):
            if r is None:
                continue
            buf_t.append(r[0]); buf_m.append(r[1]); owner.append((k, j))
            if len(buf_t) >= a.batch:
                flush()
        if len(buf_t) >= a.batch or k == len(jobs) - 1:
            flush()
        pending = {kk for kk, _ in owner}                           # tracks with frames still waiting in the batch
        done = [kk for kk in per_track if kk <= k and kk not in pending]
        for kk in done:
            finish(kk)
        if (k + 1) % 500 == 0:
            print(f"  {k + 1:,}/{len(jobs):,} tracks ({time.time() - t0:.0f} s)", flush=True)
    flush()
    for kk in list(per_track):
        finish(kk)

    if fails:
        print("frames skipped: " + "; ".join(f"{v:,} x {k}" for k, v in sorted(fails.items(), key=lambda x: -x[1])[:5]), flush=True)
    if not names:
        sys.exit("no track could be encoded (see 'frames skipped' above)")
    np.savez_compressed(out / "embeddings.npz", track=np.array(names), emb_mean=np.stack(means).astype(np.float32),
                        emb_std=np.stack(stds).astype(np.float32), emb_step=np.array(steps, np.float32),
                        n_frames=np.array(nused, np.int16))
    log = dict(tracks=len(names), of=len(jobs), frames_per_track=a.frames, cell_px=a.cell_px, mask_mode=mode,
               model=str(a.model), seconds=round(time.time() - t0))
    (out / "embed_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log), flush=True)


if __name__ == "__main__":
    main()