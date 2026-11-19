"""idr0111, one video: segment every frame, track the cells, measure them, and pool frozen Cellpose-SAM features.

Model: the general Cellpose-SAM ('cpsam'), unchanged (chosen in the segmentation check). The same network's encoder,
frozen, gives the image features (cellstate_build.Encoder: same normalisation and pooling as the cell-state model).

Per video, in <out>/<video stem>/:
  masks.npz       uint16 label image per frame (frames, H, W)       -> reused by later models (e.g. a graph model)
  cells.csv.gz    one row per cell per frame: shape, brightness, texture, neighbours, distance to the open space,
                  position, velocity, track id
  tracks.csv.gz   one row per tracked cell (>= --min-len frames): track features (mean / spread over its life,
                  movement, alignment with neighbours, divisions) + video labels (day, line, well)
  emb.npz         per track: mean / spread of the encoder features over sampled frames, how fast they change
  info.json       video, frames used, median cell diameter, counts, timings
Labels come from videos_annotation.csv (next to this script). Every video is cut to --frames frames (241 = 12 h),
so no video is longer than another.

  python idr111_process.py --index 0 --data data --out cells      # video 0 of the 198 (SLURM array: $SLURM_ARRAY_TASK_ID)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import segtest_idr0111 as ST  # noqa: E402  (read_video)

warnings.filterwarnings("ignore", category=FutureWarning)
PX_UM = 0.5826        # µm per pixel (from the OME metadata)
DT_MIN = 3.0          # minutes per frame


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def line_of(r):
    if str(r["Characteristics [Cell Line]"]).startswith("MDA"):
        return "MDA-MB-231"
    g = str(r["Experimental Condition [Genetic Perturbation]"])
    return {"none": "MCF10A", "PTEN-/-": "PTEN", "kRas activation": "KRas", "kRas/PTEN-/-": "KRas_PTEN"}.get(g, g)


def video_table(data):
    v = pd.read_csv(HERE / "videos_annotation.csv")
    v["line"] = v.apply(line_of, axis=1)
    v["series"] = np.where(v["Image File"].str.contains("10Amutants"), "mutants", "mda231")
    v["day"] = pd.to_datetime(v["Comment [Experimental Replicate Date]"], dayfirst=True).dt.strftime("%Y-%m-%d")
    v["well"] = v["Assay Name"].astype(str)
    v["path"] = [str(Path(data) / d / f) for d, f in zip(v["day"], v["Image File"])]
    return v.sort_values("Image File").reset_index(drop=True)


# ----------------------------------------------------------------------------- per frame
def neighbour_pairs(lab, gap=3):
    """pairs of cells that touch (after growing each cell by `gap` px into the background)"""
    from skimage.segmentation import expand_labels
    E = expand_labels(lab, gap)
    pr = []
    for a, b in ((E[:, :-1], E[:, 1:]), (E[:-1, :], E[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        if m.any():
            pr.append(np.stack([a[m], b[m]], 1))
    if not pr:
        return np.zeros((0, 2), np.int64)
    p = np.unique(np.sort(np.concatenate(pr), 1), axis=0)
    return p


def open_space_distance(lab, min_area=20000):
    """distance (px) from every pixel to the open, cell-free part of the dish (None if there is none in view)"""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import expand_labels
    empty = expand_labels(lab, 8) == 0
    empty = remove_small_objects(empty, min_area)
    if not empty.any():
        return None
    return distance_transform_edt(~empty)


def measure_frame(img, lab, t):
    from scipy import ndimage as ndi
    from skimage.measure import regionprops_table
    if lab.max() == 0:
        return pd.DataFrame(), np.zeros((0, 2), np.int64)
    x = img.astype(np.float64)
    x = x / max(np.median(x), 1e-9)                       # brightness relative to this frame (lamp drift)
    p = pd.DataFrame(regionprops_table(lab, properties=("label", "area", "perimeter", "eccentricity", "solidity",
                                                        "major_axis_length", "minor_axis_length", "orientation",
                                                        "centroid")))
    ids = p["label"].to_numpy()
    p["int_mean"] = ndi.mean(x, lab, ids)
    p["int_std"] = ndi.standard_deviation(x, lab, ids)
    lm = ndi.uniform_filter(x, 5)
    lsd = np.sqrt(np.clip(ndi.uniform_filter(x * x, 5) - lm * lm, 0, None))
    p["texture"] = ndi.mean(lsd, lab, ids)
    pr = neighbour_pairs(lab)
    nb = pd.Series(np.concatenate([pr[:, 0], pr[:, 1]])).value_counts() if len(pr) else pd.Series(dtype=int)
    p["n_neighbours"] = p["label"].map(nb).fillna(0).astype(int)
    D = open_space_distance(lab)
    cy, cx = p["centroid-0"].round().astype(int).clip(0, lab.shape[0] - 1), p["centroid-1"].round().astype(int).clip(0, lab.shape[1] - 1)
    p["dist_open_um"] = D[cy, cx] * PX_UM if D is not None else np.nan
    p["touches_border"] = ((p["centroid-0"] - p["major_axis_length"] / 2 < 1) | (p["centroid-1"] - p["major_axis_length"] / 2 < 1)
                           | (p["centroid-0"] + p["major_axis_length"] / 2 > lab.shape[0] - 2)
                           | (p["centroid-1"] + p["major_axis_length"] / 2 > lab.shape[1] - 2))
    p["frame"] = t
    p = p.rename(columns={"centroid-0": "y", "centroid-1": "x"})
    return p, pr


# ----------------------------------------------------------------------------- tracking
def link(a, b, min_iou=0.2):
    """one-to-one links between cells of consecutive frames by overlap (IoU >= min_iou, best first);
    also returns, for every cell of b, the cell of a it overlaps most (to find divisions)"""
    m = (a > 0) & (b > 0)
    if not m.any():
        return {}, {}
    pairs, n = np.unique(np.stack([a[m], b[m]]), axis=1, return_counts=True)
    ia, ca = np.unique(a[a > 0], return_counts=True); ib, cb = np.unique(b[b > 0], return_counts=True)
    A, B = dict(zip(ia, ca)), dict(zip(ib, cb))
    iou = n / (np.array([A[i] for i in pairs[0]]) + np.array([B[j] for j in pairs[1]]) - n)
    order = np.argsort(-iou)
    fwd, ua, ub = {}, set(), set()
    for k in order:
        if iou[k] < min_iou:
            break
        i, j = int(pairs[0, k]), int(pairs[1, k])
        if i in ua or j in ub:
            continue
        fwd[i] = j; ua.add(i); ub.add(j)
    best_parent = {}
    frac = n / np.array([B[j] for j in pairs[1]])
    for k in np.argsort(-frac):
        j = int(pairs[1, k])
        if j not in best_parent and frac[k] >= 0.3:
            best_parent[j] = int(pairs[0, k])
    return fwd, best_parent


def track(masks):
    """track id for every (frame, label); parent track for cells that appear by splitting off another cell"""
    T = len(masks)
    tid = [dict() for _ in range(T)]
    parent = {}
    nxt = 1
    for i in np.unique(masks[0]):
        if i:
            tid[0][int(i)] = nxt; nxt += 1
    for t in range(1, T):
        fwd, bp = link(masks[t - 1], masks[t])
        back = {j: i for i, j in fwd.items()}
        for j in np.unique(masks[t]):
            j = int(j)
            if not j:
                continue
            if j in back and back[j] in tid[t - 1]:
                tid[t][j] = tid[t - 1][back[j]]
            else:
                tid[t][j] = nxt
                if j in bp and bp[j] in tid[t - 1]:          # appeared on top of an existing cell: a division
                    parent[nxt] = tid[t - 1][bp[j]]
                nxt += 1
    return tid, parent


def close_gaps(cells, parent, max_gap=3, max_dist=20.0):
    """join a track that ends with a track that starts 1..max_gap frames later close to where it ended (the same
    cell, lost for a few frames or briefly cut in two by the segmentation). Greedy, nearest first; a track that
    started by splitting off a cell that is still there (a division) is not joined."""
    first = cells.sort_values("frame").groupby("track").head(1).set_index("track")
    last = cells.sort_values("frame").groupby("track").tail(1).set_index("track")
    T = int(cells["frame"].max())
    ends = last[last["frame"] < T]
    starts = first[first["frame"] > 0]
    cand = []
    for k, e in ends.iterrows():
        s = starts[(starts["frame"] > e["frame"]) & (starts["frame"] <= e["frame"] + max_gap)]
        if not len(s):
            continue
        d = np.hypot(s["x"] - e["x"], s["y"] - e["y"])
        for k2, dist in d[d <= max_dist].items():
            if k2 != k:
                cand.append((dist, k, k2))
    alive = set(cells["track"])
    merged_into, used_end, used_start = {}, set(), set()
    for dist, k, k2 in sorted(cand):
        if k in used_end or k2 in used_start:
            continue
        if k2 in parent and parent[k2] in alive and parent[k2] != k:
            continue                                   # a real division: the mother is still there
        used_end.add(k); used_start.add(k2); merged_into[k2] = k
    def root(k):
        while k in merged_into:
            k = merged_into[k]
        return k
    cells = cells.copy()
    cells["track"] = cells["track"].map(lambda k: root(k))
    for k2 in merged_into:
        parent.pop(k2, None)
    return cells, len(merged_into)


# ----------------------------------------------------------------------------- per track
SHAPE = ["area_um2", "perimeter_um", "eccentricity", "solidity", "aspect", "circularity", "int_mean", "int_std",
         "texture", "n_neighbours", "dist_open_um"]


def track_features(cells, parent, min_len):
    c = cells.sort_values(["track", "frame"]).copy()
    c["area_um2"] = c["area"] * PX_UM ** 2
    c["perimeter_um"] = c["perimeter"] * PX_UM
    c["aspect"] = c["major_axis_length"] / c["minor_axis_length"].clip(lower=1e-6)
    c["circularity"] = (4 * np.pi * c["area"] / c["perimeter"].clip(lower=1e-6) ** 2).clip(upper=1.5)
    g = c.groupby("track")
    c["df"] = g["frame"].diff()                                          # > 1 across a closed gap
    c["dx"] = g["x"].diff() * PX_UM / c["df"]; c["dy"] = g["y"].diff() * PX_UM / c["df"]   # per frame
    c["speed"] = np.hypot(c["dx"], c["dy"]) / DT_MIN                     # µm/min
    rows = []
    has_child = set(parent.values())
    for k, d in c.groupby("track"):
        if len(d) < min_len:
            continue
        r = dict(track=k, n_frames=len(d), first_frame=int(d["frame"].iloc[0]),
                 border_fraction=float(d["touches_border"].mean()),
                 born_by_division=int(k in parent), divides=int(k in has_child))
        for f in SHAPE:
            v = d[f].astype(float)
            r[f + "_mean"] = float(v.mean()); r[f + "_std"] = float(v.std())
            r[f + "_dt"] = float(np.nanmean(np.abs(np.diff(v.to_numpy())))) if len(v) > 1 else np.nan
        sp = d["speed"].dropna()
        path = float((sp * DT_MIN * d["df"].loc[sp.index]).sum())
        net = float(np.hypot((d["x"].iloc[-1] - d["x"].iloc[0]) * PX_UM, (d["y"].iloc[-1] - d["y"].iloc[0]) * PX_UM))
        dur = (d["frame"].iloc[-1] - d["frame"].iloc[0]) * DT_MIN
        st = d[["dx", "dy"]].dropna().to_numpy()
        mv = np.hypot(st[:, 0], st[:, 1]) > 0.5 * PX_UM
        u = st[mv] / np.hypot(st[mv, 0], st[mv, 1])[:, None] if mv.any() else np.zeros((0, 2))
        r.update(speed_mean=float(sp.mean()), speed_std=float(sp.std()), speed_p90=float(sp.quantile(0.9)),
                 straightness=net / path if path > 0 else np.nan, net_speed=net / dur if dur > 0 else np.nan,
                 persistence=float((u[1:] * u[:-1]).sum(1).mean()) if len(u) > 2 else np.nan,
                 alignment_mean=float(d["alignment"].mean()), alignment_std=float(d["alignment"].std()),
                 rel_speed_mean=float(d["rel_speed"].mean()))
        rows.append(r)
    return pd.DataFrame(rows)


def alignment(cells, pairs_by_frame):
    """per cell and frame: cos between its velocity (to the next frame) and its neighbours' mean velocity, and its
    speed minus the neighbours' mean speed (collective movement)"""
    c = cells.sort_values(["track", "frame"]).copy()
    g = c.groupby("track")
    nf = g["frame"].shift(-1) - c["frame"]
    c["vx"] = (g["x"].shift(-1) - c["x"]) / nf; c["vy"] = (g["y"].shift(-1) - c["y"]) / nf
    c["alignment"] = np.nan; c["rel_speed"] = np.nan
    for t, d in c.groupby("frame"):
        pr = pairs_by_frame.get(t)
        if pr is None or not len(pr):
            continue
        v = d.set_index("label")[["vx", "vy"]]
        both = pd.DataFrame(np.concatenate([pr, pr[:, ::-1]]), columns=["a", "b"])
        both = both[both["b"].isin(v.index) & both["a"].isin(v.index)]
        nb = v.loc[both["b"]].to_numpy()
        both["vx"], both["vy"] = nb[:, 0], nb[:, 1]
        both["sp"] = np.hypot(both["vx"], both["vy"])
        m = both.dropna().groupby("a")[["vx", "vy", "sp"]].mean()
        own = v.loc[m.index]
        dot = own["vx"] * m["vx"] + own["vy"] * m["vy"]
        nrm = np.hypot(own["vx"], own["vy"]) * np.hypot(m["vx"], m["vy"])
        al = (dot / nrm.where(nrm > 1e-6)).reindex(d["label"]).to_numpy()
        rs = ((np.hypot(own["vx"], own["vy"]) - m["sp"]) * PX_UM / DT_MIN).reindex(d["label"]).to_numpy()
        c.loc[d.index, "alignment"] = al
        c.loc[d.index, "rel_speed"] = rs
    return c


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="cells")
    ap.add_argument("--model", default="cpsam", help="segmentation model (general Cellpose-SAM by default)")
    ap.add_argument("--cellstate-dir", default=os.environ.get("CELLSTATE_DIR", ""))
    ap.add_argument("--frames", type=int, default=241, help="frames used per video (241 = 12 h at 3 min)")
    ap.add_argument("--min-len", type=int, default=20, help="shortest track kept (20 frames = 1 h)")
    ap.add_argument("--max-gap", type=int, default=3, help="join tracks across up to N missing frames")
    ap.add_argument("--emb-every", type=int, default=10, help="encoder features every N frames")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--debug", action="store_true", help="threshold 'segmentation' and a random encoder (tests only)")
    ap.add_argument("--force", action="store_true", help="redo a finished video")
    a = ap.parse_args()
    t0 = time.time()
    v = video_table(a.data)
    if not 0 <= a.index < len(v):
        sys.exit(f"--index must be 0..{len(v) - 1}")
    r = v.iloc[a.index]
    stem = Path(r["Image File"]).name.replace(".ome.tiff", "")
    out = Path(a.out) / stem
    if (out / "info.json").exists() and not a.force:
        log("already done:", stem); return
    out.mkdir(parents=True, exist_ok=True)
    log(f"video {a.index}: {stem} | {r['line']} | day {r['day']} | well {r['well']}")

    vid, meta = ST.read_video(r["path"])
    vid = vid[:a.frames]
    T = len(vid)
    if a.cellstate_dir:
        sys.path.insert(0, a.cellstate_dir)
    import cellstate_build as B
    seg = B.Segmenter(a.model, debug=a.debug, batch_size=a.batch)

    # 1. segmentation
    masks = np.zeros(vid.shape, np.uint16)
    for t in range(T):
        lab = seg.run(vid[t].astype(np.float32), 30) if a.debug else \
            np.asarray(seg.model.eval(vid[t].astype(np.float32), diameter=None, channel_axis=None, batch_size=a.batch)[0])
        masks[t] = lab.astype(np.uint16)
        if t % 40 == 0:
            log(f"  segmented {t + 1}/{T} ({int(masks[t].max())} cells)")
    t_seg = time.time() - t0
    np.savez_compressed(out / "masks.npz", masks=masks)

    # 2. measurements + tracking
    frames, pairs = [], {}
    for t in range(T):
        p, pr = measure_frame(vid[t], masks[t].astype(np.int32), t)
        frames.append(p); pairs[t] = pr
    cells = pd.concat(frames, ignore_index=True)
    tid, parent = track(masks)
    cells["track"] = [tid[f].get(int(l), -1) for f, l in zip(cells["frame"], cells["label"])]
    n_before = cells["track"].nunique()
    cells, n_closed = close_gaps(cells, parent, max_gap=a.max_gap, max_dist=0.5 * float(np.median(2 * np.sqrt(cells["area"] / np.pi))))
    log(f"  tracking: {n_before:,} tracks, {n_closed:,} joined across short gaps -> {cells['track'].nunique():,}")
    cells = alignment(cells, pairs)
    tracks = track_features(cells, parent, a.min_len)
    diam = float(np.median(2 * np.sqrt(cells["area"] / np.pi))) if len(cells) else 30.0
    log(f"  {len(cells):,} cell-frames, {cells['track'].nunique():,} tracks, {len(tracks):,} tracks >= {a.min_len} frames, "
        f"median diameter {diam:.1f} px")

    # 3. frozen encoder features, pooled under each cell, every --emb-every frames
    enc = B.Encoder(seg, stub=a.debug)
    keep = set(tracks["track"]) if len(tracks) else set()
    per = {}
    for t in range(0, T, a.emb_every):
        fm, rs = enc.feature_map(vid[t].astype(np.float32), diam)
        pooled = enc.pool(fm, rs, masks[t].astype(np.int32))
        for lbl, f in pooled.items():
            k = tid[t].get(int(lbl))
            if k in keep:
                per.setdefault(k, []).append(np.asarray(f, np.float32))
    ks = [k for k in tracks["track"] if len(per.get(k, [])) >= 2]
    if ks:
        E = [np.stack(per[k]) for k in ks]
        np.savez_compressed(out / "emb.npz", track=np.array(ks), emb_mean=np.stack([e.mean(0) for e in E]),
                            emb_std=np.stack([e.std(0) for e in E]),
                            emb_step=np.array([np.linalg.norm(np.diff(e, axis=0), axis=1).mean() for e in E], np.float32),
                            n=np.array([len(e) for e in E], np.int16))

    # 4. save
    lab = dict(video=stem, day=r["day"], line=r["line"], series=r["series"], well=r["well"])
    for k_, v_ in lab.items():
        tracks[k_] = v_
    keep_cols = ["frame", "label", "track", "y", "x", "area", "perimeter", "eccentricity", "solidity", "major_axis_length",
                 "minor_axis_length", "orientation", "int_mean", "int_std", "texture", "n_neighbours", "dist_open_um",
                 "touches_border", "vx", "vy", "alignment", "rel_speed"]
    cells[keep_cols].to_csv(out / "cells.csv.gz", index=False)
    tracks.to_csv(out / "tracks.csv.gz", index=False)
    info = dict(**lab, frames_used=T, frames_in_file=int(meta["shape"][0]), shape=meta["shape"], diameter_px=round(diam, 1),
                cell_frames=int(len(cells)), tracks=int(cells["track"].nunique()), tracks_kept=int(len(tracks)),
                tracks_with_features=len(ks), model=a.model, seconds_segmentation=round(t_seg),
                seconds_total=round(time.time() - t0))
    (out / "info.json").write_text(json.dumps(info, indent=1, default=str))
    log("done:", json.dumps(info, default=str))


if __name__ == "__main__":
    main()