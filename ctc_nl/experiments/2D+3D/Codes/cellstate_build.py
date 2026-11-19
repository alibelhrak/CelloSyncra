#!/usr/bin/env python
"""CelloSyncra cell-state classifier - step 1: build ONE dataset of tracked, labelled cells from several sources.

Every video goes through the SAME pipeline the website uses (whole-cell Cellpose-SAM -> IoU tracker -> division
clean-up), and per cell per frame we store:
    * its Cellpose-SAM ENCODER features (256 numbers averaged under the cell's outline)
    * its measurements (area, perimeter, shape, brightness - raw and image-normalised -, texture, position)
    * its track id (same cell over time, with lineage)
    * the expert label, when the source labelled that cell in that frame (matched to the tracked cell)

Sources (label-free only):
    alfi   ALFI (DIC; U2OS / HeLa / RPE-1)    interphase, mitosis, multipolar division, death (type unknown)
    qpi    Vicar et al. (quantitative phase; DU-145 / LNCaP / PNT1A; st, do, bp)
           apoptosis, necrosis (from the death frame on), alive (well before the death frame)
    ctc    Cell Tracking Challenge DIC-C2DH-HeLa, PhC-C2DH-U373: mitosis from the lineage (man_track.txt),
           weak interphase far from divisions

    inspect   ALFI label counts (no GPU)
    build     --source alfi|qpi|ctc  ->  out/seqs/<VIDEO>/{objects.csv, emb.npy, labels.csv, res_track.txt,
              masks.npz, info.json}  and out/build_report.csv          (GPU)

labels.csv 'state' values and what training does with them:
    interphase, mitosis, abnormal_division   alive + that alive-state
    alive                                    alive, alive-state unknown (QPI before death)
    weak_interphase                          probably interphase, low weight (never-labelled cells)
    death                                    dead, type unknown (ALFI)
    apoptosis, necrosis                      dead + that type (QPI)

Writes only inside --out. Reads the raw data and the whole-cell model.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CLASS_MAP = {                      # ALFI class -> CelloSyncra cell state (+ fine label kept separately)
    "interphase": "interphase",
    "mitosis": "mitosis", "earlymitosis": "mitosis", "latemitosis": "mitosis",
    "multipolar": "abnormal_division",
    "celldeath": "death",
}
SIZE_CANDIDATES = (15, 30, 60, 120, 240)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------- raw ALFI files
_NUM = re.compile(r"(\d+)(?!.*\d)")


def index_images(d: Path):
    out = {}
    for p in sorted(list(d.glob("*.png")) + list(d.glob("*.tif"))):
        m = _NUM.search(p.stem)
        if m:
            out[int(m.group(1))] = p
    return out


def read_img(p: Path):
    from PIL import Image
    import tifffile
    a = tifffile.imread(p) if p.suffix.lower() in (".tif", ".tiff") else np.array(Image.open(p))
    if a.ndim == 3:
        a = a[..., :3].mean(-1)
    return a.astype(np.float32)


UNKNOWN_CLASSES: dict = {}           # class names found in the CSVs but not in CLASS_MAP (reported by inspect)


def load_labels(seq_dir: Path, seq: str):
    """Reads ALL label files of a video (MI videos have two: DTLTruth = Interphase/Mitosis for every cell,
    PhenoTruth = Early/Late mitosis). The same cell in the same frame labelled in both files with the same state
    (box centres < 10 px apart) is kept once, with the finer name (EarlyMitosis rather than Mitosis).
    rows: dict(frame, gt_id, cls_raw, state, box=(x0,y0,x1,y1), parent, source)"""
    import pandas as pd
    files = [seq_dir / f"{seq}_{k}Truth.csv" for k in ("Pheno", "DTL")]
    files = [f for f in files if f.exists()]
    rows = []
    for f in files:
        df = pd.read_csv(f, dtype=str)
        df.columns = df.columns.str.strip()
        src = "pheno" if "Pheno" in f.name else "dtl"
        for _, r in df.iterrows():
            raw = str(r["Class"]).strip().lower().replace(" ", "").replace("_", "")
            if raw not in CLASS_MAP:
                key = (seq, str(r["Class"]).strip())
                UNKNOWN_CLASSES[key] = UNKNOWN_CLASSES.get(key, 0) + 1
                continue
            x0, y0 = float(r["xmin"]), float(r["ymin"])
            rows.append(dict(frame=int(float(r["ImNo"])), gt_id=f"{src}:{str(r['ID']).strip()}", cls_raw=raw,
                             state=CLASS_MAP[raw], box=(x0, y0, x0 + float(r["width"]), y0 + float(r["height"])),
                             parent=str(r.get("Parent", "") or "").strip(), source=src))
    keep, seen = [], {}
    for r in rows:                       # pheno rows come first -> the finer name wins
        cx, cy = (r["box"][0] + r["box"][2]) / 2, (r["box"][1] + r["box"][3]) / 2
        dup = any(abs(cx - x) < 10 and abs(cy - y) < 10 for x, y in seen.get((r["frame"], r["state"]), []))
        if not dup:
            keep.append(r)
            seen.setdefault((r["frame"], r["state"]), []).append((cx, cy))
    return keep, "+".join(f.name for f in files) or None


def sequences(raw: Path, pattern):
    return sorted(d.name for d in raw.iterdir() if d.is_dir() and re.fullmatch(pattern, d.name))


# ----------------------------------------------------------------------------- inspect
def cmd_inspect(a):
    import pandas as pd
    raw = Path(a.raw)
    recs = []
    for s in sequences(raw, a.seqs):
        rows, f = load_labels(raw / s, s)
        imgs = index_images(raw / s / "Images")
        cells = {}
        for r in rows:
            cells.setdefault((r["state"], r["gt_id"]), 0)
            cells[(r["state"], r["gt_id"])] += 1
        for st in sorted({k[0] for k in cells}):
            ids = [k for k in cells if k[0] == st]
            recs.append(dict(seq=s, family=s[:2], state=st, labelled_cells=len(ids),
                             labelled_frames=sum(cells[k] for k in ids), frames_in_video=len(imgs), csv=f))
    df = pd.DataFrame(recs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "label_inventory.csv", index=False)
    piv = df.pivot_table(index="state", columns="family", values="labelled_cells", aggfunc="sum", fill_value=0)
    log("labelled CELLS per state and sequence family:\n" + piv.to_string())
    piv2 = df.pivot_table(index="state", columns="family", values="labelled_frames", aggfunc="sum", fill_value=0)
    log("labelled (cell, frame) pairs:\n" + piv2.to_string())
    log(f"per sequence: {out / 'label_inventory.csv'}")
    if UNKNOWN_CLASSES:
        log("WARNING class names NOT used (not in CLASS_MAP): " +
            ", ".join(f"{s}:{c} x{n}" for (s, c), n in sorted(UNKNOWN_CLASSES.items())))
    else:
        log("all class names recognised")


# ----------------------------------------------------------------------------- models
class Segmenter:
    """The website's whole-cell model (Cellpose-SAM fine-tuned) + automatic cell size."""

    def __init__(self, path, debug=False, batch_size=8):
        self.debug, self.bs = debug, batch_size
        if not debug:
            import torch
            if not torch.cuda.is_available() and os.environ.get("CS_ALLOW_CPU") != "1":
                raise SystemExit("ERROR: no usable GPU (torch.cuda.is_available() is False; see the .err file - e.g. "
                                 "'NVIDIA driver too old' on the V100 nodes). Stopping instead of running ~50x slower on CPU. "
                                 "Set CS_ALLOW_CPU=1 to force CPU.")
            from cellpose import models
            bf16 = torch.cuda.is_bf16_supported()      # H100: yes; V100 / RTX 8000: no -> full precision
            log(f"GPU {torch.cuda.get_device_name(0)}, torch {torch.__version__} (CUDA {torch.version.cuda}), "
                f"{'bfloat16' if bf16 else 'float32'}")
            self.model = models.CellposeModel(gpu=True, pretrained_model=path, use_bfloat16=bf16)

    def run(self, img, d):
        if self.debug:
            from skimage import filters, measure, morphology
            b = img > filters.threshold_otsu(img)
            return measure.label(morphology.remove_small_objects(b, 20)).astype(np.int32)
        kw = {} if d == 30 else {"diameter": d}
        return np.asarray(self.model.eval(img, batch_size=self.bs, **kw)[0]).astype(np.int32)

    def choose_size(self, imgs):
        best = None
        for d in SIZE_CANDIDATES:
            meds = []
            for im in imgs:
                lab = self.run(im, d)
                _, ar = np.unique(lab[lab > 0], return_counts=True)
                ar = ar[ar >= 9]
                if len(ar):
                    meds.append(float(np.median(2 * np.sqrt(ar / np.pi))))
            score = abs(np.log(np.median(meds) / d)) if meds else np.inf
            if best is None or score < best[1]:
                best = (d, score)
        return best[0]


class Encoder:
    """Cellpose-SAM image encoder -> 256-d feature map at 1/8 resolution (the 'neck' output), on 256 px tiles
    with 50% overlap, after the same normalisation and rescaling as segmentation."""

    def __init__(self, segmenter: Segmenter, stub=False, tile=256):
        import torch
        self.torch, self.tile, self.stub = torch, tile, stub
        if stub:                                  # tests only: a tiny random conv net with the same output shape
            self.net = torch.nn.Sequential(torch.nn.Conv2d(1, 256, 8, stride=8))
            self.dev, self.dtype = "cpu", torch.float32
        else:
            self.net = segmenter.model.net
            self.dev = next(self.net.parameters()).device
            self.dtype = next(self.net.parameters()).dtype

    def _forward(self, x):
        F = self.torch.nn.functional
        if self.stub:
            return self.net(x)
        enc, ps = self.net.encoder, self.net.ps
        h = F.conv2d(x, enc.patch_embed.proj.weight.data[:, :x.shape[1]], bias=enc.patch_embed.proj.bias.data, stride=ps)
        h = h.permute(0, 2, 3, 1)
        if enc.pos_embed is not None:
            h = h + enc.pos_embed
        for blk in enc.blocks:
            h = blk(h)
        return enc.neck(h.permute(0, 3, 1, 2))   # (B, 256, tile/8, tile/8)

    def feature_map(self, img, diameter):
        from cellpose import transforms
        torch = self.torch
        x = transforms.convert_image(img, channel_axis=None, z_axis=None, do_3D=False)
        x = transforms.normalize_img(x)
        rs = 30.0 / diameter
        if rs != 1:
            x = transforms.resize_image(x, rsz=rs)
        x = np.asarray(x, np.float32)[..., :1]   # one channel (grayscale)
        H, W = x.shape[:2]
        t, st = self.tile, self.tile // 2
        Hp, Wp = max(t, int(np.ceil((H - t) / st)) * st + t), max(t, int(np.ceil((W - t) / st)) * st + t)
        pad = np.zeros((Hp, Wp, 1), np.float32)
        pad[:H, :W] = x
        g = t // 8
        acc = np.zeros((256, Hp // 8, Wp // 8), np.float32)
        cnt = np.zeros((Hp // 8, Wp // 8), np.float32)
        pos = [(y, xx) for y in range(0, Hp - t + 1, st) for xx in range(0, Wp - t + 1, st)]
        with torch.no_grad():
            for i in range(0, len(pos), 8):
                chunk = pos[i:i + 8]
                b = np.stack([pad[y:y + t, xx:xx + t, 0] for y, xx in chunk])[:, None]
                f = self._forward(torch.from_numpy(b).to(self.dev, self.dtype)).float().cpu().numpy()
                for (y, xx), fi in zip(chunk, f):
                    acc[:, y // 8:y // 8 + g, xx // 8:xx // 8 + g] += fi
                    cnt[y // 8:y // 8 + g, xx // 8:xx // 8 + g] += 1
        fm = acc / np.maximum(cnt, 1)
        return fm[:, :int(np.ceil(H / 8)), :int(np.ceil(W / 8))], rs

    def pool(self, fm, rs, lab):
        """mean feature vector under each cell (cell mask brought to the feature-map grid)."""
        from skimage.transform import resize
        h, w = fm.shape[1:]
        small = resize(lab, (h, w), order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)
        out = {}
        for i in np.unique(lab):
            if i == 0:
                continue
            m = small == i
            if not m.any():                       # tiny cell: the grid cell under its centre
                yy, xx = np.nonzero(lab == i)
                m = np.zeros((h, w), bool)
                m[min(h - 1, int(yy.mean() * h / lab.shape[0])), min(w - 1, int(xx.mean() * w / lab.shape[1]))] = True
            out[int(i)] = fm[:, m].mean(1)
        return out

# ----------------------------------------------------------------------------- per-frame measurements
def measurements(lab, img):
    from skimage.measure import regionprops
    rows = {}
    for r in regionprops(lab, intensity_image=img):
        per = max(r.perimeter, 1e-6)
        vals = img[tuple(r.coords.T)]
        rows[int(r.label)] = dict(
            area=float(r.area), perimeter=float(per), eccentricity=float(r.eccentricity), solidity=float(r.solidity),
            circularity=float(min(1.0, 4 * np.pi * r.area / per ** 2)),
            aspect_ratio=float(r.axis_major_length / max(r.axis_minor_length, 1e-6)),
            mean_intensity=float(vals.mean()), texture=float(vals.std() / (abs(vals.mean()) + 1e-6)),
            cy=float(r.centroid[0]), cx=float(r.centroid[1]))
    return rows


def match_labels(rows, lab, t):
    """each labelled box at frame t -> the tracked cell with most pixels inside the box (needs >= 25% of the
    cell inside the box, or the box centre on the cell)."""
    out = []
    H, W = lab.shape
    for r in rows:
        x0, y0, x1, y1 = r["box"]
        Ya, Yb, Xa, Xb = int(max(0, y0)), int(min(H, y1 + 1)), int(max(0, x0)), int(min(W, x1 + 1))
        sub = lab[Ya:Yb, Xa:Xb]
        ids, cnt = np.unique(sub[sub > 0], return_counts=True)
        tid, q = 0, 0.0
        if len(ids):
            k = int(np.argmax(cnt))
            whole = int((lab == ids[k]).sum())
            q = cnt[k] / max(whole, 1)
            cy, cx = int(min(H - 1, (y0 + y1) / 2)), int(min(W - 1, (x0 + x1) / 2))
            if q >= 0.25 or lab[cy, cx] == ids[k]:
                tid = int(ids[k])
        out.append(dict(frame=t, gt_id=r["gt_id"], source=r["source"], cls_raw=r["cls_raw"], state=r["state"], track_id=tid,
                        cell_inside_box=round(float(q), 3)))
    return out


# ----------------------------------------------------------------------------- division clean-up
# Same rule as the website (cs_tracks.py next to cellosyncra_web.py): keep a division only if every daughter
# lives >= min_life frames; otherwise undo it (the longest piece continues the parent). Keep both copies in sync.
def clean_divisions(tracks: dict, last_frame: int, min_life: int = 3):
    """Returns (new_tracks, relabel, report).
    relabel: {old_id: new_id} to apply to the masks of frames [old B, old E] (daughter -> parent id)."""
    tr = {int(L): dict(B=int(d["B"]), E=int(d["E"]), P=int(d.get("P") or 0)) for L, d in tracks.items()}
    relabel: dict[int, int] = {}
    undone = 0

    def kids_of(j):
        return sorted((L for L, d in tr.items() if d["P"] == j), key=lambda L: tr[L]["B"])

    def life(L):
        return tr[L]["E"] - tr[L]["B"] + 1

    changed = True
    while changed:
        changed = False
        parents = sorted({d["P"] for d in tr.values() if d["P"] and d["P"] in tr},
                         key=lambda j: tr[j]["E"])
        for j in parents:
            kids = kids_of(j)
            if not kids:
                continue
            ok = len(kids) >= 2 and all(life(k) >= min_life or tr[k]["E"] >= last_frame for k in kids)
            if ok:
                continue
            # undo: the longest-living piece continues the parent
            cont = max(kids, key=lambda k: (life(k), -k))
            for k in kids:
                if k != cont:
                    tr[k]["P"] = 0
            tr[j]["E"] = tr[cont]["E"]
            for g in kids_of(cont):              # the continuation's own daughters now belong to j
                tr[g]["P"] = j
            relabel[cont] = j
            del tr[cont]
            undone += 1
            changed = True
            break                                # the lineage changed: start again in time order
    # follow chains (a -> b -> c becomes a -> c)
    for k in list(relabel):
        v = relabel[k]
        while v in relabel:
            v = relabel[v]
        relabel[k] = v
    kept = len({d["P"] for d in tr.values() if d["P"] and d["P"] in tr})
    return tr, relabel, dict(divisions_kept=kept, divisions_undone=undone, min_life=min_life)


def res_track_text(tracks: dict) -> str:
    return "".join(f"{L} {d['B']} {d['E']} {d['P']}\n" for L, d in sorted(tracks.items()))


# ----------------------------------------------------------------------------- measurements (v2)
def measurements(lab, img):
    """per cell: shape, size, brightness (raw AND relative to the frame's 1st-99th percentile, so DIC, phase and
    QPI become comparable without scaling against the video's own cells), texture (std/mean, scale-free)."""
    from skimage.measure import regionprops
    lo, hi = np.percentile(img, (1, 99))
    rows = {}
    for r in regionprops(lab, intensity_image=img):
        per = max(r.perimeter, 1e-6)
        vals = img[tuple(r.coords.T)]
        m = float(vals.mean())
        rows[int(r.label)] = dict(
            area=float(r.area), perimeter=float(per), eccentricity=float(r.eccentricity), solidity=float(r.solidity),
            circularity=float(min(1.0, 4 * np.pi * r.area / per ** 2)),
            aspect_ratio=float(r.axis_major_length / max(r.axis_minor_length, 1e-6)),
            mean_intensity=m, intensity_img=float((m - lo) / max(hi - lo, 1e-6)),
            texture=float(vals.std() / (abs(m) + 1e-6)), texture_img=float(vals.std() / max(hi - lo, 1e-6)),
            cy=float(r.centroid[0]), cx=float(r.centroid[1]))
    return rows


def best_overlap(gt, lab, min_iou=0.25, min_cover=0.5):
    """predicted cell for one ground-truth region (bool mask): the one with the largest overlap, accepted if
    IoU >= min_iou or it covers >= min_cover of the ground truth. Returns (track_id or 0, iou)."""
    ids, cnt = np.unique(lab[gt], return_counts=True)
    keep = ids > 0
    ids, cnt = ids[keep], cnt[keep]
    if not len(ids):
        return 0, 0.0
    k = int(np.argmax(cnt))
    inter = cnt[k]
    iou = inter / (gt.sum() + (lab == ids[k]).sum() - inter)
    return (int(ids[k]) if iou >= min_iou or inter / gt.sum() >= min_cover else 0), float(iou)


# ----------------------------------------------------------------------------- sources
ALFI_META = {**{f"MI{i:02d}": ("U2OS", "DMSO/MLN8237", 7) for i in range(1, 6)},
             "MI06": ("HeLa", "DMSO/MLN8237", 7), "MI07": ("RPE-1", "untreated", 7), "MI08": ("RPE-1", "untreated", 7),
             **{f"CD{i:02d}": ("U2OS", "DMSO/MLN8237", 5) for i in range(1, 10)},
             **{f"TP{i:02d}": ("U2OS", "DMSO/MLN8237", 5) for i in range(1, 11)},
             "TP11": ("RPE-1", "Taxol 500nM", 7), "TP12": ("HeLa", "DMSO/MLN8237", 7)}
QPI_TREAT = {"st": "staurosporine", "do": "doxorubicin", "bp": "black phosphorus"}
CTC_META = {"DIC-C2DH-HeLa": ("HeLa", "dic", 10), "PhC-C2DH-U373": ("U373", "phase", 15)}
CANCER = {"U2OS": True, "HeLa": True, "RPE-1": False, "DU145": True, "LNCaP": True, "PNT1A": False, "U373": True}


class AlfiVideo:
    dataset, modality = "alfi", "dic"

    def __init__(self, raw: Path, s: str):
        self.raw, self.s, self.name = raw, s, f"ALFI_{s}"
        self.cell_line, self.treatment, self.minutes = ALFI_META.get(s, ("unknown", "unknown", 7))
        self.imgs = index_images(raw / s / "Images")
        self.frames = sorted(self.imgs)
        rows, _ = load_labels(raw / s, s)
        self.by_t = {}
        for r in rows:
            self.by_t.setdefault(r["frame"], []).append(r)

    def read(self, t):
        return read_img(self.imgs[t])

    def labels(self, t, lab):
        return [dict(l, gt_id=f"{self.name}:{l['gt_id']}") for l in match_labels(self.by_t.get(t, []), lab, t)]

    def boxes(self, t):
        return [(r["box"], r["state"]) for r in self.by_t.get(t, [])]


class QpiVideo:
    """Vicar et al.: QPI_<LINE>_<TR>_<k>.tif (frames x 600 x 600, float pg/um2), mask_<...>.tif (same shape,
    value = cell_num of the annotated cells), labels_<...>.xlsx (cell_num, death_type 1 apoptosis / 2 necrosis /
    3 ambiguous-or-survived, death_frame). 3 min per frame; we use every `step`-th frame.
    Labels: apoptosis/necrosis from the death frame for `dead_min` minutes; 'alive' up to `margin_min` minutes
    before the death frame; 'transition' in between (not used for training, only for the time-of-death analysis);
    type 3 cells are not used. Every row carries the expert death_frame."""
    dataset, modality = "qpi", "qpi"

    def __init__(self, raw: Path, stem: str, step=2, dead_min=180, margin_min=60):
        import pandas as pd
        import tifffile
        line, tr, k = stem.split("_")[1:4]
        d = raw / line
        self.name, self.cell_line, self.treatment = stem, line, QPI_TREAT.get(tr, tr)
        self.minutes = 3.0 * step
        self.tf_img = tifffile.TiffFile(d / f"{stem}.tif")
        self.tf_msk = tifffile.TiffFile(d / f"{stem.replace('QPI_', 'mask_')}.tif")
        n = len(self.tf_img.pages)
        self.frames = list(range(0, n, step))
        lab = pd.read_excel(d / f"{stem.replace('QPI_', 'labels_')}.xlsx")
        lab = lab[lab.death_type.isin([1, 2])]
        self.cells = {int(r.cell_num): ("apoptosis" if int(r.death_type) == 1 else "necrosis", int(r.death_frame))
                      for r in lab.itertuples()}
        self.dead_f, self.margin_f = dead_min / 3.0, margin_min / 3.0
        self.found = set()

    def read(self, t):
        return self.tf_img.pages[t].asarray().astype(np.float32)

    def labels(self, t, lab):
        gt = self.tf_msk.pages[t].asarray().astype(np.int32)
        out = []
        for c, (typ, df) in self.cells.items():
            if t >= df:
                if t > df + self.dead_f:
                    continue
                state = typ
            elif t < df - self.margin_f:
                state = "alive"
            else:
                state = "transition"
            g = gt == c
            if not g.any():
                continue
            self.found.add(c)
            tid, iou = best_overlap(g, lab)
            out.append(dict(frame=t, gt_id=f"{self.name}:{c}", source="qpi", cls_raw=state, state=state,
                            track_id=tid, cell_inside_box=round(iou, 3), death_frame=df))
        return out

    def boxes(self, t):
        return []


class CtcVideo:
    """CTC 2D training set: <set>/<set>/<SS>/tNNN.tif, <SS>_GT/TRA/man_trackNNN.tif (track markers),
    man_track.txt (L B E P). Labels: mitosis = a parent's last ~30 min before it splits + the daughters' first
    frame; weak_interphase = frames >= 60 min away from any division of that track."""
    dataset = "ctc"

    def __init__(self, root: Path, setname: str, ss: str):
        import pandas as pd
        d = next(p for p in (root / setname / setname, root / setname) if (p / ss).is_dir())
        self.cell_line, self.modality, self.minutes = CTC_META[setname]
        self.name = f"CTC_{setname}_{ss}"
        self.imgs = index_images(d / ss)
        self.frames = sorted(self.imgs)
        self.tra = index_images(d / f"{ss}_GT" / "TRA")
        tr = pd.read_csv(d / f"{ss}_GT" / "TRA" / "man_track.txt", sep=r"\s+", header=None, names=list("LBEP"))
        self.tracks = {int(r.L): (int(r.B), int(r.E), int(r.P)) for r in tr.itertuples()}
        kids = {}
        for L, (B, E, P) in self.tracks.items():
            if P:
                kids.setdefault(P, []).append(L)
        k_before = max(1, round(30 / self.minutes))
        far = round(60 / self.minutes)
        self.state = {}
        for L, (B, E, P) in self.tracks.items():
            div_end = E if L in kids else None
            for t in range(B, E + 1):
                if div_end is not None and t > div_end - k_before:
                    self.state[(L, t)] = "mitosis"
                elif P and P in self.tracks and t == B:
                    self.state[(L, t)] = "mitosis"
                elif (div_end is None or div_end - t >= far) and (not P or t - B >= far):
                    self.state[(L, t)] = "weak_interphase"
        self.treatment = "untreated"

    def read(self, t):
        return read_img(self.imgs[t])

    def labels(self, t, lab):
        if t not in self.tra:
            return []
        import tifffile
        mk = tifffile.imread(self.tra[t]).astype(np.int32)
        out = []
        for L in np.unique(mk):
            st = self.state.get((int(L), t)) if L else None
            if not st:
                continue
            tid, iou = best_overlap(mk == L, lab, min_iou=0.0, min_cover=0.5)
            out.append(dict(frame=t, gt_id=f"{self.name}:{L}", source="ctc", cls_raw=st, state=st, track_id=tid,
                            cell_inside_box=round(iou, 3)))
        return out

    def boxes(self, t):
        return []


def list_videos(a):
    if a.source == "alfi":
        raw = Path(a.raw)
        return [AlfiVideo(raw, s) for s in sequences(raw, a.videos or r"(CD|TP|MI)\d\d")]
    if a.source == "qpi":
        raw = Path(a.raw)
        stems = sorted(p.stem for p in raw.glob("*/QPI_*_*_*.tif"))
        stems = [s for s in stems if re.fullmatch(a.videos or r"QPI_.*", s)]
        return [QpiVideo(raw, s, step=a.qpi_step, dead_min=a.qpi_dead_min, margin_min=a.qpi_margin_min) for s in stems]
    if a.source == "ctc":
        root = Path(a.raw)
        out = []
        for setname in CTC_META:
            for ss in ("01", "02"):
                name = f"CTC_{setname}_{ss}"
                if re.fullmatch(a.videos or r"CTC_.*", name):
                    try:
                        out.append(CtcVideo(root, setname, ss))
                    except StopIteration:
                        log(f"{name}: not found under {root}, skipping")
        return out
    raise SystemExit(f"unknown source {a.source}")


# ----------------------------------------------------------------------------- build
def save_overlay(path, img, lab, labels, boxes):
    """check picture: tracked outlines (yellow), expert boxes (ALFI) or matched cells (dots), coloured by state."""
    from PIL import Image, ImageDraw
    from skimage.measure import regionprops
    from skimage.segmentation import find_boundaries
    lo, hi = np.percentile(img, (1, 99.5))
    g = (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([g] * 3, -1)
    rgb[find_boundaries(lab, mode="inner")] = (255, 220, 0)
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    col = dict(interphase=(0, 200, 255), alive=(0, 200, 255), transition=(200, 200, 200), weak_interphase=(120, 160, 200), mitosis=(0, 255, 0),
               abnormal_division=(255, 0, 255), death=(255, 60, 60), apoptosis=(255, 140, 0), necrosis=(255, 40, 40))
    for b, st in boxes:
        dr.rectangle(b, outline=col[st], width=3)
    cen = {r.label: r.centroid for r in regionprops(lab)}
    for l in labels:
        if l["track_id"] in cen and not boxes:
            y, x = cen[l["track_id"]]
            dr.ellipse((x - 6, y - 6, x + 6, y + 6), fill=col[l["state"]])
    im.save(path)


LABEL_COLS = ["frame", "gt_id", "source", "cls_raw", "state", "track_id", "cell_inside_box", "death_frame"]


def cmd_relabel(a):
    """QPI only, no GPU: recompute labels.csv of already-built videos from the stored masks (new time windows,
    death_frame column, 'transition' rows). The old file is kept as labels_v1.csv."""
    import pandas as pd
    out = Path(a.out)
    for d in sorted((out / "seqs").glob("QPI_*")):
        if not (d / "masks.npz").exists():
            continue
        info = json.loads((d / "info.json").read_text())
        step = max(1, round(info["minutes_per_frame"] / 3))
        v = QpiVideo(Path(a.raw), d.name, step=step, dead_min=a.qpi_dead_min, margin_min=a.qpi_margin_min)
        masks = np.load(d / "masks.npz")
        labs = []
        for k in sorted(masks.files):
            labs += v.labels(int(k[1:]), masks[k].astype(np.int32))
        if not (d / "labels_v1.csv").exists() and (d / "labels.csv").exists():
            (d / "labels.csv").rename(d / "labels_v1.csv")
        df = pd.DataFrame(labs, columns=LABEL_COLS)
        df.to_csv(d / "labels.csv", index=False)
        info["matched"] = round(float((df.track_id > 0).mean()), 3) if len(df) else None
        info["qpi_windows_min"] = dict(dead=a.qpi_dead_min, margin=a.qpi_margin_min)
        (d / "info.json").write_text(json.dumps(info, indent=1))
        log(f"{d.name}: {df.state.value_counts().to_dict()}")


def build_video(v, out, seg, enc, a):
    import pandas as pd
    import ctc_nl_pipeline as P
    od = out / "seqs" / v.name
    od.mkdir(parents=True, exist_ok=True)
    frames = v.frames[: a.max_frames or None]
    first = [v.read(f) for f in (frames[0], frames[len(frames) // 2])]
    diam = seg.choose_size(first) if a.cell_size == "auto" else float(a.cell_size)
    enc_d = a.encode_size or diam * a.encode_ratio
    trk = P.OverlapTracker(spacing=(1.0, 1.0))
    obj, embs, labs, masks = [], [], [], {}
    t0 = time.time()
    for t in frames:
        img = v.read(t)
        tl = trk.update(t, seg.run(img, diam)).astype(np.int32)
        masks[f"t{t:04d}"] = tl.astype(np.uint16)
        meas = measurements(tl, img)
        fm, rs = enc.feature_map(img, enc_d)
        pooled = enc.pool(fm, rs, tl)
        for i, m in meas.items():
            obj.append(dict(frame=t, track_id=i, **m))
            embs.append(pooled[i].astype(np.float16))
        lt = v.labels(t, tl)
        labs += lt
        if a.overlay and t in (frames[0], frames[len(frames) // 2], frames[-1]):
            save_overlay(od / f"overlay_t{t:04d}.png", img, tl, lt, v.boxes(t))
    # same division clean-up as the website (daughters must live >= min_daughter_life frames)
    tracks, relabel, rep = clean_divisions(trk.tracks, frames[-1], a.min_daughter_life)
    if relabel:
        for o in obj:
            o["track_id"] = relabel.get(o["track_id"], o["track_id"])
        for l in labs:
            l["track_id"] = relabel.get(l["track_id"], l["track_id"])
        for k, m in masks.items():
            lut = np.arange(int(m.max()) + 1, dtype=np.uint16)
            for old, new in relabel.items():
                if old < len(lut):
                    lut[old] = new
            masks[k] = lut[m]
    pd.DataFrame(obj).to_csv(od / "objects.csv", index=False)
    np.save(od / "emb.npy", np.stack(embs) if embs else np.zeros((0, 256), np.float16))
    pd.DataFrame(labs, columns=LABEL_COLS).to_csv(od / "labels.csv", index=False)
    (od / "res_track.txt").write_text(res_track_text(tracks))
    np.savez_compressed(od / "masks.npz", **masks)
    lab_df = pd.DataFrame(labs)
    matched = float((lab_df.track_id > 0).mean()) if len(lab_df) else float("nan")
    info = dict(video=v.name, dataset=v.dataset, modality=v.modality, cell_line=v.cell_line,
                cancer=CANCER.get(v.cell_line), treatment=v.treatment, minutes_per_frame=v.minutes,
                frames=[int(frames[0]), int(frames[-1])], n_frames=len(frames), diameter=diam, encode_size=enc_d,
                divisions=rep, seconds=round(time.time() - t0, 1), image_shape=list(first[0].shape),
                matched=None if np.isnan(matched) else round(matched, 3))
    if isinstance(v, QpiVideo):
        info["qpi_cells_found_in_gt_masks"] = f"{len(v.found)}/{len(v.cells)}"
        if v.cells and len(v.found) < 0.5 * len(v.cells):
            log(f"WARNING {v.name}: only {len(v.found)}/{len(v.cells)} labelled cells found in the GT masks "
                "(cell_num may not equal the mask value)")
    (od / "info.json").write_text(json.dumps(info, indent=1))
    log(f"{v.name}: {len(frames)} frames, size {diam} px, {len(obj)} cell-frames, {len(labs)} labels "
        f"({matched:.0%} matched), {time.time() - t0:.0f}s")
    counts = lab_df.state.value_counts().to_dict() if len(lab_df) else {}
    return dict(video=v.name, dataset=v.dataset, cell_line=v.cell_line, treatment=v.treatment, frames=len(frames),
                diameter=diam, cell_frames=len(obj), labels=len(labs), matched=info["matched"],
                **{f"n_{k}": int(counts.get(k, 0)) for k in ("interphase", "mitosis", "abnormal_division", "death",
                                                            "alive", "apoptosis", "necrosis", "weak_interphase")})


def cmd_build(a):
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = list_videos(a)
    log(f"{a.source}: {len(vids)} videos")
    seg = Segmenter(a.model, debug=a.debug_segmenter, batch_size=a.batch_size)
    enc = Encoder(seg, stub=a.debug_encoder)
    rep = []
    for v in vids:
        if (out / "seqs" / v.name / "info.json").exists() and not a.overwrite:
            log(f"{v.name}: done, skipping")
            continue
        rep.append(build_video(v, out, seg, enc, a))
    df = pd.DataFrame(rep)
    old = out / f"build_report_{a.source}.csv"
    if old.exists() and len(df):
        df = pd.concat([pd.read_csv(old)[lambda x: ~x.video.isin(df.video)], df])
    if len(df):
        df.sort_values("video").to_csv(old, index=False)
        log("summary:\n" + df.sort_values("video").to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("--raw", required=True, help="ALFI 'Data&Annotations' folder")
    p.add_argument("--out", required=True)
    p.add_argument("--seqs", default=r"(CD|TP|MI)\d\d")
    rl = sub.add_parser("relabel", help="QPI: recompute labels from stored masks (no GPU)")
    rl.add_argument("--raw", required=True)
    rl.add_argument("--out", required=True)
    rl.add_argument("--qpi-dead-min", type=float, default=180)
    rl.add_argument("--qpi-margin-min", type=float, default=60)
    b = sub.add_parser("build")
    b.add_argument("--source", choices=["alfi", "qpi", "ctc"], required=True)
    b.add_argument("--raw", required=True, help="alfi: Data&Annotations | qpi: .../raw (DU145, LNCaP, PNT1A) | "
                                                "ctc: folder containing DIC-C2DH-HeLa and PhC-C2DH-U373")
    b.add_argument("--out", required=True)
    b.add_argument("--videos", default="", help="regex of video names (ALFI: CD01; QPI: QPI_DU145_st_1; CTC: CTC_...)")
    b.add_argument("--model", default=os.environ.get("CELL_MODEL"), help="whole-cell model (website default)")
    b.add_argument("--cell-size", default="auto", help="auto or a diameter in px (segmentation)")
    b.add_argument("--encode-ratio", type=float, default=0.5,
                   help="encoder scale = cell size x this (0.5 = 2x finer detail per cell; ALFI 120 -> 60)")
    b.add_argument("--encode-size", type=float, default=0, help="fixed encoder scale (overrides --encode-ratio)")
    b.add_argument("--qpi-step", type=int, default=2, help="QPI: use every n-th frame (3 min -> 6 min)")
    b.add_argument("--qpi-dead-min", type=float, default=180, help="QPI: minutes after death labelled dead")
    b.add_argument("--qpi-margin-min", type=float, default=60, help="QPI: minutes before death not labelled alive")
    b.add_argument("--min-daughter-life", type=int, default=3)
    b.add_argument("--batch-size", type=int, default=8)
    b.add_argument("--max-frames", type=int, default=0, help="tests: only the first N frames")
    b.add_argument("--overwrite", action="store_true")
    b.add_argument("--overlay", action="store_true", help="save check pictures (first, middle, last frame)")
    b.add_argument("--debug-segmenter", action="store_true", help="tests: threshold instead of Cellpose")
    b.add_argument("--debug-encoder", action="store_true", help="tests: random small encoder")
    a = ap.parse_args(argv)
    {"inspect": cmd_inspect, "relabel": cmd_relabel, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()