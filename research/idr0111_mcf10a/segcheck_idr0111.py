"""idr0111: fair comparison of the two segmentation models (general Cellpose-SAM vs our fine-tuned cpsam_ft_joint).

  crops   picks 6 crops (512 x 512 px) from 6 different videos: 2 x normal MCF10A, PTEN-/-, KRas, KRas+PTEN-/-,
          MDA-MB-231, each from a different recording day, at a random time point, where the cells are
          -> segcheck/crops/<name>.tif (to annotate) + crops_info.csv (where each crop comes from)
  score   after you drew the cells by hand (Cellpose GUI -> <name>_seg.npy next to each crop):
          both models segment the FULL frame (so cells at the crop border get normal context), the result is cut to
          the crop and compared with your outlines -> precision, recall, F1 at IoU 0.5, mean IoU of matched cells
          -> segcheck/score.md, score.json, and a picture per crop (yours | general | fine-tuned)
  survey  no hand outlines: 2 videos per recording day (all lines), 10 frames each, both models
          -> cells per frame and share of the sheet covered, per model and cell line -> segcheck/survey.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import segtest_idr0111 as ST  # read_video, overlay

CROP = 512


def md(df):
    """markdown table without extra packages"""
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False)])


def line_of(r):
    cl = str(r["Characteristics [Cell Line]"])
    if cl.startswith("MDA"):
        return "MDA-MB-231"
    g = str(r["Experimental Condition [Genetic Perturbation]"])
    return {"none": "MCF10A", "PTEN-/-": "PTEN", "kRas activation": "KRas", "kRas/PTEN-/-": "KRas_PTEN"}.get(g, g)


def table(data):
    v = pd.read_csv(Path(__file__).resolve().parent / "videos_annotation.csv")
    v["line"] = v.apply(line_of, axis=1)
    v["day"] = pd.to_datetime(v["Comment [Experimental Replicate Date]"], dayfirst=True).dt.strftime("%Y-%m-%d")
    v["path"] = [str(Path(data) / d / f) for d, f in zip(v["day"], v["Image File"])]
    return v


def cell_map(img):
    """where the cells are (textured) vs empty dish (smooth)"""
    from scipy.ndimage import uniform_filter
    x = img.astype(np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    m = uniform_filter(x, 9)
    sd = np.sqrt(np.clip(uniform_filter(x * x, 9) - m * m, 0, None))
    return sd > np.percentile(sd, 60) * 0.8


def best_window(img, rng):
    """the 512 x 512 window with the most cells (a few random candidates among the good ones)"""
    from scipy.ndimage import uniform_filter
    c = uniform_filter(cell_map(img).astype(np.float64), CROP)
    H, W = img.shape
    h = CROP // 2
    c = c[h:H - h, h:W - h]
    ok = np.argwhere(c >= np.percentile(c, 90))
    y, x = ok[rng.integers(len(ok))]
    return int(y), int(x)


def crops(a):
    global CROP
    CROP = a.size
    import tifffile
    from PIL import Image
    rng = np.random.default_rng(a.seed)
    v = table(a.data)
    out = Path(a.out) / "crops"
    out.mkdir(parents=True, exist_ok=True)
    want = ["MCF10A", "MCF10A", "PTEN", "KRas", "KRas_PTEN", "MDA-MB-231"]
    used_days, rows = set(), []
    for i, ln in enumerate(want):
        cand = v[(v["line"] == ln) & ~v["day"].isin(used_days)]
        r = cand.iloc[rng.integers(len(cand))]
        used_days.add(r["day"])
        vid, _ = ST.read_video(r["path"])
        t = int(rng.integers(vid.shape[0] // 5, vid.shape[0] - 1))
        y0, x0 = best_window(vid[t], rng)
        name = f"crop{i + 1}_{ln}"
        crop = vid[t, y0:y0 + CROP, x0:x0 + CROP]
        tifffile.imwrite(out / f"{name}.tif", crop)
        lo, hi = np.percentile(crop, [1, 99])
        Image.fromarray((np.clip((crop - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)).save(out / f"{name}_preview.png")
        rows.append(dict(name=name, line=ln, day=r["day"], file=r["Image File"], frame=t, y0=y0, x0=x0, size=CROP))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(Path(a.out) / "crops_info.csv", index=False)
    print("annotate the .tif files in", out.resolve())


def match(gt, pr, thr=0.5):
    """precision, recall, F1 at IoU >= thr (one-to-one), mean IoU of the matched cells"""
    g_ids = np.unique(gt[gt > 0]); p_ids = np.unique(pr[pr > 0])
    if len(g_ids) == 0 or len(p_ids) == 0:
        return dict(gt=len(g_ids), pred=len(p_ids), tp=0, precision=0.0, recall=0.0, f1=0.0, mean_iou=0.0)
    both = (gt > 0) & (pr > 0)
    pairs, inter = np.unique(np.stack([gt[both], pr[both]]), axis=1, return_counts=True)
    ga = dict(zip(*np.unique(gt[gt > 0], return_counts=True)))
    pa = dict(zip(*np.unique(pr[pr > 0], return_counts=True)))
    iou = {(g, p): n / (ga[g] + pa[p] - n) for (g, p), n in zip(pairs.T, inter)}
    good = sorted(((v, g, p) for (g, p), v in iou.items() if v >= thr), reverse=True)
    ug, up, ious = set(), set(), []
    for v, g, p in good:
        if g in ug or p in up:
            continue
        ug.add(g); up.add(p); ious.append(v)
    tp = len(ious)
    P, R = tp / len(p_ids), tp / len(g_ids)
    return dict(gt=len(g_ids), pred=len(p_ids), tp=tp, precision=round(P, 3), recall=round(R, 3),
                f1=round(2 * P * R / max(P + R, 1e-9), 3), mean_iou=round(float(np.mean(ious)) if ious else 0.0, 3))


def models(a):
    from cellpose import models as cm
    ms = {"general": cm.CellposeModel(gpu=bool(a.gpu), pretrained_model="cpsam")}
    if a.ft_model and Path(a.ft_model).exists():
        ms["finetuned"] = cm.CellposeModel(gpu=bool(a.gpu), pretrained_model=a.ft_model)
    else:
        print("fine-tuned model not found:", a.ft_model)
    return ms


def score(a):
    from PIL import Image
    info = pd.read_csv(Path(a.out) / "crops_info.csv")
    cdir = Path(a.out) / "crops"
    v = table(a.data).set_index("Image File")
    ms = models(a)
    res, pics = [], []
    for r in info.itertuples():
        f = cdir / f"{r.name}_seg.npy"
        if not f.exists():
            print("no hand outlines yet:", f.name); continue
        gt = np.load(f, allow_pickle=True).item()["masks"].astype(np.int32)
        vid, _ = ST.read_video(v.loc[r.file, "path"])
        frame = vid[r.frame].astype(np.float32)
        C = gt.shape[0]                                   # crop size (from your outlines)
        img = vid[r.frame, r.y0:r.y0 + C, r.x0:r.x0 + C]
        row = [ST.overlay(img, gt, C)[:, :C]]
        row.append(ST.overlay(img, gt, C)[:, -C:])
        for mn, m in ms.items():
            lab = m.eval(frame, diameter=None, channel_axis=None)[0][r.y0:r.y0 + C, r.x0:r.x0 + C]
            s = match(gt, lab)
            res.append(dict(crop=r.name, line=r.line, model=mn, **s))
            print(res[-1], flush=True)
            row.append(ST.overlay(img, lab, C)[:, -C:])
        gap = np.full((C, 6, 3), 255, np.uint8)
        pics.append(np.hstack(sum([[p, gap] for p in row], [])[:-1]))
        Image.fromarray(pics[-1]).save(Path(a.out) / f"{r.name}_compare.png")
    if not res:
        return
    df = pd.DataFrame(res)
    tot = df.groupby("model")[["gt", "pred", "tp"]].sum()
    tot["precision"] = (tot["tp"] / tot["pred"]).round(3); tot["recall"] = (tot["tp"] / tot["gt"]).round(3)
    tot["f1"] = (2 * tot["precision"] * tot["recall"] / (tot["precision"] + tot["recall"])).round(3)
    tot["mean_iou"] = df.groupby("model")["mean_iou"].mean().round(3)
    L = ["# Segmentation check on idr0111 (hand outlines = truth, IoU >= 0.5)", "",
         "All crops together:", "", md(tot.reset_index()), "", "Per crop:", "",
         md(df), "",
         "Pictures: <crop>_compare.png = raw | your outlines | " + " | ".join(ms) + "."]
    (Path(a.out) / "score.md").write_text("\n".join(L) + "\n")
    (Path(a.out) / "score.json").write_text(json.dumps(dict(total=tot.reset_index().to_dict("records"), per_crop=res), indent=1, default=float))
    print("\n".join(L))


def survey(a):
    rng = np.random.default_rng(a.seed)
    v = table(a.data)
    pick = pd.concat([g.sample(min(2, len(g)), random_state=a.seed) for _, g in v.groupby("day")])
    ms = models(a)
    rows = []
    for r in pick.itertuples():
        vid, _ = ST.read_video(r.path)
        for t in np.linspace(0, vid.shape[0] - 1, 10).round().astype(int):
            img = vid[t].astype(np.float32)
            sheet = cell_map(img)
            for mn, m in ms.items():
                lab = m.eval(img, diameter=None, channel_axis=None)[0]
                rows.append(dict(day=r.day, line=r.line, video=r.path.split("/")[-1], frame=int(t), model=mn,
                                 cells=int(len(np.unique(lab[lab > 0]))),
                                 sheet_covered=round(float(((lab > 0) & sheet).sum() / max(sheet.sum(), 1)), 3)))
        print(f"{r.day} {r.line}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out) / "survey.csv", index=False)
    s = df.groupby(["line", "model"]).agg(videos=("video", "nunique"), cells_per_frame=("cells", "mean"),
                                           sheet_covered=("sheet_covered", "mean")).round(3).reset_index()
    L = ["# Segmentation survey on idr0111 (no hand outlines): 2 videos per day, 10 frames each", "",
         "sheet_covered = share of the textured (cell) area that lies inside a segmented cell", "", md(s)]
    (Path(a.out) / "survey.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["crops", "score", "survey"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="segcheck")
    ap.add_argument("--ft-model", default="")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=512, help="crop size in px (crops step); 256 = about 4x fewer cells to draw")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"crops": crops, "score": score, "survey": survey}[a.step](a)


if __name__ == "__main__":
    main()