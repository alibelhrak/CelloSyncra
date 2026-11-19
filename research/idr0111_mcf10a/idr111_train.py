"""idr0111: normal MCF10A vs MCF10A with cancer mutations (and, as a second test, MCF10A vs MDA-MB-231).

Rules fixed BEFORE the run (run once; do not tune on the result):
  unit        one tracked cell (>= 1 h); scores are also averaged per video (well position), the more robust unit
  test        leave-one-DAY-out: every day holds normal and abnormal wells, so the held-out day is never seen and
              the day cannot be learned (10 days in the mutant series, 6 in the MDA-MB-231 series)
  models      hybrid  = all measured features + frozen Cellpose-SAM features (PCA fitted inside each fold)   <- decides
              hand    = measured features only (shape, brightness, texture, movement, neighbours, divisions)
              density = how packed a cell is only (number of neighbours, distance to open space): answers
                        "is the model more than a density detector?"
              each = one gradient-boosted tree classifier; classes and lines weighted equally within each day
  reference   the whole procedure re-run with normal/abnormal shuffled between the WELLS of each day (20 times)
  PASS        hybrid video AUC and cell AUC above the reference's 95th percentile, AND on at least 8 of the 10
              held-out days the normal videos score lower on average than the mutant videos
              (MDA-MB-231 series: at least 4 of 6)
  also        AUC of each line vs MCF10A, per-day AUC, SHAP by feature family, the density baseline

Research score of how much a cell behaves like MCF10A cells carrying cancer mutations; not a diagnosis.

  python idr111_train.py --features features --series mutants --out results/mutants
  python idr111_train.py --features features --series mda231  --out results/mda231
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

META = {"track", "uid", "video", "day", "line", "series", "well", "n_frames", "first_frame", "border_fraction"}
DENSITY = ["n_neighbours_mean", "n_neighbours_std", "n_neighbours_dt", "dist_open_um_mean", "dist_open_um_std", "dist_open_um_dt"]
NAMES = {"hybrid": "hybrid: measured + frozen Cellpose-SAM features", "hand": "measured features only",
         "density": "packing only (neighbours, distance to open space)"}


def family(c):
    if c.startswith("vit_"):
        return "image features (frozen ViT)"
    if c in DENSITY:
        return "packing (neighbours, open space)"
    if c.startswith(("alignment", "rel_speed")):
        return "collective movement"
    if c.startswith(("speed", "straightness", "net_speed", "persistence")):
        return "movement"
    if c.startswith(("int_", "texture")):
        return "brightness and texture"
    if c in ("divides", "born_by_division"):
        return "divisions"
    return "shape"


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = np.asarray(s)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


class Features:
    """the columns a model gets; ViT features compressed by a PCA fitted on the training cells of each fold"""

    def __init__(self, kind, hand, emb, n_pc):
        self.kind, self.hand, self.emb, self.n_pc = kind, hand, emb, n_pc

    def fit(self, rows):
        if self.kind != "hybrid":
            return self
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        self.parts = {}
        for k, n in (("mean", self.n_pc), ("std", max(4, self.n_pc // 4))):
            X = self.emb[k][rows]
            ok = ~np.isnan(X).any(1)
            sc = StandardScaler().fit(X[ok])
            self.parts[k] = (sc, PCA(n, random_state=0).fit(sc.transform(X[ok])))
        return self

    def X(self, df, rows):
        if self.kind == "density":
            return df.loc[rows, DENSITY].to_numpy(float), list(DENSITY)
        X, cols = [df.loc[rows, self.hand].to_numpy(float)], list(self.hand)
        if self.kind == "hybrid":
            for k, (sc, pca) in self.parts.items():
                E = self.emb[k][rows]
                ok = ~np.isnan(E).any(1)
                Z = np.full((len(rows), pca.n_components_), np.nan)
                if ok.any():
                    Z[ok] = pca.transform(sc.transform(E[ok]))
                X.append(Z); cols += [f"vit_{k}_pc{i + 1}" for i in range(pca.n_components_)]
            st = self.emb["step"][rows][:, None]
            X.append(st); cols.append("vit_step")
        return np.hstack(X), cols


def weights(d):
    """each day counts equally; inside a day both classes count equally; inside a class every line counts equally"""
    g = d.groupby(["day", "y", "line"])["y"].transform("size")
    n_lines = d.groupby(["day", "y"])["line"].transform("nunique")
    w = 1.0 / (g * n_lines)
    return (w / w.mean()).to_numpy()


def model(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                          l2_regularization=1.0, random_state=seed)


def lodo(F, df, y, seed, max_train=60000):
    """leave-one-day-out scores for every cell"""
    rng = np.random.default_rng(seed)
    s = np.full(len(df), np.nan)
    d = df.assign(y=y)
    for day in sorted(df["day"].unique()):
        te = np.flatnonzero(df["day"].to_numpy() == day)
        tr = np.flatnonzero(df["day"].to_numpy() != day)
        if len(tr) > max_train:
            tr = np.sort(rng.choice(tr, max_train, replace=False))
        F.fit(tr)
        Xtr, _ = F.X(df, tr)
        m = model(seed).fit(Xtr, y[tr], sample_weight=weights(d.iloc[tr]))
        Xte, _ = F.X(df, te)
        s[te] = m.predict_proba(Xte)[:, 1]
    return s


def summarise(df, y, s):
    d = df[["day", "video", "line"]].assign(y=y, s=s)
    v = d.groupby("video").agg(day=("day", "first"), line=("line", "first"), y=("y", "first"), s=("s", "mean"))
    per_day, ok_days = {}, 0
    for day, g in v.groupby("day"):
        per_day[day] = round(auc(g["y"], g["s"]), 3)
        ok_days += int(g.loc[g["y"] == 0, "s"].mean() < g.loc[g["y"] == 1, "s"].mean())
    cell_day = {day: round(auc(g["y"], g["s"]), 3) for day, g in d.groupby("day")}
    return dict(cell_auc=round(auc(d["y"], d["s"]), 3), video_auc=round(auc(v["y"], v["s"]), 3),
                days_normal_lower=int(ok_days), n_days=int(v["day"].nunique()),
                video_auc_per_day=per_day, cell_auc_per_day=cell_day), v


def shuffled_labels(df, y, rng):
    """normal/abnormal shuffled between the wells (videos) of each day: same number of normal wells per day"""
    v = df.groupby("video").agg(day=("day", "first")).assign(y=pd.Series(y, index=df.index).groupby(df["video"]).first())
    new = {}
    for _, g in v.groupby("day"):
        new.update(dict(zip(g.index, rng.permutation(g["y"].to_numpy()))))
    return df["video"].map(new).to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="features")
    ap.add_argument("--series", choices=["mutants", "mda231"], default="mutants")
    ap.add_argument("--out", default="")
    ap.add_argument("--sets", default="hybrid,hand,density")
    ap.add_argument("--n-pc", type=int, default=32)
    ap.add_argument("--null", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out or f"results/{a.series}"); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(a.features) / "tracks.csv.gz", dtype={"day": str, "well": str})
    df = df[df["series"] == a.series].reset_index(drop=True)
    hand = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    z = np.load(Path(a.features) / "emb.npz")
    pos = {u: i for i, u in enumerate(z["uid"])}
    ix = df["uid"].map(pos)
    has = ix.notna().to_numpy()
    D = z["emb_mean"].shape[1]
    emb = {k: np.full((len(df), D), np.nan, np.float32) for k in ("mean", "std")}
    emb["step"] = np.full(len(df), np.nan, np.float32)
    ii = ix[has].astype(int).to_numpy()
    emb["mean"][has] = z["emb_mean"][ii]; emb["std"][has] = z["emb_std"][ii]; emb["step"][has] = z["emb_step"][ii]
    y = (df["line"] != "MCF10A").astype(int).to_numpy()
    data = dict(cells=int(len(df)), with_image_features=int(has.sum()), videos=int(df["video"].nunique()),
                days=int(df["day"].nunique()), cells_per_line=df["line"].value_counts().to_dict(), measured_features=len(hand))
    print("data:", json.dumps(data), flush=True)

    sets = [s.strip() for s in a.sets.split(",") if s.strip()]
    res, vids = {}, {}
    for st in sets:
        s = lodo(Features(st, hand, emb, a.n_pc), df, y, a.seed)
        r, v = summarise(df, y, s)
        r["per_line_auc_vs_MCF10A"] = {ln: round(auc((df.loc[m, "line"] != "MCF10A").astype(int), s[m.to_numpy()]), 3)
                                       for ln in sorted(df["line"].unique()) if ln != "MCF10A"
                                       for m in [df["line"].isin(["MCF10A", ln])]}
        res[st], vids[st] = r, v
        df.assign(score=s, y=y)[["uid", "video", "day", "line", "y", "score"]].to_csv(out / f"scores_{st}.csv.gz", index=False)
        print(f"[{st}] cell AUC {r['cell_auc']} | video AUC {r['video_auc']} | normal lower on {r['days_normal_lower']}/{r['n_days']} days "
              f"| per line {r['per_line_auc_vs_MCF10A']} ({time.time() - t0:.0f} s)", flush=True)

    main_set = "hybrid" if "hybrid" in sets else sets[0]
    ref = None
    if a.null:
        rng = np.random.default_rng(a.seed + 1)
        cv, vv = [], []
        for i in range(a.null):
            yn = shuffled_labels(df, y, rng)
            r, _ = summarise(df, yn, lodo(Features(main_set, hand, emb, a.n_pc), df, yn, a.seed))
            cv.append(r["cell_auc"]); vv.append(r["video_auc"])
            print(f"  reference {i + 1}/{a.null}: cell AUC {r['cell_auc']}, video AUC {r['video_auc']}", flush=True)
        ref = dict(n=a.null, cell_auc_mean=round(float(np.mean(cv)), 3), cell_auc_p95=round(float(np.percentile(cv, 95)), 3),
                   video_auc_mean=round(float(np.mean(vv)), 3), video_auc_p95=round(float(np.percentile(vv, 95)), 3))

    r = res[main_set]
    passed = bool(ref and r["video_auc"] > ref["video_auc_p95"] and r["cell_auc"] > ref["cell_auc_p95"]
                  and r["days_normal_lower"] >= r["n_days"] - 2)

    # final model on every day + SHAP by feature family (what drives the score)
    shap_out = {}
    try:
        import shap
        F = Features(main_set, hand, emb, a.n_pc).fit(np.arange(len(df)))
        X, cols = F.X(df, np.arange(len(df)))
        m = model(a.seed).fit(X, y, sample_weight=weights(df.assign(y=y)))
        rs = np.random.default_rng(a.seed).choice(len(df), min(4000, len(df)), replace=False)
        sv = shap.TreeExplainer(m).shap_values(X[rs])
        sv = sv[1] if isinstance(sv, list) else sv
        imp = pd.Series(np.abs(sv).mean(0), index=cols)
        fam = imp.groupby([family(c) for c in cols]).sum()
        shap_out = dict(by_family_percent=(100 * fam / fam.sum()).sort_values(ascending=False).round(1).to_dict(),
                        top_features=imp.sort_values(ascending=False).head(15).round(4).to_dict())
        import joblib
        joblib.dump(dict(model=m, features=cols, feature_set=main_set, series=a.series, pca=getattr(F, "parts", None)), out / "model.pkl")
    except Exception as e:
        shap_out = dict(error=f"{type(e).__name__}: {e}")

    summ = dict(series=a.series, data=data, models=res, reference=ref, passed=passed, shap=shap_out, minutes=round((time.time() - t0) / 60, 1))
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    lab = "MCF10A vs MCF10A + PTEN-/-, KRas, KRas+PTEN-/-" if a.series == "mutants" else "MCF10A vs MDA-MB-231"
    L = [f"# idr0111 {lab}: leave-one-day-out", "",
         f"{data['cells']:,} tracked cells in {data['videos']} videos over {data['days']} days "
         f"({data['with_image_features']:,} with image features)", "",
         "| model | cell AUC | video AUC | days where normal scores lower | AUC per line vs MCF10A |", "|---|---|---|---|---|"]
    for st, x in res.items():
        L.append(f"| {NAMES.get(st, st)} | {x['cell_auc']} | {x['video_auc']} | {x['days_normal_lower']} / {x['n_days']} | "
                 + ", ".join(f"{k} {v}" for k, v in x["per_line_auc_vs_MCF10A"].items()) + " |")
    L += ["", f"Video AUC per held-out day ({main_set}): " + ", ".join(f"{k} {v}" for k, v in r["video_auc_per_day"].items())]
    if ref:
        L += ["", f"**No-signal reference** (labels shuffled between the wells of each day, whole procedure re-run {ref['n']} times): "
                  f"cell AUC mean {ref['cell_auc_mean']} (95th pct {ref['cell_auc_p95']}), video AUC mean {ref['video_auc_mean']} "
                  f"(95th pct {ref['video_auc_p95']})."]
    if shap_out.get("by_family_percent"):
        L += ["", "What drives the score (SHAP, % by feature family): " + ", ".join(f"{k} {v}%" for k, v in shap_out["by_family_percent"].items())]
    L += ["", f"**Verdict ({main_set}):** " + ("PASS: beats the no-signal reference and separates normal from mutant wells on held-out days."
                                              if passed else "does NOT pass the rule fixed before the run."),
          "", "_Research score of how much a cell behaves like MCF10A cells with cancer mutations; not a diagnosis._"]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
