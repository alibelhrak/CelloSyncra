"""High vs low metastatic efficiency from single-cell behaviour (idr0109), evaluated honestly.

Unit: one tracked cell (tracks.csv.gz from idr_build.py). Label: metEff of the tumour (PDX) the cell comes from.
Only 7 tumours carry a label (3 high, 4 low), so the effective sample size is 7, not thousands of cells.

Evaluation
  1. leave-one-tumour-out (LOTO): train on 6 tumours, score the 7th; repeated for all 7 -> out-of-fold scores.
  2. tumour level: mean score per held-out tumour; AUC over the 7 tumours with an exact permutation p-value
     (all 35 ways to pick 3 'high' of 7).
  3. cell level: pooled AUC of the out-of-fold scores (optimistic: cells of one tumour are not independent).
  4. same-day check: AUC inside each recording day that holds both high and low tumours (removes day effects).
  5. shortcut check: how well the same features predict the recording day (if very well, day effects are strong).
  6. untouched groups, scored by a model trained on all 7 tumours: clones of m405 (high), tumours with unknown
     outcome (um12, ut8), melanoma cell lines, normal melanocytes (atcc, m116). Reported, never used for training or for choosing anything.
  7. SHAP (TreeExplainer) of the final model, grouped into shape / brightness / texture / movement / fluctuation.

Feature sets compared on the same cells and the same folds (with --embed):
  hand        hand-made features (shape, brightness, texture, movement, change over time)   -> baseline
  vit_motion  Cellpose-SAM ViT features (PCA fitted inside each fold) + movement
  hybrid      ViT features + all hand-made features                                       -> expected best
Each gets its own folder with the full report; comparison.md puts them side by side.

Research use only: a score of how much a cell behaves like cells from highly metastatic tumours. Not a diagnosis.

  python idr_train.py --tracks out/idr_features/tracks.csv.gz --out out/idr_train

--day-norm: every feature (hand-made and ViT) is standardised within its recording day, using all cells filmed that
day (labelled or not; labels are never used). This removes day-to-day imaging differences, the way the paper
compares cells imaged on the same day. It also removes whatever differs only between days, so a tumour filmed
alone on its days keeps only its spread around that day, not its offset.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

META = ["track", "cellType", "metEff", "date", "expStr", "locationStr", "n_frames", "t_start", "n_masked"]
GROUPS = {"movement": ("speed", "straight", "net_disp", "msd", "turn"),
          "shape": ("area", "perimeter", "major", "minor", "aspect", "eccentric", "solidity", "extent", "circular", "rough"),
          "brightness": ("int_mean", "int_std", "int_p", "contrast", "int_cv"),
          "texture": ("grad", "lap")}


def group_of(col):
    if col.startswith("vit_"):
        return "image (ViT)"
    base = col.rsplit("_", 1)[0] if col.endswith(("_mean", "_std", "_dt")) else col
    if col.endswith("_dt"):
        return "fluctuation over time"
    for g, keys in GROUPS.items():
        if base.startswith(keys) or any(base.startswith(k) for k in keys):
            return g
    return "other"


def model(seed=0, n=10000):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=int(min(40, max(5, n // 50))), l2_regularization=1.0,
                                          random_state=seed)


def weights(df):
    """every tumour counts equally inside its class, and both classes count equally."""
    n_t = df.groupby("cellType")["cellType"].transform("size")
    n_c = df.groupby("y")["cellType"].transform("nunique")
    w = 1.0 / (n_t * n_c)
    return (w / w.mean()).to_numpy()


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else None


def exact_perm_p(scores_by_tumour, labels_by_tumour):
    """P(AUC >= observed) over all assignments of the high labels to tumours."""
    t = list(scores_by_tumour)
    s = np.array([scores_by_tumour[k] for k in t])
    y = np.array([labels_by_tumour[k] for k in t])
    obs = auc(y, s)
    k = int(y.sum())
    hits = tot = 0
    for idx in itertools.combinations(range(len(t)), k):
        yy = np.zeros(len(t), int)
        yy[list(idx)] = 1
        tot += 1
        hits += auc(yy, s) >= obs - 1e-12
    return obs, hits / tot, tot


MOTION = ("speed_", "straightness", "net_disp", "msd_alpha", "turn_mean")

# Metastatic efficiency of the PDX tumours. Zaritsky et al. (2021): 7 PDX with known efficiency, "4 low, 3 high",
# and 2 unknown. idr0109's mask table matches this; its ROI table swaps um12 and m597 (um12 'high', m597 'na'),
# which gives 4 high / 3 low and contradicts the paper - so labels are taken from here, not from the table.
MET_LABELS = {"m405": "high", "m481": "high", "m634": "high",
              "m498": "low", "m528": "low", "m597": "low", "m610": "low",
              "um12": "na", "ut8": "na"}


class Features:
    """builds the model input for one feature set. ViT parts are fitted on the training tumours only
    (standardise + PCA), inside every leave-one-tumour-out fold, so nothing leaks from the held-out tumour."""

    def __init__(self, kind, hand_cols, emb=None, n_pc=64):
        self.kind, self.hand, self.emb, self.n_pc = kind, hand_cols, emb, n_pc
        self.motion = [c for c in hand_cols if c.startswith(MOTION)]

    def fit(self, idx):
        if self.kind != "hand":
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            E = self.emb["mean"][idx]
            self.sc = StandardScaler().fit(E)
            self.pca = PCA(min(self.n_pc, E.shape[1], len(idx) - 1), random_state=0).fit(self.sc.transform(E))
        return self

    def X(self, df, idx):
        parts = []
        if self.kind in ("hand", "hybrid"):
            parts.append(df.loc[idx, self.hand].reset_index(drop=True))
        if self.kind == "vit_motion":
            parts.append(df.loc[idx, self.motion].reset_index(drop=True))
        if self.kind != "hand":
            Z = self.pca.transform(self.sc.transform(self.emb["mean"][idx]))
            v = pd.DataFrame(Z, columns=[f"vit_pc{i + 1:02d}" for i in range(Z.shape[1])])
            v["vit_spread"] = self.emb["std"][idx].mean(1)
            v["vit_step"] = self.emb["step"][idx]
            parts.append(v)
        return pd.concat(parts, axis=1)


SET_NAMES = {"hand": "hand-made features (shape, brightness, texture, movement)",
             "vit_motion": "ViT image features + movement",
             "hybrid": "hybrid: ViT image features + all hand-made features"}


def day_normalise(df, cols, emb, min_cells=30):
    """z-score each feature inside each recording day (days with < min_cells cells: whole-data statistics)."""
    df = df.copy()
    X = df[cols].astype(float)
    g = df["date"].fillna("?")
    big = g.map(g.value_counts()) >= min_cells
    mu = X.groupby(g).transform("mean").where(big, X.mean(), axis=0)
    sd = X.groupby(g).transform("std").where(big, X.std(), axis=0)
    df[cols] = ((X - mu) / sd.replace(0, np.nan)).fillna(0.0)
    if emb is not None:
        emb = dict(emb)
        for k in ("mean", "std"):
            E = pd.DataFrame(emb[k])
            m_ = E.groupby(g.values).transform("mean").to_numpy().copy()
            s_ = E.groupby(g.values).transform("std").to_numpy().copy()
            m_[~big.values], s_[~big.values] = E.mean().to_numpy(), E.std().to_numpy()
            emb[k] = np.nan_to_num((E.to_numpy() - m_) / np.where(s_ > 0, s_, np.nan)).astype(np.float32)
        st = pd.Series(emb["step"])
        m_ = st.groupby(g.values).transform("mean").where(big.values, st.mean())
        s_ = st.groupby(g.values).transform("std").where(big.values, st.std())
        emb["step"] = np.nan_to_num(((st - m_) / s_.replace(0, np.nan)).to_numpy()).astype(np.float32)
    return df, emb, int(big.sum()), int(g[big].nunique())


def loto_null(F, df, lab, n, seed):
    """What leave-one-tumour-out gives when there is NO signal: re-run the whole procedure (training included) with
    the high/low labels randomly re-assigned to the 7 tumours (3 high, 4 low), same feature set F. With few, very
    different tumours this reference is often below 0.5; a real result has to beat it, not just 0.5."""
    rng = np.random.default_rng(seed + 1)
    tumours = sorted(lab["cellType"].unique())
    n_hi = int(lab.groupby("cellType")["y"].first().sum())
    true_hi = set(lab.loc[lab["y"] == 1, "cellType"])
    combos = [set(c) for c in itertools.combinations(tumours, n_hi) if set(c) != true_hi]
    rng.shuffle(combos)
    res = []
    for hi in combos[:n]:
        L = lab.copy()
        L["y"] = L["cellType"].isin(hi).astype(int)
        L["oof"] = np.nan
        for t in tumours:
            tr, te = L.index[L["cellType"] != t], L.index[L["cellType"] == t]
            F.fit(L.loc[tr, "row"].to_numpy())
            m = model(seed, len(tr)).fit(F.X(df, L.loc[tr, "row"].to_numpy()), L.loc[tr, "y"], sample_weight=weights(L.loc[tr]))
            L.loc[te, "oof"] = m.predict_proba(F.X(df, L.loc[te, "row"].to_numpy()))[:, 1]
        per = L.groupby("cellType").agg(y=("y", "first"), s=("oof", "mean"))
        res.append(dict(tumour_auc=auc(per["y"], per["s"]), cell_auc=auc(L["y"], L["oof"])))
        print(f"  null {len(res)}/{min(n, len(combos))}: tumour AUC {res[-1]['tumour_auc']:.3f}", flush=True)
    ta = np.array([r["tumour_auc"] for r in res])
    ca = np.array([r["cell_auc"] for r in res])
    return dict(n=len(res), tumour_auc_mean=round(float(ta.mean()), 3), tumour_auc_p95=round(float(np.percentile(ta, 95)), 3),
                cell_auc_mean=round(float(ca.mean()), 3), cell_auc_p95=round(float(np.percentile(ca, 95)), 3))


def cluster_boot_auc(lab, col, n=2000, seed=0):
    """95% CI of the pooled cell AUC, resampling whole tumours (within each class) with replacement."""
    rng = np.random.default_rng(seed)
    by = {t: g for t, g in lab.groupby("cellType")}
    hi = [t for t, g in by.items() if g["y"].iloc[0] == 1]
    lo = [t for t, g in by.items() if g["y"].iloc[0] == 0]
    vals = []
    for _ in range(n):
        pick = list(rng.choice(hi, len(hi))) + list(rng.choice(lo, len(lo)))
        g = pd.concat([by[t] for t in pick])
        vals.append(auc(g["y"], g[col]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def evaluate(name, F, df, lab, rest, out, seed):
    """leave-one-tumour-out + checks + final model for one feature set; writes out/<name>/."""
    o = out / name
    o.mkdir(parents=True, exist_ok=True)
    tumours = sorted(lab["cellType"].unique())
    lab = lab.copy()
    lab["oof"] = np.nan
    for t in tumours:
        tr = lab.index[lab["cellType"] != t]
        te = lab.index[lab["cellType"] == t]
        F.fit(lab.loc[tr, "row"].to_numpy())
        Xtr, Xte = F.X(df, lab.loc[tr, "row"].to_numpy()), F.X(df, lab.loc[te, "row"].to_numpy())
        m = model(seed, len(tr)).fit(Xtr, lab.loc[tr, "y"], sample_weight=weights(lab.loc[tr]))
        lab.loc[te, "oof"] = m.predict_proba(Xte)[:, 1]
    per_t = lab.groupby("cellType").agg(label=("metEff", "first"), cells=("oof", "size"), mean_score=("oof", "mean"),
                                        median_score=("oof", "median"), frac_above_half=("oof", lambda v: float((v > .5).mean())))
    t_auc, t_p, n_perm = exact_perm_p(per_t["mean_score"].to_dict(), (per_t["label"] == "high").astype(int).to_dict())
    rep = dict(feature_set=SET_NAMES[name])
    rep["leave_one_tumour_out"] = dict(
        tumour_level_auc=round(t_auc, 3), tumour_level_p_exact=round(t_p, 4), n_label_permutations=n_perm,
        cell_level_auc_pooled=round(auc(lab["y"], lab["oof"]), 3),
        cell_level_auc_95ci_tumour_bootstrap=cluster_boot_auc(lab, "oof", seed=seed),
        per_tumour=per_t.round(3).reset_index().to_dict("records"))
    days = []
    for d, g in lab.groupby("date"):
        if g["y"].nunique() == 2:
            days.append(dict(day=d, tumours=sorted(g["cellType"].unique()), cells=len(g), auc=round(auc(g["y"], g["oof"]), 3)))
    rep["same_day_check"] = dict(days=days, median_auc=round(float(np.median([x["auc"] for x in days])), 3) if days else None)

    # final model on all labelled tumours (ViT PCA fitted on them too)
    rows = lab["row"].to_numpy()
    F.fit(rows)
    Xall = F.X(df, rows)
    feats = list(Xall.columns)
    rep["n_features"] = len(feats)
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    vc = lab["date"].value_counts()
    keep = lab["date"].isin(vc[vc >= 10].index).to_numpy()
    if keep.sum() > 50 and lab.loc[keep, "date"].nunique() > 2:
        acc = cross_val_score(HistGradientBoostingClassifier(max_iter=150, random_state=seed), Xall[keep],
                              lab.loc[keep, "date"], cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean()
        rep["shortcut_check_day"] = dict(accuracy=round(float(acc), 3), chance=round(float(vc[vc >= 10].max() / keep.sum()), 3),
                                         n_days=int(lab.loc[keep, "date"].nunique()))
    final = model(seed, len(lab)).fit(Xall, lab["y"], sample_weight=weights(lab))
    if len(rest):
        rest = rest.copy()
        rest["score"] = final.predict_proba(F.X(df, rest["row"].to_numpy()))[:, 1]
        # groups as named on IDR (idr0109 / experimentA): Melanocyte_ATCC and Melanocyte_m116 are normal melanocytes
        kind = lambda c: ("clone of m405 (high)" if c.startswith("m405c") else "clone of a375" if c.startswith("a375c")
                          else "melanocyte (normal)" if c in ("atcc", "m116")
                          else "tumour (PDX), outcome unknown" if re.match(r"^(m|um|ut)\d", c)
                          else "melanoma cell line")
        rest["group"] = rest["cellType"].map(kind)
        rep["untouched_groups"] = (rest.groupby(["group", "cellType"])["score"].agg(["size", "mean", "median"])
                                   .round(3).reset_index().to_dict("records"))
        rest[["track", "cellType", "group", "expStr", "score"]].to_csv(o / "scores_untouched.csv", index=False)
    try:
        import shap
        Xs = Xall.sample(min(3000, len(Xall)), random_state=seed)
        sv = shap.TreeExplainer(final).shap_values(Xs)
        sv = sv[1] if isinstance(sv, list) else sv
        imp = pd.Series(np.abs(sv).mean(0), index=feats)
        rep["shap_top_features"] = imp.sort_values(ascending=False).head(15).round(4).to_dict()
        g = imp.groupby([group_of(c) for c in feats]).sum()
        rep["shap_by_group_percent"] = (100 * g / g.sum()).sort_values(ascending=False).round(1).to_dict()
    except Exception as e:
        rep["shap_error"] = f"{type(e).__name__}: {e}"
    lab[["track", "cellType", "metEff", "date", "expStr", "oof"]].to_csv(o / "scores_loto.csv", index=False)
    import joblib
    joblib.dump(dict(model=final, features=feats, feature_set=name,
                     vit=None if name == "hand" else dict(scaler=F.sc, pca=F.pca)), o / "model.pkl")
    (o / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    write_md(rep, o / "report.md")
    plot(per_t, rest if len(rest) else None, o / "scores.png")
    L = rep["leave_one_tumour_out"]
    print(f"[{name}] tumour AUC {L['tumour_level_auc']} (p {L['tumour_level_p_exact']}), cell AUC "
          f"{L['cell_level_auc_pooled']} {L['cell_level_auc_95ci_tumour_bootstrap']}, same-day "
          f"{rep['same_day_check']['median_auc']}", flush=True)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--embed", default=None, help="embeddings.npz from idr_embed.py (adds the ViT and hybrid models)")
    ap.add_argument("--sets", default="hand,vit_motion,hybrid", help="feature sets to compare")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-frames", type=int, default=30, help="drop shorter tracks")
    ap.add_argument("--n-pc", type=int, default=64, help="ViT features kept after PCA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null", type=int, default=20, help="random label assignments for the no-signal reference (0 = skip)")
    ap.add_argument("--null-set", default="", help="feature set of the no-signal reference (default: hybrid if run, else the first)")
    ap.add_argument("--day-norm", action="store_true", help="standardise every feature within its recording day (see top)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.tracks, dtype={"date": str, "locationStr": str, "metEff": str})
    df["metEff"] = df["metEff"].fillna("na").str.lower()
    fixed = {}
    for ct, lab in MET_LABELS.items():
        old = set(df.loc[df["cellType"] == ct, "metEff"])
        if old and old != {lab}:
            fixed[ct] = f"{'/'.join(sorted(old))} -> {lab}"
        df.loc[df["cellType"] == ct, "metEff"] = lab
    if fixed:
        print("labels corrected to the paper (4 low, 3 high): " + "; ".join(f"{k} {v}" for k, v in fixed.items()), flush=True)
    hand = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    df = df[(df["n_frames"] >= a.min_frames) & (df["n_masked"] >= 3)]
    sets = [s.strip() for s in a.sets.split(",") if s.strip()]
    emb = None
    if a.embed:
        z = np.load(a.embed)
        pos = {t: i for i, t in enumerate(z["track"])}
        n0 = len(df)
        df = df[df["track"].isin(pos)]                      # same cells for every model, so they compare fairly
        ix = df["track"].map(pos).to_numpy()
        emb = dict(mean=z["emb_mean"][ix], std=z["emb_std"][ix], step=z["emb_step"][ix])
        print(f"ViT features for {len(df):,} of {n0:,} tracks", flush=True)
    else:
        sets = [s for s in sets if s == "hand"]
    df = df.reset_index(drop=True)
    if a.day_norm:
        df, emb, n_norm, n_days = day_normalise(df, hand, emb)
        print(f"day normalisation: {n_norm:,} of {len(df):,} cells standardised within their own day ({n_days} days)", flush=True)
    df["row"] = np.arange(len(df))
    lab = df[df["metEff"].isin(["high", "low"])].copy()
    lab["y"] = (lab["metEff"] == "high").astype(int)
    rest = df[~df["metEff"].isin(["high", "low"])]
    print(f"{len(lab):,} labelled cells from {lab['cellType'].nunique()} tumours; sets: {', '.join(sets)}", flush=True)

    reps = {s: evaluate(s, Features(s, hand, emb, a.n_pc), df, lab, rest, out, a.seed) for s in sets}
    ns = a.null_set or ("hybrid" if "hybrid" in sets else sets[0])
    null = loto_null(Features(ns, hand, emb, a.n_pc), df, lab, a.null, a.seed) if a.null else None
    if null:
        null["feature_set"] = SET_NAMES[ns]
    comp = dict(labels_corrected=fixed, null_hand=null, day_norm=bool(a.day_norm),
                cells_per_tumour=lab.groupby(["cellType", "metEff"]).size().rename("cells").reset_index().to_dict("records"),
                models={s: dict(feature_set=r["feature_set"], n_features=r["n_features"],
                                tumour_auc=r["leave_one_tumour_out"]["tumour_level_auc"],
                                tumour_p=r["leave_one_tumour_out"]["tumour_level_p_exact"],
                                cell_auc=r["leave_one_tumour_out"]["cell_level_auc_pooled"],
                                cell_auc_ci=r["leave_one_tumour_out"]["cell_level_auc_95ci_tumour_bootstrap"],
                                same_day_auc=r["same_day_check"]["median_auc"],
                                day_shortcut=r.get("shortcut_check_day", {}).get("accuracy"),
                                per_tumour={t["cellType"]: t["mean_score"] for t in r["leave_one_tumour_out"]["per_tumour"]})
                        for s, r in reps.items()})
    (out / "comparison.json").write_text(json.dumps(comp, indent=1, default=float))
    write_comparison(comp, out / "comparison.md")
    print((out / "comparison.md").read_text())


def write_comparison(c, path):
    M = c["models"]
    L = ["# Model comparison — high vs low metastatic efficiency (idr0109), leave-one-tumour-out"
         + (", features standardised within each recording day" if c.get("day_norm") else ""), "",
         "| model | features | tumour AUC (exact p) | cell AUC [95% CI, tumour bootstrap] | same-day AUC | predicts recording day |",
         "|---|---|---|---|---|---|"]
    for s, m in M.items():
        L.append(f"| {m['feature_set']} | {m['n_features']} | {m['tumour_auc']} ({m['tumour_p']}) | {m['cell_auc']} "
                 f"{m['cell_auc_ci']} | {m['same_day_auc']} | {m['day_shortcut']} |")
    tumours = sorted(next(iter(M.values()))["per_tumour"])
    lab = {r["cellType"]: r["metEff"] for r in c["cells_per_tumour"]}
    L += ["", "Mean score of each held-out tumour (the model never saw it):", "",
          "| tumour | label | " + " | ".join(M) + " |", "|---|---|" + "---|" * len(M)]
    for t in tumours:
        L.append(f"| {t} | {lab.get(t, '')} | " + " | ".join(str(M[s]['per_tumour'][t]) for s in M) + " |")
    if c.get("labels_corrected"):
        L += ["", "Labels corrected to the paper (4 low, 3 high): " + "; ".join(f"{k} {v}" for k, v in c["labels_corrected"].items())]
    nl = c.get("null_hand")
    if nl:
        L += ["", f"**No-signal reference** ({nl.get('feature_set', 'hand-made features')}, the whole leave-one-tumour-out procedure re-run with the "
                  f"high/low labels randomly re-assigned, {nl['n']} times): tumour AUC mean {nl['tumour_auc_mean']} "
                  f"(95th percentile {nl['tumour_auc_p95']}), cell AUC mean {nl['cell_auc_mean']} (95th percentile "
                  f"{nl['cell_auc_p95']}). A real result has to beat these, not just 0.5."]
    if nl:
        for s, m in M.items():
            beat = m["tumour_auc"] > nl["tumour_auc_p95"] and m["cell_auc"] > nl["cell_auc_p95"]
            L.append(f"\n**Verdict {s}:** " + ("beats the no-signal reference (tumour and cell AUC above its 95th percentile)."
                     if beat else "does NOT beat the no-signal reference: no evidence of a real signal."))
    L += ["", "How to read it: with 7 tumours the tumour AUC moves in steps of 1/12, so call one model better only if it",
          "wins clearly on the per-tumour scores and the same-day AUC, not on a small AUC difference. Each model's",
          "folder has the full report, SHAP and the scores of the groups never used in training.", "",
          "_Research score of how much a cell behaves like cells from highly metastatic tumours; not a diagnosis._"]
    path.write_text("\n".join(L) + "\n")


def write_md(r, path):
    L = r["leave_one_tumour_out"]
    lines = ["# High vs low metastatic efficiency (idr0109) — leave-one-tumour-out", "",
             f"**Tumour-level AUC {L['tumour_level_auc']}** (exact permutation p = {L['tumour_level_p_exact']}, "
             f"{L['n_label_permutations']} label orders; with 7 tumours the smallest possible p is "
             f"{1 / L['n_label_permutations']:.3f}).  Cell-level pooled AUC {L['cell_level_auc_pooled']} "
             "(optimistic: cells of one tumour are not independent).", "",
             "| tumour | label | cells | mean score | share of cells > 0.5 |", "|---|---|---|---|---|"]
    lines += [f"| {t['cellType']} | {t['label']} | {t['cells']} | {t['mean_score']} | {t['frac_above_half']} |" for t in L["per_tumour"]]
    s = r["same_day_check"]
    lines += ["", f"**Same-day check** (days with both high and low tumours): median AUC {s['median_auc']}", ""]
    lines += [f"- {d['day']}: {', '.join(d['tumours'])} — AUC {d['auc']} ({d['cells']} cells)" for d in s["days"]]
    if "shortcut_check_day" in r:
        c = r["shortcut_check_day"]
        lines += ["", f"**Shortcut check:** the features predict the recording day with accuracy {c['accuracy']} "
                      f"(chance {c['chance']}, {c['n_days']} days). High accuracy means strong day effects: trust the same-day check more."]
    if "untouched_groups" in r:
        lines += ["", "**Never used in training** (model trained on all 7 tumours):", "", "| group | cells | n | mean score |", "|---|---|---|---|"]
        lines += [f"| {g['group']} | {g['cellType']} | {g['size']} | {g['mean']} |" for g in r["untouched_groups"]]
    if "shap_by_group_percent" in r:
        lines += ["", "**What drives the score (SHAP, % of total):** " +
                  ", ".join(f"{k} {v}%" for k, v in r["shap_by_group_percent"].items())]
    lines += ["", "_Research score of how much a cell behaves like cells from highly metastatic tumours; not a diagnosis._"]
    path.write_text("\n".join(lines) + "\n")


def plot(per_t, rest, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2 if rest is not None else 1, figsize=(12, 4), squeeze=False)
    p = per_t.sort_values("mean_score")
    col = ["#d1495b" if l == "high" else "#2e86ab" for l in p["label"]]
    ax[0, 0].barh(p.index, p["mean_score"], color=col)
    ax[0, 0].axvline(0.5, color="gray", lw=0.8, ls="--")
    ax[0, 0].set_xlim(0, 1)
    ax[0, 0].set_title("Held-out tumours (red = high, blue = low)", fontsize=10)
    ax[0, 0].set_xlabel("mean score of its cells (model never saw this tumour)")
    if rest is not None:
        q = rest.groupby("cellType")["score"].mean().sort_values()
        ax[0, 1].barh(q.index, q.values, color="#8b6cf0")
        ax[0, 1].axvline(0.5, color="gray", lw=0.8, ls="--")
        ax[0, 1].set_xlim(0, 1)
        ax[0, 1].set_title("Never used in training", fontsize=10)
        ax[0, 1].set_xlabel("mean score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


import re  # noqa: E402  (used in main)

if __name__ == "__main__":
    main()
