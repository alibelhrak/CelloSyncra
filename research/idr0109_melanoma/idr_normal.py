"""Normal vs abnormal cells in idr0109: melanoma cells (abnormal) vs normal melanocytes, evaluated honestly.

Abnormal = every melanoma group (PDX tumours, their clones, melanoma cell lines); normal = melanocytes (atcc, m116).
Unit: one tracked cell (tracks.csv.gz from idr_build.py, ViT features from idr_embed.py).

Rules fixed BEFORE the run (run once, do not tune on the result):
  model     hybrid = frozen Cellpose-SAM ViT features (PCA fitted inside each fold) + all hand-made features,
            one gradient-boosted tree classifier; 'hand' (hand-made only) is run beside it as the comparison.
  test      leave-one-family-out: each family (a group with its clones, e.g. m405 + m405c*) is held out in turn
            and scored by a model that never saw it. Both melanocyte groups are held out like any other group.
  weights   every family counts equally inside its class, both classes count equally.
  checks    group AUC (mean score per held-out family) with an exact permutation p; cell AUC with a
            family-bootstrap CI; same-day AUC (days holding both classes); how well the features predict the day.
  reference the whole procedure re-run with 'normal' given to 2 random families (20 times): what no signal gives.
  PASS      hybrid group AUC and cell AUC above the 95th percentile of the reference, AND both held-out
            melanocyte groups score below every melanoma family, AND the same-day AUC (only days that hold both
            melanocytes and melanoma cells) above the reference's 95th percentile of the same-day AUC.
            Why the last rule: the data check showed melanocytes share only 1 of their 6 recording days with
            melanoma cells, so without it the model could pass by recognising the days, not the cells.

Research use only: a score of how much a cell looks and moves like the idr0109 melanoma cells rather than the two
normal melanocyte lines. Not a diagnosis.

  python idr_normal.py --tracks out/features/tracks.csv.gz --embed out/embed/embeddings.npz --out out/normal
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import idr_train as T  # noqa: E402  (features, classifier, AUC helpers: same as the metastasis test)

NORMAL = ("atcc", "m116")


def family(ct):
    """clones are held out together with their parent (a clone of m405 must not be in training when m405 is tested)"""
    m = re.match(r"^(m405|a375)c", ct)
    return m.group(1) if m else ct


def fam_weights(d):
    n_f = d.groupby("family")["family"].transform("size")
    n_c = d.groupby("y")["family"].transform("nunique")
    w = 1.0 / (n_f * n_c)
    return (w / w.mean()).to_numpy()


def logo(F, df, d, seed):
    """leave-one-family-out scores (out of fold) for every cell of d"""
    oof = pd.Series(np.nan, index=d.index)
    for f in sorted(d["family"].unique()):
        tr, te = d.index[d["family"] != f], d.index[d["family"] == f]
        F.fit(d.loc[tr, "row"].to_numpy())
        m = T.model(seed, len(tr)).fit(F.X(df, d.loc[tr, "row"].to_numpy()), d.loc[tr, "y"], sample_weight=fam_weights(d.loc[tr]))
        oof[te] = m.predict_proba(F.X(df, d.loc[te, "row"].to_numpy()))[:, 1]
    return oof


def group_auc(d, col):
    per = d.groupby("family").agg(y=("y", "first"), s=(col, "mean"))
    return T.auc(per["y"], per["s"]), per


def same_day_auc(d, col):
    v = [T.auc(g["y"], g[col]) for _, g in d.groupby("date") if g["y"].nunique() == 2]
    return float(np.median(v)) if v else None


def null_ref(F, df, d, n, seed):
    rng = np.random.default_rng(seed + 1)
    fams = sorted(d["family"].unique())
    true_norm = set(d.loc[d["y"] == 0, "family"])
    combos = [set(c) for c in itertools.combinations(fams, len(true_norm)) if set(c) != true_norm]
    rng.shuffle(combos)
    res = []
    for nm in combos[:n]:
        L = d.copy()
        L["y"] = (~L["family"].isin(nm)).astype(int)
        L["oof"] = logo(F, df, L, seed)
        ga, _ = group_auc(L, "oof")
        res.append(dict(group_auc=ga, cell_auc=T.auc(L["y"], L["oof"]), same_day=same_day_auc(L, "oof")))
        print(f"  reference {len(res)}/{min(n, len(combos))}: group AUC {ga:.3f}", flush=True)
    ga = np.array([r["group_auc"] for r in res]); ca = np.array([r["cell_auc"] for r in res])
    sd = np.array([r["same_day"] for r in res if r["same_day"] is not None])
    return dict(n=len(res), group_auc_mean=round(float(ga.mean()), 3), group_auc_p95=round(float(np.percentile(ga, 95)), 3),
                cell_auc_mean=round(float(ca.mean()), 3), cell_auc_p95=round(float(np.percentile(ca, 95)), 3),
                same_day_mean=round(float(sd.mean()), 3) if len(sd) else None,
                same_day_p95=round(float(np.percentile(sd, 95)), 3) if len(sd) else None)


def fam_boot(d, col, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    by = {f: g for f, g in d.groupby("family")}
    ab = [f for f, g in by.items() if g["y"].iloc[0] == 1]
    no = [f for f, g in by.items() if g["y"].iloc[0] == 0]
    v = []
    for _ in range(n):
        g = pd.concat([by[f] for f in list(rng.choice(ab, len(ab))) + list(rng.choice(no, len(no)))])
        v.append(T.auc(g["y"], g[col]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def evaluate(name, F, df, d, out, seed):
    o = out / name
    o.mkdir(parents=True, exist_ok=True)
    d = d.copy()
    d["oof"] = logo(F, df, d, seed)
    ga, per = group_auc(d, "oof")
    obs, p, n_perm = T.exact_perm_p(per["s"].to_dict(), per["y"].to_dict())
    days = [dict(day=k, cells=len(g), normal_cells=int((g["y"] == 0).sum()), auc=round(T.auc(g["y"], g["oof"]), 3))
            for k, g in d.groupby("date") if g["y"].nunique() == 2]
    rep = dict(feature_set=T.SET_NAMES[name], group_auc=round(ga, 3), group_p_exact=round(p, 4), n_permutations=n_perm,
               cell_auc=round(T.auc(d["y"], d["oof"]), 3), cell_auc_ci=fam_boot(d, "oof", seed=seed),
               same_day=dict(days=days, median_auc=round(float(np.median([x["auc"] for x in days])), 3) if days else None),
               per_family=per.rename(columns={"y": "abnormal", "s": "mean_score"}).round(3).reset_index().to_dict("records"))
    norm_s = per.loc[per["y"] == 0, "s"]; ab_s = per.loc[per["y"] == 1, "s"]
    rep["normals_below_every_melanoma_family"] = bool(norm_s.max() < ab_s.min())
    # final model on everything + day shortcut + SHAP
    rows = d["row"].to_numpy()
    F.fit(rows)
    X = F.X(df, rows)
    feats = list(X.columns)
    rep["n_features"] = len(feats)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    vc = d["date"].value_counts()
    keep = d["date"].isin(vc[vc >= 10].index).to_numpy()
    if keep.sum() > 50 and d.loc[keep, "date"].nunique() > 2:
        acc = cross_val_score(HistGradientBoostingClassifier(max_iter=150, random_state=seed), X[keep], d.loc[keep, "date"],
                              cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean()
        rep["predicts_day"] = dict(accuracy=round(float(acc), 3), chance=round(float(vc[vc >= 10].max() / keep.sum()), 3))
    final = T.model(seed, len(d)).fit(X, d["y"], sample_weight=fam_weights(d))
    try:
        import shap
        Xs = X.sample(min(3000, len(X)), random_state=seed)
        sv = shap.TreeExplainer(final).shap_values(Xs)
        sv = sv[1] if isinstance(sv, list) else sv
        imp = pd.Series(np.abs(sv).mean(0), index=feats)
        rep["shap_top_features"] = imp.sort_values(ascending=False).head(15).round(4).to_dict()
        g = imp.groupby([T.group_of(c) for c in feats]).sum()
        rep["shap_by_group_percent"] = (100 * g / g.sum()).sort_values(ascending=False).round(1).to_dict()
    except Exception as e:
        rep["shap_error"] = f"{type(e).__name__}: {e}"
    import joblib
    joblib.dump(dict(model=final, features=feats, feature_set=name, task="abnormal (melanoma) vs normal melanocyte",
                     vit=None if name == "hand" else dict(scaler=F.sc, pca=F.pca)), o / "model.pkl")
    d[["track", "cellType", "family", "date", "y", "oof"]].to_csv(o / "scores_logo.csv", index=False)
    (o / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"[{name}] group AUC {rep['group_auc']} (p {rep['group_p_exact']}), cell AUC {rep['cell_auc']} {rep['cell_auc_ci']}, "
          f"same-day {rep['same_day']['median_auc']}, normals below every melanoma family: "
          f"{rep['normals_below_every_melanoma_family']}", flush=True)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--embed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="hybrid,hand")
    ap.add_argument("--min-frames", type=int, default=30)
    ap.add_argument("--n-pc", type=int, default=64)
    ap.add_argument("--null", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-only", action="store_true", help="print the cell counts and shared days, then stop")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.tracks, dtype={"date": str, "locationStr": str, "metEff": str})
    hand = [c for c in df.columns if c not in T.META and pd.api.types.is_numeric_dtype(df[c])]
    df = df[(df["n_frames"] >= a.min_frames) & (df["n_masked"] >= 3)]
    z = np.load(a.embed)
    pos = {t: i for i, t in enumerate(z["track"])}
    df = df[df["track"].isin(pos)].reset_index(drop=True)
    ix = df["track"].map(pos).to_numpy()
    emb = dict(mean=z["emb_mean"][ix], std=z["emb_std"][ix], step=z["emb_step"][ix])
    df["row"] = np.arange(len(df))
    df["family"] = df["cellType"].map(family)
    df["y"] = (~df["cellType"].isin(NORMAL)).astype(int)

    # ---- data check (counts, shared days): printed first, written to data_check.json
    nd = df[df["y"] == 0]
    shared = sorted(set(nd["date"]) & set(df.loc[df["y"] == 1, "date"]))
    chk = dict(cells_per_normal_group=nd.groupby("cellType").size().to_dict(),
               abnormal_cells=int(df["y"].sum()), abnormal_families=int(df.loc[df["y"] == 1, "family"].nunique()),
               normal_days=sorted(nd["date"].unique().tolist()), days_shared_with_melanoma=shared,
               normal_cells_on_shared_days=int(nd["date"].isin(shared).sum()),
               on_shared_days={k: g.groupby("cellType").size().to_dict() for k, g in df[df["date"].isin(shared)].groupby("date")})
    (out / "data_check.json").write_text(json.dumps(chk, indent=1, default=int))
    print("data check:", json.dumps(chk, default=int), flush=True)
    if len(chk["cells_per_normal_group"]) < 2:
        sys.exit("need both melanocyte groups (atcc, m116) in the data")
    if a.check_only:
        return

    sets = [s.strip() for s in a.sets.split(",") if s.strip()]
    reps = {s: evaluate(s, T.Features(s, hand, emb, a.n_pc), df, df, out, a.seed) for s in sets}
    ref = null_ref(T.Features(sets[0], hand, emb, a.n_pc), df, df, a.null, a.seed) if a.null else None
    main_set = sets[0]
    r = reps[main_set]
    sd = r["same_day"]["median_auc"]
    passed = bool(ref and r["group_auc"] > ref["group_auc_p95"] and r["cell_auc"] > ref["cell_auc_p95"]
                  and r["normals_below_every_melanoma_family"]
                  and sd is not None and ref.get("same_day_p95") is not None and sd > ref["same_day_p95"])
    (out / "summary.json").write_text(json.dumps(dict(data_check=chk, reference=ref, passed=passed,
                                                      models={s: {k: v for k, v in x.items() if k != "per_family"} for s, x in reps.items()}),
                                                 indent=1, default=float))
    L = ["# Normal vs abnormal (melanoma vs melanocyte), idr0109, leave-one-family-out", "",
         f"Normal cells: {chk['cells_per_normal_group']}; abnormal: {chk['abnormal_cells']:,} cells in "
         f"{chk['abnormal_families']} families; normal cells on days shared with melanoma: {chk['normal_cells_on_shared_days']:,} "
         f"({len(shared)} days)", "",
         "| model | features | group AUC (exact p) | cell AUC [95% CI, family bootstrap] | same-day AUC | predicts day | melanocytes below every melanoma family |",
         "|---|---|---|---|---|---|---|"]
    for s, x in reps.items():
        L.append(f"| {x['feature_set']} | {x['n_features']} | {x['group_auc']} ({x['group_p_exact']}) | {x['cell_auc']} {x['cell_auc_ci']} | "
                 f"{x['same_day']['median_auc']} | {x.get('predicts_day', {}).get('accuracy')} | {x['normals_below_every_melanoma_family']} |")
    L += ["", "Mean score of each held-out family (1 = like melanoma; the model never saw it):", "",
          "| family | class | " + " | ".join(reps) + " |", "|---|---|" + "---|" * len(reps)]
    pf = {s: {p["family"]: p["mean_score"] for p in x["per_family"]} for s, x in reps.items()}
    cls = {p["family"]: ("abnormal" if p["abnormal"] else "**normal**") for p in r["per_family"]}
    for f in sorted(cls, key=lambda f: pf[main_set][f]):
        L.append(f"| {f} | {cls[f]} | " + " | ".join(str(pf[s][f]) for s in reps) + " |")
    if ref:
        L += ["", f"**No-signal reference** ({T.SET_NAMES[main_set]}, the whole procedure re-run with 'normal' given to 2 random "
                  f"families, {ref['n']} times): group AUC mean {ref['group_auc_mean']} (95th percentile {ref['group_auc_p95']}), "
                  f"cell AUC mean {ref['cell_auc_mean']} (95th percentile {ref['cell_auc_p95']}), same-day AUC mean "
                  f"{ref.get('same_day_mean')} (95th percentile {ref.get('same_day_p95')})."]
    sdays = r["same_day"]["days"]
    if sdays:
        L += ["", "Days with both melanocytes and melanoma cells (the fairest comparison): " +
              "; ".join(f"{x['day']}: {x['normal_cells']} melanocyte + {x['cells'] - x['normal_cells']} melanoma cells, AUC {x['auc']}" for x in sdays)]
    L += ["", f"**Verdict ({main_set}):** " + ("PASS: beats the no-signal reference and both held-out melanocyte groups score "
          "below every melanoma family, also on the shared day(s)." if passed else "does NOT pass the rule fixed before the run."), "",
          "_Research score: how much a cell looks and moves like the idr0109 melanoma cells rather than the two normal "
          "melanocyte lines. Not a diagnosis._"]
    (out / "comparison.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()