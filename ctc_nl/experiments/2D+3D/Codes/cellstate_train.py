#!/usr/bin/env python
"""CelloSyncra cell-state classifier - step 2: train and evaluate on the combined dataset (ALFI + QPI + CTC).

Hierarchical model (each part trained only on the videos that label it - nothing is treated as a negative just
because a dataset did not annotate it):

    life         alive | dead                          ALFI, QPI, CTC
    alive state  interphase | mitosis | abnormal_division   ALFI, CTC (+ weak interphase)
    death type   apoptosis | necrosis                  QPI
    final        P(interphase)=P(alive)P(int|alive) ... P(apoptosis)=P(dead)P(apo|dead)  (5 classes)

Each part = gradient boosting on: measurements (shape, size, image-normalised brightness and texture), their
mean / std / change over +-3 frames, and 64 principal components of the Cellpose-SAM encoder features (PCA fitted
on the training videos only). Research flags of cell behaviour - never a diagnosis.

Evaluation units are whole VIDEOS (never cells or frames):
    --mode cv      5-fold cross-validation on the development videos (locked test videos excluded)
    --mode test    train on all development videos, evaluate ONCE on the locked test videos
    --mode lodo    leave one dataset out        (train on 2 of ALFI / QPI / CTC, test on the 3rd)
    --mode lolo    leave one cell line out
    --mode loto    leave one QPI treatment out  (staurosporine / doxorubicin / black phosphorus)
    --mode final   train on everything and save the model for the website (no evaluation)

Locked test set (written to <data>/test_videos.txt the first time, then reused): QPI field of view 7 of every
line x treatment, CTC sequence 02, and 1 CD + 1 MI + 2 TP ALFI videos chosen with a fixed seed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["interphase", "mitosis", "abnormal_division", "apoptosis", "necrosis"]
ALIVE = ["interphase", "mitosis", "abnormal_division"]
DEAD = ["apoptosis", "necrosis"]
LIFE = ["alive", "dead"]
# labels.csv state -> (life, alive state, death type)
MAP = {"interphase": ("alive", "interphase", None), "weak_interphase": ("alive", "interphase", None),
       "mitosis": ("alive", "mitosis", None), "abnormal_division": ("alive", "abnormal_division", None),
       "alive": ("alive", None, None), "death": ("dead", None, None),
       "apoptosis": ("dead", None, "apoptosis"), "necrosis": ("dead", None, "necrosis"),
       "transition": (None, None, None)}
PRIORITY = {"abnormal_division": 6, "apoptosis": 5, "necrosis": 5, "death": 4, "mitosis": 3, "interphase": 2,
            "alive": 1, "weak_interphase": 0, "transition": -1}
MEAS = ["area", "perimeter", "eccentricity", "solidity", "circularity", "aspect_ratio", "intensity_img", "texture",
        "texture_img"]
MODALITIES = ["dic", "qpi", "phase"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ============================================================================ data
EVROWS: list = []


def make_tracks(obj: pd.DataFrame, emb: np.ndarray, info: dict) -> dict:
    """per-cell histories used by the features: {track_id: frames, minutes, measurements, encoder features,
    motion / dry-mass proxy, QPI physical values}. Shared by training and by the web app (cs_states.py)."""
    m = obj[MEAS].astype(np.float32).copy()
    m["area"], m["perimeter"] = np.log1p(m["area"]), np.log1p(m["perimeter"])
    obj = obj.reset_index(drop=True)
    tracks = {}
    for tid, g in obj.groupby("track_id"):
        g = g.sort_values("frame")
        idx = g.index.to_numpy()
        fr = g.frame.to_numpy()
        mins = fr * float(info.get("minutes_per_frame") or 1.0)
        # motion: displacement per minute, in cell diameters (dead cells stop moving)
        disp = np.hypot(np.diff(g.cy.to_numpy()), np.diff(g.cx.to_numpy()))
        diam = 2 * np.sqrt(g.area.to_numpy() / np.pi) + 1e-6
        speed = np.concatenate([[np.nan], disp / np.maximum(np.diff(mins), 1e-6) / diam[1:]])
        mass = np.log1p(g.area.to_numpy() * np.clip(g.intensity_img.to_numpy(), 0, None))   # dry-mass proxy
        # v7, QPI only: the phase value IS physical - mean = cell dry-mass density (pg/um2), area x mean = total dry
        # mass. These are the published QPI markers of death (density rises in apoptosis, mass is lost in necrosis).
        if info.get("modality") == "qpi":
            cdmd = g.mean_intensity.to_numpy()
            phys = np.stack([cdmd, np.log1p(np.clip(g.area.to_numpy() * cdmd, 0, None))], 1)
        else:
            phys = np.full((len(g), 2), np.nan)       # missing for DIC / phase: gradient boosting handles NaN
        tracks[int(tid)] = dict(frames=fr, mins=mins, meas=m.loc[idx].to_numpy(np.float32), emb=emb[idx],
                                dyn=np.stack([speed, mass], 1).astype(np.float32), phys=phys.astype(np.float32))
    return tracks


def load_video(d: Path, a, rng):
    info = json.loads((d / "info.json").read_text())
    obj = pd.read_csv(d / "objects.csv")
    emb = np.load(d / "emb.npy").astype(np.float32)
    obj = obj.reset_index(drop=True)
    tracks = make_tracks(obj, emb, info)
    lab = pd.read_csv(d / "labels.csv")
    lab = lab[lab.track_id > 0].copy()
    rows = []
    if len(lab):
        lab["pri"] = lab.state.map(PRIORITY)
        lab = lab.sort_values("pri").drop_duplicates(["frame", "track_id"], keep="last")
        lab = lab[[int(t) in tracks and f in set(tracks[int(t)]["frames"]) for t, f in zip(lab.track_id, lab.frame)]]
        # whole cell histories for the time-of-death analysis (uncapped): QPI cells that died (with the expert
        # death frame) and cells known to survive (ALFI MI, CTC) as negatives
        if info["dataset"] == "qpi" and "death_frame" in lab:
            ev = lab[lab.state.isin(["alive", "transition", "apoptosis", "necrosis"])]
        elif info["dataset"] == "ctc" or info["video"].startswith("ALFI_MI"):
            ev = lab[lab.state.isin(["interphase", "mitosis", "weak_interphase"])]
        else:
            ev = lab.iloc[:0]
        for r in ev.itertuples():
            EVROWS.append(dict(video=info["video"], event=str(r.gt_id), track=int(r.track_id), frame=int(r.frame),
                               death_frame=float(getattr(r, "death_frame", np.nan)) if info["dataset"] == "qpi" else np.nan,
                               minutes_per_frame=float(info.get("minutes_per_frame") or 1.0),
                               qpi_step=max(1, round(float(info.get("minutes_per_frame") or 3) / 3)),
                               dataset=info["dataset"], modality=info["modality"]))
        lab = lab[lab.state != "transition"]
        # at most `cap` frames per cell and state (long QPI 'alive' periods would otherwise dominate)
        lab = lab.sample(frac=1, random_state=int(rng.integers(1 << 30))).groupby(["gt_id", "state"]).head(a.cap)
        for r in lab.itertuples():
            rows.append(dict(video=info["video"], track=int(r.track_id), frame=int(r.frame), state=r.state,
                             event=str(r.gt_id), w=0.1 if r.state == "weak_interphase" else 1.0))
    # ALFI CD/TP: never-labelled tracks -> weak interphase (they label only event cells)
    if info["dataset"] == "alfi" and not info["video"].startswith("ALFI_MI"):
        done = set(lab.track_id.astype(int)) if len(lab) else set()
        for tid, T in tracks.items():
            if tid in done or len(T["frames"]) < 5:
                continue
            for f in rng.choice(T["frames"], size=3, replace=False):
                rows.append(dict(video=info["video"], track=tid, frame=int(f), state="weak_interphase", event="", w=0.1))
    return info, tracks, rows


def load_all(data: Path, a):
    rng = np.random.default_rng(a.seed)
    INFO, T, S = {}, {}, []
    for d in sorted((data / "seqs").iterdir()):
        if (d / "info.json").exists():
            info, tracks, rows = load_video(d, a, rng)
            INFO[info["video"]], T[info["video"]] = info, tracks
            S += rows
    S = pd.DataFrame(S)
    S = S[S.state != "transition"].reset_index(drop=True)
    for k, col in enumerate(["life", "alive_state", "death_type"]):
        S[col] = S.state.map(lambda s: MAP[s][k])
    meta = pd.DataFrame(INFO.values()).set_index("video")
    S = S.join(meta[["dataset", "cell_line", "treatment", "modality"]], on="video")
    log(f"{len(INFO)} videos ({meta.dataset.value_counts().to_dict()}), {len(S)} labelled cell-frames")
    log("cell-frames per state and dataset:\n" + pd.crosstab(S.state, S.dataset).to_string())
    return INFO, T, S


def locked_test(data: Path, INFO, seed=0):
    f = data / "test_videos.txt"
    if f.exists():
        return set(f.read_text().split())
    v = pd.Series(list(INFO))
    test = set(v[v.str.match(r"QPI_.*_7$")]) | set(v[v.str.match(r"CTC_.*_02$")])
    rng = np.random.default_rng(seed)
    for fam, n in (("CD", 1), ("MI", 1), ("TP", 2)):
        pool = sorted(x for x in v if x.startswith(f"ALFI_{fam}") and x not in ("ALFI_TP11", "ALFI_TP12"))
        test |= set(rng.choice(pool, size=min(n, len(pool)), replace=False))
    f.write_text("\n".join(sorted(test)) + "\n")
    log(f"locked test set written: {f} ({len(test)} videos) - never used for development")
    return test


# ============================================================================ features
BASE = ["area", "intensity_img", "texture", "circularity", "solidity"]   # compared with the cell's own past
DYN = ["speed", "mass"]


def features(T, S, k=3, win_min=30.0, past_min=(120.0, 30.0)):
    """per frame:
    * measurements now, their mean / std / change over +-k frames (as before)
    * v5: time in MINUTES - motion and dry-mass proxy averaged over +-win_min, and each key measurement relative to
      the SAME cell's own median 2 h to 30 min earlier (death and division are changes of a cell, not absolute
      values: a dying cell shrinks / brightens / stops moving compared with itself)."""
    X = []
    bi = [MEAS.index(b) for b in BASE]
    for r in S.itertuples():
        tr = T[r.video][r.track]
        i = int(np.searchsorted(tr["frames"], r.frame))
        a_, b_ = max(0, i - k), min(len(tr["frames"]), i + k + 1)
        w = tr["meas"][a_:b_]
        slope = tr["meas"][min(i + 1, b_ - 1)] - tr["meas"][max(i - 1, a_)]
        mod = [float(r.modality == m) for m in MODALITIES]
        t = tr["mins"][i]
        win = (tr["mins"] >= t - win_min) & (tr["mins"] <= t + win_min)
        dyn = tr["dyn"][win]
        dyn_mean = np.nanmean(dyn, 0) if np.isfinite(dyn).any() else np.zeros(len(DYN))
        dyn_mean = np.nan_to_num(dyn_mean)
        past = (tr["mins"] >= t - past_min[0]) & (tr["mins"] <= t - past_min[1])
        if past.any():
            rel = tr["meas"][i, bi] - np.median(tr["meas"][past][:, bi], 0)
            mrel = tr["dyn"][i, 1] - np.median(tr["dyn"][past, 1])
            has = 1.0
        else:
            rel, mrel, has = np.zeros(len(BASE)), 0.0, 0.0
        P_ = tr["phys"]
        ph = P_[i]
        ph_rel = ph - np.median(P_[past], 0) if past.any() else np.full(2, np.nan)
        ph_slope = P_[min(i + 1, b_ - 1)] - P_[max(i - 1, a_)]
        # longer memory: mass compared with the most the cell ever had so far (necrotic bursting = big drop),
        # density compared with the cell's first 2 h in the video (apoptotic condensation = rise)
        so_far = tr["mins"] <= t
        first = tr["mins"] <= tr["mins"][0] + 120
        ph_long = np.array([ph[1] - np.nanmax(P_[so_far, 1]), ph[0] - np.nanmedian(P_[first, 0])]) \
            if np.isfinite(ph).all() else np.full(2, np.nan)
        X.append(np.concatenate([tr["meas"][i], w.mean(0), w.std(0), slope, mod, dyn_mean, rel, [mrel, has],
                                 ph, ph_rel, ph_slope, ph_long]))
    names = MEAS + [f"{m}_mean" for m in MEAS] + [f"{m}_std" for m in MEAS] + [f"{m}_change" for m in MEAS] + \
        [f"is_{m}" for m in MODALITIES] + [f"{d}_{int(win_min)}min" for d in DYN] + \
        [f"{b}_vs_own_past" for b in BASE] + ["mass_vs_own_past", "has_past"] + \
        ["qpi_density", "qpi_dry_mass", "qpi_density_vs_past", "qpi_dry_mass_vs_past", "qpi_density_change",
         "qpi_dry_mass_change", "qpi_dry_mass_vs_own_max", "qpi_density_vs_own_start"]
    return np.array(X, np.float32), names


def embeddings(T, S):
    return np.stack([T[r.video][r.track]["emb"][int(np.searchsorted(T[r.video][r.track]["frames"], r.frame))]
                     for r in S.itertuples()])


class Head:
    """one gradient-boosting classifier for one level of the hierarchy (class-balanced weights)."""

    def __init__(self, col, classes, seed):
        self.col, self.classes, self.seed, self.m = col, classes, seed, None

    def fit(self, X, S):
        from sklearn.ensemble import HistGradientBoostingClassifier
        keep = S[self.col].notna().to_numpy()
        y = S[self.col][keep].map({c: i for i, c in enumerate(self.classes)}).to_numpy()
        if len(set(y)) < 2:
            log(f"  {self.col}: only one class in training, skipped")
            return self
        w = S.w[keep].to_numpy()
        cnt = np.bincount(y, weights=w, minlength=len(self.classes))
        cw = np.where(cnt > 0, cnt.sum() / (len(self.classes) * np.maximum(cnt, 1e-9)), 0)
        self.m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                l2_regularization=1.0, early_stopping=True,
                                                validation_fraction=0.15, random_state=self.seed)
        self.cols = ~np.isnan(X[keep]).all(0)     # e.g. QPI-only features when this part sees no QPI videos
        self.m.fit(X[keep][:, self.cols], y, sample_weight=w * cw[y])
        return self

    def proba(self, X):
        P = np.full((len(X), len(self.classes)), np.nan)   # untrained part (e.g. its only dataset held out)
        if self.m is not None:
            P[:] = 1e-6
            P[:, self.m.classes_] = self.m.predict_proba(X[:, self.cols])
            P /= P.sum(1, keepdims=True)
        return P


class Model:
    def __init__(self, a):
        self.a = a

    def _X(self, T, S, fit=False):
        X, self.names = features(T, S)
        if self.a.pcs:
            from sklearn.decomposition import PCA
            E = embeddings(T, S)
            if fit:
                self.pca = PCA(self.a.pcs, random_state=self.a.seed).fit(E)
            X = np.hstack([X, self.pca.transform(E)])
            self.names = self.names + [f"enc_pc{i}" for i in range(self.a.pcs)]
        return X

    def fit(self, T, S):
        X = self._X(T, S, fit=True)
        self.heads = [Head("life", LIFE, self.a.seed).fit(X, S), Head("alive_state", ALIVE, self.a.seed).fit(X, S),
                      Head("death_type", DEAD, self.a.seed).fit(X, S)]
        return self

    def predict(self, T, S):
        X = self._X(T, S)
        L, A, D = (h.proba(X) for h in self.heads)
        final = np.hstack([L[:, :1] * A, L[:, 1:] * D])
        return dict(life=L, alive_state=A, death_type=D, final=final)


# ============================================================================ metrics
def head_metrics(y_true, P, classes, videos, n_boot, seed):
    from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support
    keep = pd.notna(y_true) & ~np.isnan(P).any(1)
    if keep.sum() == 0:
        return None
    y = pd.Series(y_true[keep]).map({c: i for i, c in enumerate(classes)}).to_numpy()
    P, vids = P[keep], np.asarray(videos)[keep]
    p = P.argmax(1)
    present = sorted(set(y))
    pr, rc, f1, n = precision_recall_fscore_support(y, p, labels=range(len(classes)), zero_division=0)
    res = dict(macro_f1=round(float(f1_score(y, p, labels=present, average="macro", zero_division=0)), 3), per_class={},
               confusion=dict(labels=classes, matrix=confusion_matrix(y, p, labels=range(len(classes))).tolist()))
    for k, c in enumerate(classes):
        if (y == k).any():
            res["per_class"][c] = dict(precision=round(pr[k], 3), recall=round(rc[k], 3), f1=round(f1[k], 3),
                                       n=int(n[k]), auprc=round(float(average_precision_score(y == k, P[:, k])), 3),
                                       prevalence=round(float((y == k).mean()), 4))
    rng = np.random.default_rng(seed)
    uv = np.unique(vids)
    rows = {v: np.flatnonzero(vids == v) for v in uv}
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[v] for v in rng.choice(uv, len(uv), replace=True)])
        boot.append(f1_score(y[idx], p[idx], labels=present, average="macro", zero_division=0))
    res["macro_f1_ci95"] = ([round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]
                            if boot else ["-", "-"])
    return res


def evaluate(S, PR, a):
    S = S.reset_index(drop=True)
    out = {}
    for col, classes in (("life", LIFE), ("alive_state", ALIVE), ("death_type", DEAD)):
        out[col] = head_metrics(S[col].to_numpy(dtype=object), PR[col], classes, S.video, a.boot, a.seed)
    # final 5-class, on frames whose fine class is known
    fine = S.apply(lambda r: r.alive_state if r.life == "alive" else r.death_type, axis=1)
    out["final"] = head_metrics(fine.to_numpy(dtype=object), PR["final"], CLASSES, S.video, a.boot, a.seed)
    # per dataset (life and the fine heads)
    out["per_dataset"] = {}
    for ds in sorted(S.dataset.unique()):
        m = (S.dataset == ds).to_numpy()
        out["per_dataset"][ds] = {col: (head_metrics(S[col].to_numpy(dtype=object)[m], PR[col][m], cl, S.video[m], 0, 0)
                                        or {}).get("macro_f1")
                                  for col, cl in (("life", LIFE), ("alive_state", ALIVE), ("death_type", DEAD))}
    # event level: an expert event (one cell, one state) is caught if any of its frames gets that fine class
    ev = S[(S.event != "") & S.state.isin(["mitosis", "abnormal_division", "apoptosis", "necrosis", "death"])].copy()
    if len(ev):
        ok = ~np.isnan(PR["final"]).any(1)
        ev = ev[ok[ev.index] | (ev.state == "death")]
        pred_fine = np.array(CLASSES)[np.nan_to_num(PR["final"], nan=-1).argmax(1)]
        dead_pred = PR["life"][:, 1] > 0.5
        ev["hit"] = [(dead_pred[i] if s == "death" else pred_fine[i] == s) for i, s in zip(ev.index, ev.state)]
        out["event_recall"] = {s: dict(recall=round(float(g.groupby("event").hit.any().mean()), 3),
                                       events=int(g.event.nunique()))
                               for s, g in ev.groupby("state")}
    return out


# ============================================================================ runs
def make_cv_folds(S_videos: pd.DataFrame, k):
    v = S_videos.sort_values(["dataset", "cell_line", "treatment", "video"])
    return {vid: i % k for i, vid in enumerate(v.video)}


TAUS_DEATH = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)
DEATH_TAU = 5.0   # operating point FIXED on Nov 12 from the v7 cross-validation, BEFORE the locked test was run   # detection vs false-alarm trade-off; pick from CV, confirm on test


def death_events(T, model, EV):
    """per cell history: P(dead) on every frame, then ONE irreversible switch alive -> dead (or none), chosen by
    maximum likelihood with a penalty tau (the switch must gain > tau in log-likelihood). Returns one row per cell:
    did it die (truth), was a death detected, predicted vs expert time of death (minutes)."""
    if not len(EV):
        return pd.DataFrame()
    EV = EV.sort_values(["event", "frame"]).reset_index(drop=True)
    pdead = model.predict(T, EV.assign(w=1.0))["life"][:, 1]
    rows = []
    EVp = EV.assign(p=pdead)
    # within-QPI negatives: the part of each dying cell's history that is still clearly alive (> 1 h before the
    # expert death frame). A death detected there is a false alarm on a living QPI cell, in the same videos.
    qa = EVp[(EVp.dataset == "qpi") & (EVp.frame < EVp.death_frame - 20)].copy()
    qa = qa.groupby("event").filter(lambda g: len(g) >= 5)
    qa["event"], qa["death_frame"], qa["dataset"] = qa.event + ":alive_part", np.nan, "qpi_alive_part"
    for e, g in pd.concat([EVp, qa]).groupby("event", sort=False):
        lo = np.log(np.clip(g.p.to_numpy(), 1e-4, 1 - 1e-4))
        la = np.log(np.clip(1 - g.p.to_numpy(), 1e-4, 1 - 1e-4))
        gain = np.concatenate([[0.0], np.cumsum((lo - la)[::-1])])[::-1]   # gain[c] = dead from index c on
        c = int(np.argmax(gain[:-1])) if len(g) else 0
        best = gain[c]
        died = bool(np.isfinite(g.death_frame.iloc[0]))
        mins = g.frame.to_numpy() * 3.0 if g.dataset.iloc[0].startswith("qpi") else g.frame.to_numpy() * g.minutes_per_frame.iloc[0]
        for tau in TAUS_DEATH:
            det = best > tau
            rows.append(dict(event=e, video=g.video.iloc[0], dataset=g.dataset.iloc[0], tau=tau, died=died,
                             detected=det, pred_min=float(mins[c]) if det else np.nan,
                             true_min=float(g.death_frame.iloc[0] * 3.0) if died else np.nan))
    return pd.DataFrame(rows)


def death_summary(D):
    out = {}
    for tau, g in D.groupby("tau"):
        pos, neg = g[g.died], g[~g.died & (g.dataset != "qpi_alive_part")]
        qneg = g[g.dataset == "qpi_alive_part"]
        err = (pos.pred_min - pos.true_min).dropna()
        out[f"tau={tau}"] = dict(
            cells_died=len(pos), detected=round(float(pos.detected.mean()), 3) if len(pos) else None,
            cells_survived=len(neg), false_alarm=round(float(neg.detected.mean()), 3) if len(neg) else None,
            qpi_alive_histories=len(qneg),
            false_alarm_qpi_alive=round(float(qneg.detected.mean()), 3) if len(qneg) else None,
            early_over_3h=round(float((err < -180).mean()), 3) if len(err) else None,
            median_abs_error_min=round(float(err.abs().median()), 1) if len(err) else None,
            median_signed_error_min=round(float(err.median()), 1) if len(err) else None,
            within_60min=round(float((err.abs() <= 60).mean()), 3) if len(err) else None,
            within_180min=round(float((err.abs() <= 180).mean()), 3) if len(err) else None)
    return out


def run_split(T, S, train_v, test_v, a, EV=None):
    tr, te = S[S.video.isin(train_v)], S[S.video.isin(test_v) & (S.w >= 1)]
    if not len(te):
        return None, None, None
    model = Model(a).fit(T, tr)
    D = death_events(T, model, EV[EV.video.isin(test_v)]) if EV is not None and len(EV) else None
    return te, model.predict(T, te), D


def stack(parts):
    S = pd.concat([p[0] for p in parts])
    PR = {k: np.concatenate([p[1][k] for p in parts]) for k in parts[0][1]}
    return S, PR


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="build output folder (contains seqs/)")
    ap.add_argument("--mode", choices=["cv", "test", "lodo", "lolo", "loto", "final"], default="cv")
    ap.add_argument("--tag", default="v7")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pcs", type=int, default=64, help="encoder principal components (0 = measurements only)")
    ap.add_argument("--cap", type=int, default=20, help="max labelled frames per cell and state")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    data = Path(a.data)
    INFO, T, S = load_all(data, a)
    test = locked_test(data, INFO, a.seed)
    meta = pd.DataFrame(INFO.values())
    dev = meta[~meta.video.isin(test)]
    out = data / "train" / a.tag / a.mode
    out.mkdir(parents=True, exist_ok=True)
    parts, splits = [], []
    if a.mode == "final":
        model = Model(a).fit(T, S)
        import pickle
        (out / "model.pkl").write_bytes(pickle.dumps(model))
        (out / "model_card.json").write_text(json.dumps(dict(
            trained_on=sorted(INFO), classes=CLASSES, hidden=["abnormal_division"], features=model.names,
            datasets=meta.dataset.value_counts().to_dict(), cell_lines=sorted(meta.cell_line.unique()),
            death_tau=DEATH_TAU,
            death_output="per cell: onset of signs of dying (not a confirmed time of death; QPI cells decline hours "
                         "before the fluorescence death marker)",
            intended_use="research flags of cell behaviour in label-free time-lapse microscopy; not a diagnosis"),
            indent=1))
        log(f"final model saved: {out / 'model.pkl'}")
        return
    if a.mode == "cv":
        folds = make_cv_folds(dev, a.folds)
        for k in range(a.folds):
            te = [v for v, f in folds.items() if f == k]
            splits.append((f"fold{k}", [v for v in folds if v not in te], te))
    elif a.mode == "test":
        splits.append(("locked_test", list(dev.video), sorted(test)))
    else:
        key = dict(lodo="dataset", lolo="cell_line", loto="treatment")[a.mode]
        pool = meta if a.mode != "loto" else meta[meta.dataset == "qpi"]
        for g in sorted(pool[key].unique()):
            te = list(pool[pool[key] == g].video)
            splits.append((f"{key}={g}", [v for v in meta.video if v not in te], te))
    summary = {}
    EV = pd.DataFrame(EVROWS)
    if len(EV):
        EV = EV.join(pd.DataFrame(INFO.values()).set_index("video")[["cell_line", "treatment"]], on="video")
    deaths = []
    for name, tr_v, te_v in splits:
        te, pr, D = run_split(T, S, tr_v, te_v, a, EV)
        if te is None:
            continue
        parts.append((te, pr))
        if D is not None and len(D):
            deaths.append(D)
        if a.mode in ("lodo", "lolo", "loto"):
            summary[name] = evaluate(te, pr, argparse.Namespace(boot=200, seed=a.seed))
        log(f"{name}: {len(te_v)} test videos, {len(te)} frames")
    Sx, PR = stack(parts)
    res = evaluate(Sx, PR, a)
    if deaths:
        Dx = pd.concat(deaths)
        Dx.to_csv(out / "death_events.csv", index=False)
        res["death_events"] = death_summary(Dx)
    res["splits"] = {n: dict(train=len(tr), test=te) for n, tr, te in splits}
    if summary:
        res["per_split"] = summary
    pred = Sx[["video", "track", "frame", "state", "event", "dataset", "cell_line", "treatment"]].copy()
    for k, cl in (("life", LIFE), ("final", CLASSES)):
        for i, c in enumerate(cl):
            pred[f"p_{c}"] = PR[k][:, i].round(4)
    pred.to_csv(out / "predictions.csv", index=False)
    q = pred[pred.dataset == "qpi"]
    if len(q):     # when does the model see death? P(dead) vs minutes from the expert death frame
        dead = q[q.state.isin(DEAD)]
        df_ = dead.groupby("event").frame.min()

        q = q[q.event.isin(df_.index)].copy()
        q["min"] = (q.frame - q.event.map(df_)) * 3.0
        bins = [-1e5, -600, -300, -120, -60, 0, 30, 60, 120, 181]
        q["bin"] = pd.cut(q["min"], bins, right=False)
        res["death_timing"] = {str(k): round(float(v), 3) for k, v in
                               q.groupby("bin", observed=True).p_dead.apply(lambda x: (x > 0.5).mean()).items()}
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=str))
    write_report(out, res, a)


def write_report(out, r, a):
    L = [f"# Cell-state classifier {a.tag} - {a.mode}", "",
         "Units: whole videos. Research flags of cell behaviour, not diagnoses.", "",
         "| part | macro-F1 [95% CI] | per class: F1 / AUPRC (n frames) |", "|---|---|---|"]
    for col in ("life", "alive_state", "death_type", "final"):
        h = r.get(col)
        if not h:
            continue
        pc = " · ".join(f"{c} {v['f1']} / {v['auprc']} ({v['n']})" for c, v in h["per_class"].items())
        L.append(f"| {col} | {h['macro_f1']} [{h['macro_f1_ci95'][0]}-{h['macro_f1_ci95'][1]}] | {pc} |")
    if "event_recall" in r:
        L += ["", "Event recall (expert event caught on >= 1 frame): " +
              " · ".join(f"{s} {v['recall']} ({v['events']})" for s, v in r["event_recall"].items())]
    L += ["", "| dataset | life | alive state | death type |", "|---|---|---|---|"]
    for ds, v in r["per_dataset"].items():
        L.append(f"| {ds} | {v['life']} | {v['alive_state']} | {v['death_type']} |")
    if "death_events" in r:
        L += ["", "## Cell death as an event (one irreversible switch per cell)", "",
              "| penalty | cells that died: detected | median time error (min) | within 60 / 180 min | "
              ">3 h early | false alarm: QPI cells while alive | false alarm: ALFI/CTC survivors |",
              "|---|---|---|---|---|---|---|"]
        for k, v in r["death_events"].items():
            k = f"**{k} (chosen)**" if k == f"tau={DEATH_TAU}" else k
            L.append(f"| {k} | {v['detected']} ({v['cells_died']}) | {v['median_abs_error_min']} "
                     f"(signed {v['median_signed_error_min']}) | {v['within_60min']} / {v['within_180min']} | "
                     f"{v['early_over_3h']} | {v['false_alarm_qpi_alive']} ({v['qpi_alive_histories']}) | "
                     f"{v['false_alarm']} ({v['cells_survived']}) |")
    if "death_timing" in r:
        L += ["", "QPI: share of frames called dead, by minutes from the expert death frame: " +
              " · ".join(f"{k} {v}" for k, v in r["death_timing"].items())]
    if "per_split" in r:
        L += ["", "| held out | life | alive state | death type | final |", "|---|---|---|---|---|"]
        for n, v in r["per_split"].items():
            L.append(f"| {n} | " + " | ".join(str((v.get(c) or {}).get("macro_f1", "-"))
                                               for c in ("life", "alive_state", "death_type", "final")) + " |")
    (out / "report.md").write_text("\n".join(L) + "\n")
    log(f"report: {out / 'report.md'}\n" + "\n".join(L))


if __name__ == "__main__":
    main()
