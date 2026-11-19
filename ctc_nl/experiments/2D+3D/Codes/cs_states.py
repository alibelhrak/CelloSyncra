"""CelloSyncra cell states for the web app: runs the trained cell-state classifier (cellstate_train.py v7,
model.pkl) on an uploaded video that is already segmented and tracked.

Same features as training (cellstate_build.measurements + Cellpose-SAM encoder features averaged under each
cell + cellstate_train.make_tracks/features), then per cell per frame:
    P(alive / dead), P(interphase / mitosis / abnormal division | alive), P(apoptosis / necrosis | dead)
and per cell: the onset of signs of dying (one irreversible switch, tau fixed before the locked test).

What may be shown depends on what was validated (leave-one-dataset-out results, Nov 12):
    normal / dividing        DIC and phase-contrast videos (ALFI, CTC); NOT validated for QPI
    dying, apoptosis/necrosis QPI videos only (death does not transfer between microscope types)
    abnormal division        never shown (30 training examples, chance level)
plus a domain check: encoder features far from the training data -> nothing shown.

Research flags of cell behaviour. Never a diagnosis; never "cancer".
"""
from __future__ import annotations

import io
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# one copy of the classifier code: the CellState folder that trained the model (set by run_hpc.sh);
# copies next to this file are used only if that folder is not given
_CS = os.environ.get("CELLSTATE_DIR", "")
sys.path.insert(0, str(HERE))
if _CS and Path(_CS, "cellstate_train.py").is_file():
    sys.path.insert(0, _CS)
import cellstate_build as B   # noqa: E402
import cellstate_train as TR  # noqa: E402

DEFAULT_MIN_PER_FRAME = 5.0        # uploads carry no time stamps; ALFI 5-7 min, QPI 6 min as trained
OOD_FACTOR = 3.0
MIN_FOLLOW_MIN = 120.0             # a cell must be followed >= 2 h to be judged (short tracks = broken / new pieces)
MIN_ALIVE_MIN = 30.0               # and seen >= 30 min before its onset (a track that starts "dead" is a fragment)                   # median Mahalanobis^2 in encoder-PCA space > 3 x dims -> unlike training


class _Unpickler(pickle.Unpickler):
    """model.pkl was written by `python cellstate_train.py` (classes live in __main__)."""

    def find_class(self, module, name):
        if module == "__main__":
            module = "cellstate_train"
        return super().find_class(module, name)


_MODEL = {}


def load_model(path):
    if path not in _MODEL:
        with open(path, "rb") as f:
            _MODEL[path] = _Unpickler(f).load()
    return _MODEL[path]


def guess_modality(img: np.ndarray) -> str:
    """QPI images are floating-point phase / dry-mass maps (negative background values, small range);
    8/16-bit images and video frames are DIC or phase contrast."""
    a = np.asarray(img)
    if a.dtype.kind == "f":
        lo, hi = np.percentile(a, (0.5, 99.5))
        if lo < 0 and hi - lo < 50:
            return "qpi"
    return "dic_or_phase"


def availability(modality: str, ood: bool):
    if ood:
        why = "this video looks unlike the videos the cell-state model was trained on"
        return dict(normal_dividing=False, dying=False, why_normal=why, why_dying=why)
    if modality == "qpi":
        return dict(normal_dividing=False, dying=True,
                    why_normal="division detection was validated on DIC and phase-contrast videos only, not QPI",
                    why_dying="")
    return dict(normal_dividing=True, dying=False, why_normal="",
                why_dying="detection of dying cells was validated on quantitative-phase (QPI) videos only; "
                          "it does not transfer to DIC or phase-contrast images")


def _predict_X(model, X):
    """Model.predict on ready features (same arithmetic), so the features can be kept for the explanations."""
    L, A, D = (h.proba(X) for h in model.heads)
    return dict(life=L, alive_state=A, death_type=D, final=np.hstack([L[:, :1] * A, L[:, 1:] * D]))


def targets_for_explain(av):
    return bool(av.get("dying") or av.get("normal_dividing"))


def compute(seq, seg, model_path, cell_model_path, diameter, out_dir: Path, debug=False, on_frame=None):
    """seq: P.Seq (2D), seg: P.SegResult of the whole-cell layer. Writes states.csv + cells.json + info.json."""
    model = load_model(model_path)
    frames = sorted(t for t in seq.frames if t in seg.files)
    first = seq.image(frames[0])
    modality = guess_modality(first)
    if modality == "qpi":
        # as in training: QPI videos (3 min per frame) were read every 2nd frame, and their time features used
        # frame index x 6 min. Reproduce exactly that (it also halves the work on these long videos).
        frames = frames[::2]
        mpf_model, mpf_real = 6.0, 3.0
    else:
        mpf_model = mpf_real = DEFAULT_MIN_PER_FRAME
    segm = B.Segmenter(cell_model_path, debug=debug)
    enc = B.Encoder(segm, stub=debug)
    enc_d = float(diameter or 30) * 0.5
    obj, embs = [], []
    for k, t in enumerate(frames):
        img = np.asarray(seq.image(t), np.float32)
        lab = seg.label(t).astype(np.int32)
        meas = B.measurements(lab, img)
        fm, rs = enc.feature_map(img, enc_d)
        pooled = enc.pool(fm, rs, lab)
        for i, m in meas.items():
            obj.append(dict(frame=t, track_id=i, **m))
            embs.append(pooled[i])
        if on_frame:
            on_frame(k, len(frames))
    obj = pd.DataFrame(obj)
    emb = np.stack(embs).astype(np.float32) if embs else np.zeros((0, 256), np.float32)
    mpf = mpf_model
    tr_modality = "qpi" if modality == "qpi" else "dic"
    info = dict(video="upload", dataset="upload", modality=tr_modality, minutes_per_frame=mpf)
    T = {"upload": TR.make_tracks(obj, emb, info)}
    rows = pd.DataFrame(dict(video="upload", track=obj.track_id.astype(int), frame=obj.frame.astype(int)))
    # domain check in the encoder-PCA space of the trained model
    z = model.pca.transform(emb) if getattr(model, "pca", None) is not None and len(emb) else np.zeros((0, 1))
    d2 = (z ** 2 / np.maximum(model.pca.explained_variance_, 1e-9)).sum(1) if len(z) else np.zeros(0)
    ood_score = float(np.median(d2) / max(1, z.shape[1])) if len(d2) else 0.0
    ood = ood_score > OOD_FACTOR
    # DIC vs phase is not known for uploads: average the predictions made under both assumptions
    Xs, preds = {}, []
    for mod in (["qpi"] if modality == "qpi" else ["dic", "phase"]):
        Xs[mod] = model._X(T, rows.assign(modality=mod, w=1.0))
        preds.append(_predict_X(model, Xs[mod]))
    P_ = {k: np.mean([p[k] for p in preds], 0) for k in preds[0]}
    out = rows.copy()
    out["p_dead"] = P_["life"][:, 1]
    for i, c in enumerate(TR.CLASSES):
        out[f"p_{c}"] = P_["final"][:, i]
    for i, c in enumerate(TR.ALIVE):
        out[f"p_{c}_if_alive"] = P_["alive_state"][:, i]
    for i, c in enumerate(TR.DEAD):
        out[f"p_{c}_if_dead"] = P_["death_type"][:, i]
    av = availability(modality, ood)
    # frame-level state, only from the parts validated for this image type
    out["state"], out["confidence"] = "unknown", np.nan
    if av["dying"]:                      # QPI: alive vs signs of dying (+ apoptosis / necrosis)
        dead = out.p_dead >= 0.5
        typ = np.where(out.p_apoptosis_if_dead >= out.p_necrosis_if_dead, "apoptosis", "necrosis")
        out["state"] = np.where(dead, typ, "alive")
        out["confidence"] = np.where(dead, out.p_dead, 1 - out.p_dead)
    elif av["normal_dividing"]:          # DIC / phase: interphase vs mitosis (death not validated here)
        pi, pm = out.p_interphase_if_alive, out.p_mitosis_if_alive
        conf = np.maximum(pi, pm) / np.maximum(pi + pm, 1e-9)
        out["state"] = np.where(pm > pi, "mitosis", "interphase")
        out["confidence"] = conf
    out.loc[out.confidence < 0.6, "state"] = "unknown"
    shown = (["alive", "apoptosis", "necrosis"] if av["dying"] else []) + \
        (["interphase", "mitosis"] if av["normal_dividing"] else [])
    # per cell (track): divisions seen as mitosis, onset of signs of dying (one irreversible switch)
    cells = {}
    for tid, g in out.sort_values("frame").groupby("track"):
        c = dict(track=int(tid), first_frame=int(g.frame.min()), last_frame=int(g.frame.max()))
        c["judged"] = bool((c["last_frame"] - c["first_frame"]) * mpf_real >= MIN_FOLLOW_MIN)
        if av["normal_dividing"]:
            mit = g[g.state == "mitosis"].frame.tolist()
            c["mitosis_frames"] = [int(x) for x in mit]
        if av["dying"]:
            p = np.clip(g.p_dead.to_numpy(), 1e-4, 1 - 1e-4)
            gain = np.concatenate([[0.0], np.cumsum((np.log(p) - np.log(1 - p))[::-1])])[::-1]
            k = int(np.argmax(gain[:-1]))
            f0 = int(g.frame.iloc[k])
            if c["judged"] and gain[k] > TR.DEATH_TAU and (f0 - c["first_frame"]) * mpf_real >= MIN_ALIVE_MIN:
                after = g.iloc[k:]
                typ = "apoptosis" if after.p_apoptosis_if_dead.mean() >= after.p_necrosis_if_dead.mean() else "necrosis"
                c.update(dying_from_frame=f0, dying_from_min=round(f0 * mpf_real, 1), death_type=typ,
                         death_type_confidence=round(float(max(after.p_apoptosis_if_dead.mean(),
                                                               after.p_necrosis_if_dead.mean())), 3))
        cells[int(tid)] = c
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in ("xai.npz", "xai.json"):              # explanations of an older run must not outlive it
        (out_dir / f).unlink(missing_ok=True)
    xai_note = ""
    if targets_for_explain(av):
        try:
            import cs_xai
            cs_xai.save(out_dir, model, Xs, rows, av, log=lambda *a: print("[cs_states]", *a, flush=True))
        except ImportError as e:
            xai_note = f"explanations skipped: {e} (pip install shap in the web app's environment)"
        except Exception as e:                       # explanations must never block the states themselves
            import traceback
            traceback.print_exc()
            xai_note = f"explanations failed: {type(e).__name__}: {e}"
    out.round(4).to_csv(out_dir / "states.csv", index=False)
    (out_dir / "cells.json").write_text(json.dumps(cells))
    summary = dict(modality=modality, minutes_per_frame_assumed=mpf_real, ood_score=round(ood_score, 2), ood=ood,
                   availability=av, shown_states=shown, n_cells=len(cells),
                   n_dying=sum(1 for c in cells.values() if "dying_from_frame" in c),
                   n_followed=sum(1 for c in cells.values() if c.get("judged")),
                   n_dividing=sum(1 for c in cells.values() if c.get("mitosis_frames")),
                   explanations=(out_dir / "xai.npz").exists(), explanations_note=xai_note)
    (out_dir / "info.json").write_text(json.dumps(summary, indent=1))
    return summary


class States:
    """read side, used by the planner tool and the page."""

    def __init__(self, d: Path):
        d = Path(d)
        self.info = json.loads((d / "info.json").read_text())
        self.cells = {int(k): v for k, v in json.loads((d / "cells.json").read_text()).items()}
        self.df = pd.read_csv(d / "states.csv")
        self.by_frame = {int(t): g.set_index("track") for t, g in self.df.groupby("frame")}
        self.frames = np.array(sorted(self.by_frame))

    def _f(self, t):
        """QPI states exist for every 2nd frame: use the nearest frame that has them."""
        if not len(self.frames):
            return None
        return int(self.frames[np.argmin(np.abs(self.frames - int(t)))])

    def at(self, t, cid):
        g = self.by_frame.get(self._f(t))
        if g is None or cid not in g.index:
            return None
        r = g.loc[cid]
        return dict(state=r.state, confidence=float(r.get("confidence", np.nan)), p_dead=float(r.p_dead))

    def summary(self, cell_ids=None):
        """one authoritative summary (the page, the pill and the assistant all use these numbers)."""
        av = self.info["availability"]
        mpf = float(self.info["minutes_per_frame_assumed"])
        cs = [c for i, c in self.cells.items() if not cell_ids or i in set(map(int, cell_ids))]
        out = dict(minutes_per_frame=mpf, n_followed=sum(1 for c in cs if c.get("judged", True)))
        if av["dying"]:
            dy = sorted((c for c in cs if "dying_from_frame" in c), key=lambda c: c["dying_from_frame"])
            n_apo = sum(c["death_type"] == "apoptosis" for c in dy)
            out.update(kind="dying", n_dying=len(dy), n_apoptosis=n_apo, n_necrosis=len(dy) - n_apo,
                       rows=[dict(cell=c["track"], type=c["death_type"], onset_frame=c["dying_from_frame"],
                                  onset_h=round(c["dying_from_frame"] * mpf / 60, 1),
                                  confidence=c["death_type_confidence"]) for c in dy])
            if dy:
                h = [r["onset_h"] for r in out["rows"]]
                out.update(first_onset_frame=dy[0]["dying_from_frame"], first_onset_h=h[0],
                           median_onset_h=round(float(np.median(h)), 1),
                           hist_h=np.histogram(h, bins=np.arange(0, max(h) + 2, 2))[0].tolist())
                out["text"] = (f"Of {out['n_followed']} cells followed for at least 2 h, {len(dy)} show signs of dying: "
                               f"{n_apo} apoptosis-like and {len(dy) - n_apo} necrosis-like. The first signs appear at "
                               f"frame {dy[0]['dying_from_frame']} (about {h[0]} h); half of them by about "
                               f"{out['median_onset_h']} h.")
            else:
                out["text"] = f"Of {out['n_followed']} cells followed for at least 2 h, none shows signs of dying."
        elif av["normal_dividing"]:
            dv = [c for c in cs if c.get("mitosis_frames")]
            out.update(kind="dividing", n_dividing=len(dv),
                       rows=[dict(cell=c["track"], first_mitosis_frame=c["mitosis_frames"][0],
                                  first_mitosis_h=round(c["mitosis_frames"][0] * mpf / 60, 1)) for c in dv])
            # dividing cells are counted among ALL tracked cells: a cell's track ends when it divides, so many
            # dividing cells are followed for less than 2 h and would not be in n_followed
            out["n_tracked"] = len(cs)
            out["text"] = (f"{len(dv)} of the {len(cs)} tracked cells were seen in mitosis (dividing) at some point "
                           f"in the video" + (f"; the first at frame {min(c['mitosis_frames'][0] for c in dv)}." if dv else "."))
        return out

    def tool(self, t, state=None, cell_ids=None):
        """what the planner gets: availability, counts at frame t and over the video, per-cell events."""
        av = self.info["availability"]
        res = dict(frame=int(t), shown_states=self.info["shown_states"],
                   availability={"normal_and_dividing": av["normal_dividing"] or av["why_normal"],
                                 "dying_apoptosis_necrosis": av["dying"] or av["why_dying"]},
                   meaning="research flags of cell BEHAVIOUR (division, signs of dying). Never a diagnosis; "
                           "never cancer, malignancy or health.")
        g = self.by_frame.get(self._f(t))
        if g is not None:
            res["counts_this_frame"] = {k: int(v) for k, v in g.state.value_counts().items()}
            if state:
                ids = [int(i) for i in g.index[g.state == state]]
                res["cells_in_state_this_frame"] = ids
        cs = [c for i, c in self.cells.items() if not cell_ids or i in set(map(int, cell_ids))]
        sm = self.summary(cell_ids)
        res["summary_text"] = sm.get("text", "")
        res["answer_style"] = ("Say it in your own words with the same numbers as summary_text (a conversational "
                               "note, not a copy). Name at most 3 example cells. The page shows the full table, a "
                               "timeline and each cell's onset frame - do not list every cell. Submit the cell ids "
                               "of the list with count = its length.")
        if av["dying"]:
            dy = sorted((c for c in cs if "dying_from_frame" in c), key=lambda c: c["dying_from_frame"])
            res["cells_with_signs_of_dying"] = [dict(cell=c["track"], from_frame=c["dying_from_frame"],
                                                     type=c["death_type"]) for c in dy][:200]
            res["timing_note"] = ("onset of visible signs of dying, not a confirmed time of death (label-free images "
                                  f"show decline before fluorescence markers); minutes assume "
                                  f"{self.info['minutes_per_frame_assumed']} min per frame")
        if av["normal_dividing"]:
            dv = [c for c in cs if c.get("mitosis_frames")]
            res["cells_seen_dividing"] = [dict(cell=c["track"], first_mitosis_frame=c["mitosis_frames"][0]) for c in dv][:200]
        res["n_cells_in_video"] = len(self.cells)
        return res