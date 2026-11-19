"""CelloSyncra explanations (XAI) for the web app.

1. SHAP for the cell-state classifier (cellstate_train.py v7: gradient-boosted trees, one per level).
   TreeSHAP (Lundberg et al. 2020) gives EXACT Shapley values for tree ensembles in log-odds units: for every
   cell and frame, base value + sum of contributions = the model's score. The ~150 features are summed into a few
   readable factors (size, shape, brightness, movement, QPI dry mass / density, image appearance ...); summing is
   valid because Shapley values are additive.
     dying      (QPI)        life head, log-odds(dead vs alive)
     necrosis   (QPI)        death-type head, log-odds(necrosis vs apoptosis)
     dividing   (DIC/phase)  alive-state head, log(p_mitosis / p_interphase)
   DIC vs phase is unknown for uploads, so (as for the predictions) the explanations under both assumptions are
   averaged.

2. Grad-CAM++ (Chattopadhay et al. 2018) on Cellpose-SAM, at the output of the second-to-last transformer block
   (32 x 32 tokens of a 256 px tile; the last block is global attention, so gradients reach the whole tile).
     seg    target = mean cell-probability logit inside the selected cell: which image regions made the
            segmenter draw this cell. Exact gradients of the real network.
     state  the trees have no gradients. Stand-in: a ridge regression from the 64 encoder components to the trees'
            IMAGE part (the sum of the SHAP values of those components), fitted per video when the states are
            computed (R^2 in-sample and cross-validated by cell are reported). It is linear in the cell's pooled
            encoder vector, so it can be differentiated back into the network.
   Faithfulness: deletion test (Petsiuk et al. 2018) - the 32 x 32 token areas are replaced by a blurred copy in
   the order of the heatmap and in random order; the drop of the score is averaged (AOPC). For "state" the score
   is the REAL tree model (the cell's other features kept, the encoder components recomputed from the edited image).

Research explanations of a research classifier. Never a diagnosis.
"""
from __future__ import annotations

import base64
import contextlib
import copy
import io
import json
import threading
from pathlib import Path

import numpy as np

ENC = "enc_pc"
TARGETS = {
    "dying": dict(head=0, pos=1, neg=0, title="Signs of dying", label="signs of dying", neg_label="alive"),
    "necrosis": dict(head=2, pos=1, neg=0, title="Type of dying", label="necrosis-like", neg_label="apoptosis-like"),
    "dividing": dict(head=1, pos=1, neg=0, title="Dividing", label="dividing (mitosis)",
                     neg_label="not dividing (interphase)"),
}
GROUPS = [("qpi_mass", "Dry mass (QPI)"), ("qpi_density", "Density (QPI)"), ("size", "Cell size"),
          ("shape", "Cell shape"), ("brightness", "Brightness"), ("texture", "Texture"), ("motion", "Movement"),
          ("imgmass", "Size × brightness"), ("appearance", "Image appearance (encoder)"),
          ("context", "Other (image type, history)")]
GROUP_LABEL = dict(GROUPS)
MEAS_NICE = {"area": "area", "perimeter": "perimeter", "eccentricity": "eccentricity", "solidity": "solidity",
             "circularity": "roundness", "aspect_ratio": "elongation", "intensity_img": "brightness",
             "texture": "texture", "texture_img": "texture (image)"}


def targets_for(av: dict) -> list[str]:
    if av.get("dying"):
        return ["dying", "necrosis"]
    if av.get("normal_dividing"):
        return ["dividing"]
    return []


def _base_meas(n):
    for m in sorted(MEAS_NICE, key=len, reverse=True):
        if n == m or n.startswith(m + "_"):
            return m
    return None


def group_of(n: str) -> str:
    if n.startswith(ENC):
        return "appearance"
    if n.startswith("qpi_dry_mass"):
        return "qpi_mass"
    if n.startswith("qpi_density"):
        return "qpi_density"
    if n.startswith("is_") or n == "has_past":
        return "context"
    if n.startswith("speed"):
        return "motion"
    if n.startswith("mass"):
        return "imgmass"
    b = _base_meas(n)
    if b in ("area", "perimeter"):
        return "size"
    if b in ("eccentricity", "solidity", "circularity", "aspect_ratio"):
        return "shape"
    if b == "intensity_img":
        return "brightness"
    if b in ("texture", "texture_img"):
        return "texture"
    return "context"


def feat_label(n: str) -> str:
    special = {"speed_30min": "speed (mean over ±30 min)", "mass_30min": "size × brightness (mean over ±30 min)",
               "mass_vs_own_past": "size × brightness vs. its own past", "has_past": "has 2 h of history",
               "qpi_density": "density", "qpi_dry_mass": "dry mass",
               "qpi_density_vs_past": "density vs. its own past (30 min–2 h ago)",
               "qpi_dry_mass_vs_past": "dry mass vs. its own past (30 min–2 h ago)",
               "qpi_density_change": "density change (±1 frame)", "qpi_dry_mass_change": "dry mass change (±1 frame)",
               "qpi_dry_mass_vs_own_max": "dry mass vs. its maximum so far",
               "qpi_density_vs_own_start": "density vs. its first 2 h"}
    if n in special:
        return special[n]
    if n.startswith("is_"):
        return "image type: " + n[3:].upper()
    if n.startswith(ENC):
        return "encoder component " + n[len(ENC):]
    b = _base_meas(n)
    if b is None:
        return n
    rest = n[len(b):]
    suf = {"": "", "_mean": " (mean over ±3 frames)", "_std": " (variation over ±3 frames)", "_change": " (change)",
           "_vs_own_past": " vs. its own past (30 min–2 h ago)"}.get(rest, rest.replace("_", " "))
    return MEAS_NICE[b] + suf


# ----------------------------------------------------------------------------- SHAP (write side)
_EXPL = {}


def _explainer(m):
    import shap
    k = id(m)
    if k not in _EXPL:
        _EXPL[k] = shap.TreeExplainer(m)
    return _EXPL[k]


def head_raw_contrast(head, X, pos, neg):
    """the tree model's own score for pos vs neg (log-odds); None if the head cannot tell them apart."""
    m = head.m
    if m is None or pos not in m.classes_ or neg not in m.classes_:
        return None
    d = m.decision_function(X[:, head.cols])
    cls = list(m.classes_)
    if d.ndim == 1:
        return d if cls[1] == pos else -d
    return d[:, cls.index(pos)] - d[:, cls.index(neg)]


def head_shap(head, X, pos, neg):
    """exact TreeSHAP values of pos-vs-neg log-odds, full feature width (features the head did not use = 0)."""
    m = head.m
    if m is None or pos not in m.classes_ or neg not in m.classes_:
        return None
    ex = _explainer(m)
    sv = ex.shap_values(X[:, head.cols], check_additivity=False)
    if isinstance(sv, list):
        sv = np.stack(sv, -1)
    sv = np.asarray(sv, np.float64)
    ev = np.atleast_1d(np.asarray(ex.expected_value, np.float64))
    cls = list(m.classes_)
    if sv.ndim == 2:                                  # binary: score of classes_[1]
        s = 1.0 if cls[1] == pos else -1.0
        S, base = s * sv, s * float(ev[0])
    else:
        ip, ineg = cls.index(pos), cls.index(neg)
        S, base = sv[:, :, ip] - sv[:, :, ineg], float(ev[ip] - ev[ineg])
    full = np.zeros((len(X), X.shape[1]))
    full[:, head.cols] = S
    return full, base


def _ridge(Z, y, lam_rel=1e-3):
    mu, ym = Z.mean(0), y.mean()
    Zc = Z - mu
    A = Zc.T @ Zc
    lam = lam_rel * np.trace(A) / max(1, Z.shape[1]) + 1e-9
    beta = np.linalg.solve(A + lam * np.eye(Z.shape[1]), Zc.T @ (y - ym))
    return beta, float(ym - mu @ beta)


def _r2(y, p):
    v = float(((y - y.mean()) ** 2).sum())
    return float(1 - ((y - p) ** 2).sum() / v) if v > 0 else 0.0


def save(out_dir: Path, model, Xs: dict, rows, av: dict, log=print):
    """after the predictions in cs_states.compute: exact SHAP values for every (cell, frame) + the stand-in head
    for the cell-state heatmap. Writes xai.npz + xai.json."""
    mods = list(Xs)
    names = list(model.names)
    X0 = np.asarray(Xs[mods[0]], np.float32)
    enc_idx = np.array([i for i, n in enumerate(names) if n.startswith(ENC)], int)
    data = dict(track=rows.track.to_numpy(int), frame=rows.frame.to_numpy(int), X=X0, names=np.array(names),
                mods=np.array(mods), enc_idx=enc_idx)
    meta = dict(targets=[], method="TreeSHAP (exact, log-odds); averaged over " + " and ".join(mods))
    tracks = data["track"]
    for key in targets_for(av):
        T = TARGETS[key]
        res = [head_shap(model.heads[T["head"]], np.asarray(Xs[md], np.float64), T["pos"], T["neg"]) for md in mods]
        if any(r is None for r in res):
            log(f"xai: {key}: head cannot separate the classes, skipped")
            continue
        S = np.mean([r[0] for r in res], 0)
        base = float(np.mean([r[1] for r in res]))
        raw = np.mean([head_raw_contrast(model.heads[T["head"]], np.asarray(Xs[md], np.float64), T["pos"], T["neg"])
                       for md in mods], 0)
        add_err = float(np.abs(base + S.sum(1) - raw).max()) if len(raw) else 0.0
        data[f"sv_{key}"] = S.astype(np.float32)
        tm = dict(key=key, base=round(base, 4), additivity_error=add_err, **{k: T[k] for k in ("title", "label", "neg_label")})
        if len(enc_idx) and getattr(model, "pca", None) is not None and len(X0) >= 20:
            Z = X0[:, enc_idx].astype(np.float64)
            y = S[:, enc_idx].sum(1)
            if not np.isfinite(Z).all() or y.std() < 1e-6:
                tm.update(stand_in=None, stand_in_note="the model's image-appearance part does not vary in this video, "
                                                       "so there is nothing for a cell-state heatmap to show")
                meta["targets"].append(tm)
                data[f"sv_{key}"] = S.astype(np.float32)
                log(f"xai: {key}: SHAP for {len(S)} rows, additivity error {add_err:.2e}, no stand-in (image part constant)")
                continue
            beta, b0 = _ridge(Z, y)
            r2 = _r2(y, Z @ beta + b0)
            fold = np.array([hash(int(t)) % 3 for t in tracks])  # cross-validated by CELL (frames of a cell together)
            pred = np.zeros_like(y)
            for f in range(3):
                tr, te = fold != f, fold == f
                if te.any() and tr.sum() > Z.shape[1]:
                    b_, c_ = _ridge(Z[tr], y[tr])
                    pred[te] = Z[te] @ b_ + c_
            r2cv = _r2(y, pred)
            data[f"ge_{key}"] = (beta @ model.pca.components_).astype(np.float32)   # d(stand-in)/d(encoder vector)
            tm.update(stand_in=dict(r2=round(r2, 3), r2_cv_by_cell=round(r2cv, 3), n=int(len(y)),
                                    image_part_sd=round(float(y.std()), 3)))
        meta["targets"].append(tm)
        log(f"xai: {key}: SHAP for {len(S)} rows, additivity error {add_err:.2e}, stand-in {tm.get('stand_in')}")
    out_dir = Path(out_dir)
    np.savez_compressed(out_dir / "xai.npz", **data)
    (out_dir / "xai.json").write_text(json.dumps(meta, indent=1))
    return meta


# ----------------------------------------------------------------------------- SHAP (read side)
def _sig(x):
    return float(1 / (1 + np.exp(-np.clip(x, -40, 40))))


class Explanations:
    def __init__(self, d: Path):
        d = Path(d)
        z = np.load(d / "xai.npz", allow_pickle=False)
        self.d = {k: z[k] for k in z.files}
        self.meta = json.loads((d / "xai.json").read_text())
        self.names = [str(n) for n in self.d["names"]]
        self.groups = np.array([group_of(n) for n in self.names])

    def row(self, cid, t):
        idx = np.nonzero(self.d["track"] == int(cid))[0]
        if not len(idx):
            return None
        return int(idx[np.argmin(np.abs(self.d["frame"][idx] - int(t)))])

    def target(self, key):
        return next((tm for tm in self.meta["targets"] if tm["key"] == key), None)

    def explain(self, cid, t, top_groups=7):
        k = self.row(cid, t)
        if k is None:
            return None
        X = self.d["X"][k]
        out = dict(cell=int(cid), frame=int(self.d["frame"][k]), method=self.meta["method"], targets=[])
        for tm in self.meta["targets"]:
            sv = self.d[f"sv_{tm['key']}"][k].astype(np.float64)
            score = tm["base"] + float(sv.sum())
            gs = []
            for g, lab in GROUPS:
                ii = np.nonzero(self.groups == g)[0]
                if not len(ii):
                    continue
                val = float(sv[ii].sum())
                top = ii[np.argsort(-np.abs(sv[ii]))][:3]
                gs.append(dict(group=g, label=lab, shap=round(val, 4),
                               top=[dict(feature=self.names[i], label=feat_label(self.names[i]),
                                         value=(None if not np.isfinite(X[i]) else round(float(X[i]), 4)),
                                         shap=round(float(sv[i]), 4)) for i in top if abs(sv[i]) > 1e-6]))
            gs = sorted([g for g in gs if abs(g["shap"]) > 1e-6], key=lambda g: -abs(g["shap"]))
            rest = gs[top_groups:]
            gs = gs[:top_groups]
            if rest:
                gs.append(dict(group="rest", label=f"{len(rest)} smaller factors", shap=round(sum(g["shap"] for g in rest), 4), top=[]))
            p = _sig(score)
            out["targets"].append(dict(key=tm["key"], title=tm["title"], label=tm["label"], neg_label=tm["neg_label"],
                                       base=tm["base"], base_prob=round(_sig(tm["base"]), 3), score=round(score, 4),
                                       prob=round(p, 3), leans=tm["label"] if score >= 0 else tm["neg_label"],
                                       groups=gs, stand_in=tm.get("stand_in")))
        return out

    def text(self, cid, t):
        """short plain-language version for the assistant."""
        e = self.explain(cid, t, top_groups=4)
        if e is None:
            return None
        parts = []
        for tg in e["targets"]:
            f = "; ".join(f"{g['label']} {'+' if g['shap'] >= 0 else '−'}{abs(g['shap']):.2f} "
                          f"(toward {tg['label'] if g['shap'] >= 0 else tg['neg_label']})" for g in tg["groups"] if g["group"] != "rest")
            parts.append(dict(question=tg["title"], model_leans=tg["leans"], score_log_odds=tg["score"],
                              average_cell_log_odds=tg["base"], main_factors=f))
        return dict(cell=e["cell"], frame_used=e["frame"], explanations=parts,
                    method="exact TreeSHAP contributions (log-odds) of the cell-state model, grouped into readable "
                           "factors; positive = toward the first state named. Research explanation, not a diagnosis.")


# ----------------------------------------------------------------------------- Grad-CAM++
def gradcampp(A, G):
    """A, G: (C, h, w) activations and gradients of the target -> (h, w) Grad-CAM++ map (closed form with the
    exponential-of-score assumption, as in the original paper / pytorch-grad-cam)."""
    g2, g3 = G ** 2, G ** 3
    sA = A.sum((1, 2), keepdims=True)
    den = 2 * g2 + sA * g3
    alpha = np.where(G != 0, g2 / np.where(np.abs(den) > 1e-12, den, 1e-12), 0.0)
    w = (alpha * np.maximum(G, 0)).sum((1, 2))
    return np.maximum((w[:, None, None] * A).sum(0), 0)


def _spear(a, b):
    """Spearman rank correlation, None when undefined."""
    from scipy.stats import spearmanr
    a, b = np.ravel(np.asarray(a, np.float64)), np.ravel(np.asarray(b, np.float64))
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return round(float(spearmanr(a, b)[0]), 3)


def _frac(cam, m):
    s = cam.sum()
    return float(cam[m].sum() / s) if s > 0 else 0.0


def _colorize(v):
    try:
        from matplotlib import colormaps
        rgba = colormaps["turbo"](v)
    except Exception:                                  # simple blue-red ramp
        rgba = np.stack([v, 0.2 + 0.3 * (1 - np.abs(v - 0.5) * 2), 1 - v, np.ones_like(v)], -1)
    rgba = np.asarray(rgba, np.float32)
    # weak heat fully transparent, strong heat clearly visible: broad context maps no longer wash over the image
    rgba[..., 3] = np.clip((v - 0.35) / 0.65, 0, 1) ** 1.3 * 0.8
    return (rgba * 255).astype(np.uint8)


def _diverging(v, inside):
    """signed map in [-1, 1]: towards the shown state = warm (orange -> yellow), against = blue; transparent off-cell."""
    a = np.abs(v)
    rgba = np.zeros(v.shape + (4,), np.float32)
    posc = np.stack([np.ones_like(a), 0.45 + 0.5 * a, 0.15 * (1 - a)], -1)          # orange -> yellow
    negc = np.stack([0.15 * (1 - a), 0.45 + 0.25 * a, np.ones_like(a)], -1)         # blue
    rgba[..., :3] = np.where((v >= 0)[..., None], posc, negc)
    rgba[..., 3] = np.where(inside, 0.12 + 0.78 * a ** 0.8, 0.0)
    return (np.clip(rgba, 0, 1) * 255).astype(np.uint8)


def _heat(v):
    """[0, 1] -> 'hot' overlay: weak = transparent, then red -> orange -> yellow -> white for the most important pixels."""
    v = np.clip(np.asarray(v, np.float32), 0, 1)
    rgba = np.zeros(v.shape + (4,), np.float32)
    rgba[..., 0] = np.clip(v * 3, 0, 1)
    rgba[..., 1] = np.clip(v * 3 - 1, 0, 1)
    rgba[..., 2] = np.clip(v * 3 - 2, 0, 1)
    rgba[..., 0] = np.maximum(rgba[..., 0], 0.85 * (v > 0.02))
    rgba[..., 3] = np.clip((v - 0.06) / 0.5, 0, 1) ** 0.9 * 0.9
    return (rgba * 255).astype(np.uint8)


def _png(rgba):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()


class CamEngine:
    TILE = 256

    def __init__(self, model_path=None, debug=False, net=None):
        import torch
        self.torch = torch
        if net is None:
            if debug:                                     # tests: Cellpose-SAM architecture, random weights, CPU
                from cellpose.vit import CPSAM
                net = CPSAM().eval()
            else:
                import cs_states                   # puts the CellState code folder on the path
                net = cs_states.B.Segmenter(model_path).model.net
        self.net = net.eval()
        p = next(self.net.parameters())
        self.dev, self.dtype = p.device, p.dtype
        self.lock = threading.Lock()

    # ---- image -> 256 px tile around the cell, exactly the segmentation preprocessing
    def _prep(self, img, diameter):
        from cellpose import transforms
        x = transforms.convert_image(np.asarray(img, np.float32), channel_axis=None, z_axis=None, do_3D=False)
        x = transforms.normalize_img(x)
        rs = 30.0 / float(diameter)
        if rs != 1:
            x = transforms.resize_image(x, rsz=rs)
        return np.asarray(x, np.float32)[..., 0], rs

    def _tile(self, x, lab_shape, cell, rs):
        H0, W0 = lab_shape
        t = self.TILE
        ys, xs = np.nonzero(cell)
        cy, cx = (ys.mean() + 0.5) * x.shape[0] / H0, (xs.mean() + 0.5) * x.shape[1] / W0
        y0, x0 = int(round(cy - t / 2)), int(round(cx - t / 2))
        tile = np.zeros((t, t), np.float32)
        sy0, sx0 = max(0, y0), max(0, x0)
        sy1, sx1 = min(x.shape[0], y0 + t), min(x.shape[1], x0 + t)
        tile[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = x[sy0:sy1, sx0:sx1]
        valid = np.zeros((t, t), bool)
        valid[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = True
        # cell mask in tile pixels (nearest original pixel)
        fy, fx = x.shape[0] / H0, x.shape[1] / W0
        oy = np.floor((y0 + np.arange(t) + 0.5) / fy).astype(int)
        ox = np.floor((x0 + np.arange(t) + 0.5) / fx).astype(int)
        oky, okx = (oy >= 0) & (oy < H0), (ox >= 0) & (ox < W0)
        m = np.zeros((t, t), bool)
        m[np.ix_(oky, okx)] = cell[np.ix_(oy[oky], ox[okx])]
        return tile, valid, m, (y0, x0, fy, fx)

    # ---- network pieces (same as CPSAM.forward)
    def _trunk(self, tiles):
        torch, net = self.torch, self.net
        F = torch.nn.functional
        x = torch.zeros((len(tiles), 3, self.TILE, self.TILE), device=self.dev, dtype=self.dtype)
        x[:, 0] = torch.from_numpy(np.asarray(tiles, np.float32)).to(self.dev, self.dtype)
        enc = net.encoder
        with torch.no_grad():
            h = F.conv2d(x, enc.patch_embed.proj.weight.data[:, :3], bias=enc.patch_embed.proj.bias.data, stride=net.ps)
            h = h.permute(0, 2, 3, 1)
            if enc.pos_embed is not None:
                h = h + enc.pos_embed
            for blk in enc.blocks[:-1]:
                h = blk(h)
        return h                                          # (B, 32, 32, C): input of the last (global) block

    def _head_from_ln(self, h, xln):
        """last block written out (SAM Block, global attention): the Grad-CAM++ layer is its LayerNorm-ed input
        (norm1 output) - as usual for ViTs, because the raw residual stream has a few huge channels that would
        dominate the weighted sum."""
        torch, net = self.torch, self.net
        blk = net.encoder.blocks[-1]
        x = h + blk.attn(xln)
        x = x + blk.mlp(blk.norm2(x))
        n = net.encoder.neck(x.permute(0, 3, 1, 2))
        y = torch.nn.functional.conv_transpose2d(net.out(n), net.W2, stride=net.ps, padding=0)
        return y, n

    def _head(self, h):
        torch, net = self.torch, self.net
        h = net.encoder.blocks[-1](h)
        n = net.encoder.neck(h.permute(0, 3, 1, 2))      # (B, 256, 32, 32): the 'neck' (encoder features)
        y = torch.nn.functional.conv_transpose2d(net.out(n), net.W2, stride=net.ps, padding=0)
        return y, n

    def _cam(self, tile, valid, mask_t, small_t, kind, ge_t):
        """Grad-CAM++ on the token grid (g x g) for one target, and the target's score. Call under self.lock."""
        torch = self.torch
        h = self._trunk([tile]).detach()
        blk = self.net.encoder.blocks[-1]
        with torch.enable_grad():
            if getattr(blk, "window_size", 0) == 0:
                act = blk.norm1(h).detach().requires_grad_(True)
                y, n = self._head_from_ln(h, act)
            else:                                       # windowed block: fall back to the block input
                act = h.requires_grad_(True)
                y, n = self._head(act)
            tgt = self._target(y, n, mask_t, small_t, kind, ge_t)
            G, = torch.autograd.grad(tgt.sum(), act)
        A = act.detach().float()[0].permute(2, 0, 1).cpu().numpy()
        Gn = G.float()[0].permute(2, 0, 1).cpu().numpy()
        cam = gradcampp(A, Gn)
        cam = np.where(self._small(valid, self.TILE // self.net.ps), cam, 0)   # padding outside the image: no evidence
        return cam, float(tgt.detach().float()[0])

    @staticmethod
    def _small(mask, g):
        from skimage.transform import resize
        s = resize(mask.astype(np.float32), (g, g), order=0, preserve_range=True, anti_aliasing=False) > 0.5
        if not s.any():
            ys, xs = np.nonzero(mask)
            s[min(g - 1, int(ys.mean() * g / mask.shape[0])), min(g - 1, int(xs.mean() * g / mask.shape[1]))] = True
        return s

    def _target(self, y, n, mask_t, small_t, kind, ge_t):
        if kind == "seg":
            return (y[:, 2].float() * mask_t).sum((1, 2)) / mask_t.sum()
        e = (n.float() * small_t).sum((2, 3)) / small_t.sum()
        return e @ ge_t

    def explain(self, img, lab, cid, diameter, kind="seg", ge=None, sign=1.0, scorer=None, faith=True, seed=0):
        """-> dict(bbox=[x0, y0, x1, y1] in image pixels, png=base64 RGBA heatmap for that box, faithfulness...)."""
        torch = self.torch
        if int(cid) <= 0:
            raise ValueError("cell ids start at 1 (0 is the background)")
        cell = np.asarray(lab) == int(cid)
        if not cell.any():
            raise ValueError(f"cell {cid} is not in this frame")
        x, rs = self._prep(img, diameter)
        tile, valid, mask, (y0, x0, fy, fx) = self._tile(x, cell.shape, cell, rs)
        g = self.TILE // self.net.ps
        small = self._small(mask, g)
        mask_t = torch.from_numpy(mask.astype(np.float32)).to(self.dev)
        small_t = torch.from_numpy(small.astype(np.float32)).to(self.dev)
        ge_t = None if ge is None else torch.from_numpy(np.asarray(ge, np.float32) * float(sign)).to(self.dev)
        with self.lock:
            cam, score0 = self._cam(tile, valid, mask_t, small_t, kind, ge_t)
            fa = self._deletion(tile, valid, cam, mask_t, small_t, kind, ge_t, scorer, seed) if faith else None
        # heatmap -> image pixels of the tile's area
        from skimage.transform import resize
        from scipy.ndimage import distance_transform_edt
        t = self.TILE
        up = resize(cam.astype(np.float64), (t, t), order=1)
        up[~valid] = 0
        # focus: heat further than ~1 cell diameter from the cell fades out (its share is reported, not hidden in the
        # numbers), and colours are scaled to the heat near the cell, so a broad context map no longer paints the
        # whole 256-px view
        dcell = 2 * np.sqrt(max(mask.sum(), 1) / np.pi)
        dist = distance_transform_edt(~mask)
        win = np.exp(-(dist / max(dcell, 8.0)) ** 2)
        near = dist <= dcell
        far_share = round(float(up[~near].sum() / up.sum()), 3) if up.sum() > 0 else None
        ref = float(np.percentile(up[near & valid], 99)) if (near & valid).any() and up[near & valid].max() > 0 else float(up.max())
        up = np.clip(up / ref, 0, 1) * win if ref > 0 else up
        H0, W0 = cell.shape
        ox0, oy0 = max(0, int(np.floor(x0 / fx))), max(0, int(np.floor(y0 / fy)))
        ox1, oy1 = min(W0, int(np.ceil((x0 + t) / fx))), min(H0, int(np.ceil((y0 + t) / fy)))
        ty0, ty1 = int(round(oy0 * fy - y0)), int(round(oy1 * fy - y0))
        tx0, tx1 = int(round(ox0 * fx - x0)), int(round(ox1 * fx - x0))
        crop = up[max(0, ty0):min(t, ty1), max(0, tx0):min(t, tx1)]
        crop = resize(crop, (oy1 - oy0, ox1 - ox0), order=1)
        # soft fade where the model's view ends inside the image (no hard straight edge in the overlay)
        hh, ww = crop.shape
        fy_, fx_ = np.ones(hh), np.ones(ww)
        k = max(4, int(0.08 * min(hh, ww)))
        ramp = np.linspace(0, 1, k)
        if oy0 > 0: fy_[:k] = np.minimum(fy_[:k], ramp)
        if oy1 < H0: fy_[-k:] = np.minimum(fy_[-k:], ramp[::-1])
        if ox0 > 0: fx_[:k] = np.minimum(fx_[:k], ramp)
        if ox1 < W0: fx_[-k:] = np.minimum(fx_[-k:], ramp[::-1])
        crop = crop * fy_[:, None] * fx_[None, :]
        frac_in = round(float((cam * small).sum() / cam.sum()), 3) if cam.sum() > 0 else None
        return dict(bbox=[ox0, oy0, ox1, oy1], png=_png(_colorize(np.clip(crop, 0, 1))), score=round(score0, 4),
                    heat_inside_cell=frac_in, heat_far_faded=far_share, layer=f"LayerNorm-ed input of block {len(self.net.encoder.blocks)} of "
                    f"{len(self.net.encoder.blocks)} (the last, global-attention block)", faithfulness=fa,
                    tile_scale=round(rs, 4))

    def patches(self, img, lab, cid, diameter, scorer, faith=True, seed=0, n_sub=None):
        """Cell-state heatmap, exact with respect to the REAL tree model (no stand-in, no gradients).
        The cell-state model sees the average of the encoder vectors of the cell's own patches (8 x 8 px tokens).
        We hide random halves of those patches, recompute the average, score it with the real model (other
        features kept), and fit score ~ b + sum_i w_i [patch i kept]: w_i is the average effect of patch i
        (a KernelSHAP-style estimate for the pooled tokens). Positive = pushes towards the shown state.
        Fidelity: R^2 of that linear fit on held-out subsets. Display: only on the cell, signed colours."""
        torch = self.torch
        if int(cid) <= 0:
            raise ValueError("cell ids start at 1 (0 is the background)")
        cell = np.asarray(lab) == int(cid)
        if not cell.any():
            raise ValueError(f"cell {cid} is not in this frame")
        x, rs = self._prep(img, diameter)
        tile, valid, mask, geo = self._tile(x, cell.shape, cell, rs)
        g, ps = self.TILE // self.net.ps, self.net.ps
        small = self._small(mask, g)
        with self.lock, torch.no_grad():
            _, n = self._head(self._trunk([tile]))
            Fm = n.float()[0][:, torch.from_numpy(small).to(n.device)].T.cpu().numpy().astype(np.float64)   # (N, 256)
        N = len(Fm)
        full = float(scorer(Fm.mean(0, keepdims=True))[0])
        rng = np.random.default_rng(seed)
        if N == 1:
            w, r2, b0 = np.array([0.0]), None, full
        else:
            K = int(n_sub or min(2048, max(256, 12 * N)))
            Z = rng.random((K, N)) < 0.5
            Z[Z.sum(1) == 0, rng.integers(0, N)] = True
            E = (Z.astype(np.float64) @ Fm) / Z.sum(1, keepdims=True)
            s = np.asarray(scorer(E), np.float64)
            Zf = Z.astype(np.float64)

            def fit(Zs, ss, lam=1e-3):
                Zc, sc = Zs - Zs.mean(0), ss - ss.mean()
                A = Zc.T @ Zc
                ww = np.linalg.solve(A + lam * np.trace(A) / len(A) * np.eye(len(A)), Zc.T @ sc)
                return ww, ss.mean() - Zs.mean(0) @ ww
            cut = int(0.8 * K)
            w_tr, b_tr = fit(Zf[:cut], s[:cut])
            pred = Zf[cut:] @ w_tr + b_tr
            r2 = _r2(s[cut:], pred) if s[cut:].std() > 1e-9 else None
            w, b0 = fit(Zf, s)
        W = np.zeros((g, g))
        W[small] = w
        fa = self._deletion_cell(tile, valid, W, small, scorer, seed) if faith and N >= 4 else None
        # display: patch values -> pixels of the cell only (smoothed inside the outline), signed colours
        from scipy.ndimage import gaussian_filter
        pix = np.kron(W, np.ones((ps, ps)))
        m = mask.astype(np.float64)
        sm = gaussian_filter(pix * m, ps * 0.6) / np.maximum(gaussian_filter(m, ps * 0.6), 1e-6) * m
        ref = float(np.percentile(np.abs(sm[mask]), 98)) if mask.any() else 0.0
        v = np.clip(sm / ref, -1, 1) if ref > 0 else sm
        bbox, crop = self._tile_to_image(v, valid, geo, cell.shape, pad=0.15, box=mask)
        cmask = cell[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        crop = np.where(cmask, crop, 0.0)
        pos, neg = float(w[w > 0].sum()), float(-w[w < 0].sum())
        return dict(bbox=bbox, png=_png(_diverging(crop, cmask)), score=round(full, 4), n_patches=int(N),
                    surrogate_r2=None if r2 is None else round(float(r2), 3),
                    share_towards=round(pos / (pos + neg), 3) if pos + neg > 0 else None,
                    score_no_patches_hidden=round(full, 4), faithfulness=fa, tile_scale=round(rs, 4),
                    layer="encoder features of the cell's own patches (what the cell-state model averages)")

    def _forward_grad(self, xt):
        """full Cellpose-SAM forward from pixels WITH gradients: xt (B, 256, 256) tensor -> (y, neck)."""
        torch, net = self.torch, self.net
        F = torch.nn.functional
        enc = net.encoder
        x = torch.zeros((xt.shape[0], 3) + tuple(xt.shape[1:]), device=self.dev, dtype=self.dtype)
        x = x.index_copy(1, torch.tensor([0], device=self.dev), xt[:, None].to(self.dtype))
        h = F.conv2d(x, enc.patch_embed.proj.weight[:, :3], bias=enc.patch_embed.proj.bias, stride=net.ps).permute(0, 2, 3, 1)
        if enc.pos_embed is not None:
            h = h + enc.pos_embed
        for blk in enc.blocks:
            h = blk(h)
        n = enc.neck(h.permute(0, 3, 1, 2))
        y = F.conv_transpose2d(net.out(n), net.W2, stride=net.ps, padding=0)
        return y, n

    def gradcam_pixels(self, img, lab, cid, diameter, faith=True, seed=0, n_smooth=8, noise=0.1, edge_px=6, detail=False):
        """Segmentation heatmap: pixel-level Grad-CAM++ for ONE cell.
        1. Grad-CAM++ on the neck (the encoder output that feeds the cell-probability head; each 8 x 8 px patch only
           affects its own pixels there, so the map stays local), target = mean cell-probability logit in the cell;
        2. only with detail=True: times the pixel saliency |d target / d pixel| averaged over n_smooth noisy copies
           (SmoothGrad) - pixel detail ('Guided Grad-CAM'); pixel gradients mostly follow edges, so this view shows
           where the outline was decided, while the default view shows which parts of the cell body mattered;
        3. shown on the cell and a thin edge (edge_px, in the model's input scale)."""
        from scipy.ndimage import gaussian_filter, binary_dilation
        from skimage.transform import resize
        torch = self.torch
        if int(cid) <= 0:
            raise ValueError("cell ids start at 1 (0 is the background)")
        cell = np.asarray(lab) == int(cid)
        if not cell.any():
            raise ValueError(f"cell {cid} is not in this frame")
        x, rs = self._prep(img, diameter)
        tile, valid, mask, geo = self._tile(x, cell.shape, cell, rs)
        g, ps, T = self.TILE // self.net.ps, self.net.ps, self.TILE
        small = self._small(mask, g)
        mask_t = torch.from_numpy(mask.astype(np.float32)).to(self.dev)
        small_t = torch.from_numpy(small.astype(np.float32)).to(self.dev)
        rng = np.random.default_rng(seed)
        sd = float(np.std(tile[valid])) if valid.any() else 1.0
        batch = np.stack([tile] + [tile + rng.normal(0, noise * sd, tile.shape).astype(np.float32) * valid
                                   for _ in range(n_smooth if detail else 0)])
        with self.lock:
            with torch.enable_grad():
                xt = torch.from_numpy(batch).to(self.dev).requires_grad_(True)
                y, n = self._forward_grad(xt)
                tgt = self._target(y, n, mask_t, small_t, "seg", None)
                gx, gn = torch.autograd.grad(tgt.sum(), [xt, n])     # only these: no gradients stored on the weights
            A = n.detach().float()[0].cpu().numpy()                 # (256, g, g)
            G = gn.detach().float()[0].cpu().numpy()
            sal = gx.detach().float().abs().mean(0).cpu().numpy()
            score0 = float(tgt.detach().float()[0])
        cam = gradcampp(A, G)
        # at the neck only the patches with a non-zero gradient can change this cell's probability (the head is
        # local), so Grad-CAM++ values elsewhere (other cell-like areas) are not evidence for this cell
        cam = np.where(self._small(valid, g) & (np.abs(G).sum(0) > 0), cam, 0)
        # pixel map = Grad-CAM++ (upsampled) [x smoothed pixel saliency], on the cell + thin edge
        show = (binary_dilation(mask, iterations=int(edge_px)) if detail else mask) & valid   # default: exactly the cell
        up = resize(cam, (T, T), order=1, preserve_range=True)
        sal = gaussian_filter(sal, 1.0) if detail else np.ones_like(up)
        pix = (up / (up[show].max() + 1e-12) if show.any() else up) * (sal / (np.percentile(sal[show], 99) + 1e-12) if show.any() else sal)
        pix = np.where(show, pix, 0.0)
        ref = float(np.percentile(pix[show], 99.5)) if show.any() and pix[show].max() > 0 else 0.0
        v = np.clip(pix / ref, 0, 1) if ref > 0 else pix
        if detail:          # pixel gradients reach the whole view: where does this evidence lie?
            tot_all, tot_show = float((up * sal * valid).sum()), float((up * sal * show).sum())
            on_cell = float((up * sal * mask).sum() / tot_all) if tot_all > 0 else None
        else:
            tot_all = tot_show = 1.0
            on_cell = None if cam.max() <= 0 else 1.0
        region_t = binary_dilation(small, iterations=1) & self._small(valid, g)
        fa = self._deletion_region(tile, valid, np.where(region_t, cam, 0.0), region_t, mask_t, small_t, seed) if faith else None
        bbox, crop = self._tile_to_image(v, valid, geo, cell.shape, pad=0.05, box=show)
        return dict(bbox=bbox, png=_png(_heat(crop)), score=round(score0, 4),
                    heat_inside_cell=None if on_cell is None else round(on_cell, 3),
                    heat_shown=round(tot_show / tot_all, 3) if tot_all > 0 else None, faithfulness=fa,
                    tile_scale=round(rs, 4), detail=bool(detail), smoothgrad_samples=n_smooth if detail else 0,
                    layer="Grad-CAM++ on the neck (encoder output feeding the cell-probability head)"
                          + (" x SmoothGrad pixel saliency" if detail else ""))

    def occlusion_seg(self, img, lab, cid, diameter, faith=True, seed=0, reach=1.0, win=2, batch=None):
        """Segmentation heatmap by occlusion (Zeiler & Fergus 2014), faithful by construction: around the cell, a
        win x win-token square (8 px tokens) is replaced by a blurred copy, sliding one token at a time; the drop in
        the segmenter's cell probability for THIS cell is recorded. Each token gets the mean drop of the squares
        covering it. Positive = the outline depends on this area; negative = hiding it makes the cell more certain
        (the area argues against it). Searched up to `reach` cell diameters from the cell."""
        from scipy.ndimage import gaussian_filter, distance_transform_edt
        torch = self.torch
        if int(cid) <= 0:
            raise ValueError("cell ids start at 1 (0 is the background)")
        cell = np.asarray(lab) == int(cid)
        if not cell.any():
            raise ValueError(f"cell {cid} is not in this frame")
        x, rs = self._prep(img, diameter)
        tile, valid, mask, geo = self._tile(x, cell.shape, cell, rs)
        g, ps = self.TILE // self.net.ps, self.net.ps
        small = self._small(mask, g)
        vs = self._small(valid, g)
        dcell_t = 2 * np.sqrt(max(small.sum(), 1) / np.pi)                    # cell diameter in tokens
        region = (distance_transform_edt(~small) <= max(2.0, reach * dcell_t)) & vs
        mask_t = torch.from_numpy(mask.astype(np.float32)).to(self.dev)
        small_t = torch.from_numpy(small.astype(np.float32)).to(self.dev)
        blur = gaussian_filter(tile, 6)
        starts = [(i, j) for i in range(g - win + 1) for j in range(g - win + 1) if region[i:i + win, j:j + win].any()]
        imgs = [tile]
        for i, j in starts:
            pm = np.zeros(tile.shape, bool)
            pm[i * ps:(i + win) * ps, j * ps:(j + win) * ps] = True
            imgs.append(np.where(pm & valid, blur, tile))
        batch = batch or (32 if self.dev.type == "cuda" else 8)
        sc = []
        with self.lock, torch.no_grad():
            for k in range(0, len(imgs), batch):
                y, n = self._head(self._trunk(imgs[k:k + batch]))
                sc += self._target(y, n, mask_t, small_t, "seg", None).float().cpu().tolist()
        s0 = sc[0]
        acc, cnt = np.zeros((g, g)), np.zeros((g, g))
        for (i, j), s in zip(starts, sc[1:]):
            acc[i:i + win, j:j + win] += s0 - s
            cnt[i:i + win, j:j + win] += 1
        M = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0) * region
        pos = np.clip(M, 0, None)
        heat_in = round(float(pos[small].sum() / pos.sum()), 3) if pos.sum() > 0 else None
        fa = self._deletion_region(tile, valid, M, region, mask_t, small_t, seed) if faith else None
        # display: token map -> smooth pixels in the searched area, signed colours, soft edge
        from skimage.transform import resize
        up = resize(M, (self.TILE, self.TILE), order=1, preserve_range=True)
        reg_px = resize(region.astype(np.float32), (self.TILE, self.TILE), order=1, preserve_range=True)
        ref = float(np.percentile(np.abs(M[region]), 99)) if region.any() and np.abs(M[region]).max() > 0 else 0.0
        v = np.clip(up / ref, -1, 1) * np.clip(reg_px, 0, 1) if ref > 0 else up * 0
        bbox, crop = self._tile_to_image(v, valid, geo, cell.shape, pad=0.02, box=reg_px > 0.05)
        return dict(bbox=bbox, png=_png(_diverging(crop, np.abs(crop) > 0.02)), score=round(float(s0), 4),
                    heat_inside_cell=heat_in, areas_tested=len(starts), faithfulness=fa, tile_scale=round(rs, 4),
                    layer=f"occlusion of {win * ps} x {win * ps} px squares (in the segmenter's input scale)")

    def _deletion_region(self, tile, valid, M, region, mask_t, small_t, seed, steps=(0.05, 0.1, 0.2, 0.3), n_random=3):
        """deletion check for a signed map inside a searched region: blur the region's most supporting tokens first
        vs random tokens of the same region (same counts), scored with the segmenter's cell probability."""
        from scipy.ndimage import gaussian_filter
        torch = self.torch
        ps, g = self.net.ps, self.TILE // self.net.ps
        blur = gaussian_filter(tile, 6)
        idx = np.flatnonzero(region.ravel())
        order = idx[np.argsort(-M.ravel()[idx], kind="stable")]
        rng = np.random.default_rng(seed)
        orders = [order] + [rng.permutation(idx) for _ in range(n_random)]
        imgs = [tile]
        for od in orders:
            for f in steps:
                tm = np.zeros(g * g, bool)
                tm[od[:max(1, int(round(f * len(idx))))]] = True
                imgs.append(np.where(np.kron(tm.reshape(g, g), np.ones((ps, ps), bool)) & valid, blur, tile))
        sc = []
        with self.lock, torch.no_grad():
            for i in range(0, len(imgs), 8):
                y, n = self._head(self._trunk(imgs[i:i + 8]))
                sc += self._target(y, n, mask_t, small_t, "seg", None).float().cpu().tolist()
        sc = np.asarray(sc, np.float64)
        s0, rest = sc[0], sc[1:].reshape(len(orders), len(steps))
        drop = s0 - rest
        return dict(score_unchanged=round(float(s0), 4), steps=list(steps), curve_heatmap=np.round(rest[0], 4).tolist(),
                    curve_random=np.round(rest[1:].mean(0), 4).tolist(), aopc_heatmap=round(float(drop[0].mean()), 4),
                    aopc_random=round(float(drop[1:].mean()), 4),
                    scored_with="the segmenter's cell probability for this cell", within="the area around the cell")

    def _tile_to_image(self, v, valid, geo, shape, pad=0.0, box=None):
        """tile-resolution map -> image pixels (bilinear): the whole tile area, or the box around `box` (a tile
        mask, e.g. the cell) widened by `pad`."""
        from skimage.transform import resize
        y0, x0, fy, fx = geo
        H0, W0 = shape
        t = self.TILE
        vv = np.where(valid, v, 0.0)
        ys, xs = np.nonzero(box) if box is not None else (None, None)
        if box is not None and len(ys):
            ty0, ty1, tx0, tx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            py, px = int((ty1 - ty0) * pad) + 2, int((tx1 - tx0) * pad) + 2
            ty0, ty1, tx0, tx1 = max(0, ty0 - py), min(t, ty1 + py), max(0, tx0 - px), min(t, tx1 + px)
        else:
            ty0, ty1, tx0, tx1 = 0, t, 0, t
        ox0, oy0 = max(0, int(np.floor((x0 + tx0) / fx))), max(0, int(np.floor((y0 + ty0) / fy)))
        ox1, oy1 = min(W0, int(np.ceil((x0 + tx1) / fx))), min(H0, int(np.ceil((y0 + ty1) / fy)))
        a0, a1 = int(round(oy0 * fy - y0)), int(round(oy1 * fy - y0))
        b0, b1 = int(round(ox0 * fx - x0)), int(round(ox1 * fx - x0))
        crop = vv[max(0, a0):min(t, a1), max(0, b0):min(t, b1)]
        crop = resize(crop, (max(1, oy1 - oy0), max(1, ox1 - ox0)), order=1, preserve_range=True)
        return [ox0, oy0, ox1, oy1], crop

    def _deletion_cell(self, tile, valid, W, small, scorer, seed, steps=(0.1, 0.2, 0.3, 0.5), n_random=3):
        """Deletion inside the cell, scored with the real tree model: blur the cell's patches that push most
        towards the shown state first, vs the cell's patches in random order (same counts)."""
        from scipy.ndimage import gaussian_filter
        torch = self.torch
        ps, g = self.net.ps, self.TILE // self.net.ps
        blur = gaussian_filter(tile, 6)
        idx = np.flatnonzero(small.ravel())
        order = idx[np.argsort(-W.ravel()[idx], kind="stable")]
        rng = np.random.default_rng(seed)
        orders = [order] + [rng.permutation(idx) for _ in range(n_random)]
        imgs = [tile]
        for od in orders:
            for f in steps:
                k = max(1, int(round(f * len(idx))))
                tm = np.zeros(g * g, bool)
                tm[od[:k]] = True
                pm = np.kron(tm.reshape(g, g), np.ones((ps, ps), bool))
                imgs.append(np.where(pm & valid, blur, tile))
        st = torch.from_numpy(small.astype(np.float32)).to(self.dev)
        es = []
        with self.lock, torch.no_grad():
            for i in range(0, len(imgs), 8):
                _, n = self._head(self._trunk(imgs[i:i + 8]))
                es.append(((n.float() * st).sum((2, 3)) / st.sum()).cpu().numpy())
        sc = np.asarray(scorer(np.concatenate(es)), np.float64)
        s0, rest = sc[0], sc[1:].reshape(len(orders), len(steps))
        drop = s0 - rest
        return dict(score_unchanged=round(float(s0), 4), steps=list(steps), curve_heatmap=np.round(rest[0], 4).tolist(),
                    curve_random=np.round(rest[1:].mean(0), 4).tolist(), aopc_heatmap=round(float(drop[0].mean()), 4),
                    aopc_random=round(float(drop[1:].mean()), 4),
                    scored_with="the real cell-state tree model (other features kept)", within="the cell's own patches")

    def _deletion(self, tile, valid, cam, mask_t, small_t, kind, ge_t, scorer, seed, steps=(0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
                  n_random=3):
        """Deletion test: replace token areas (8 x 8 px) by a blurred copy, most important first vs random order.
        Score: the same target (seg) or the REAL tree model via scorer(encoder vectors) (state)."""
        from scipy.ndimage import gaussian_filter
        torch = self.torch
        ps, g = self.net.ps, self.TILE // self.net.ps
        blur = gaussian_filter(tile, 6)
        order_cam = np.argsort(-cam.ravel(), kind="stable")
        rng = np.random.default_rng(seed)
        orders = [("heatmap", order_cam)] + [("random", rng.permutation(g * g)) for _ in range(n_random)]
        imgs = [tile]
        for _, od in orders:
            for f in steps:
                k = max(1, int(round(f * g * g)))
                tm = np.zeros((g, g), bool)
                tm.ravel()[od[:k]] = True
                pm = np.kron(tm, np.ones((ps, ps), bool))
                imgs.append(np.where(pm & valid, blur, tile))
        scores, es = [], []
        with torch.no_grad():
            for i in range(0, len(imgs), 8):
                y, n = self._head(self._trunk(imgs[i:i + 8]))
                if kind == "seg" or scorer is None:
                    scores += self._target(y, n, mask_t, small_t, kind, ge_t).float().cpu().tolist()
                else:
                    es.append(((n.float() * small_t).sum((2, 3)) / small_t.sum()).cpu().numpy())
        if kind != "seg" and scorer is not None:
            scores = list(scorer(np.concatenate(es)))
        scores = np.asarray(scores, np.float64)
        s0, rest = scores[0], scores[1:].reshape(len(orders), len(steps))
        drop = s0 - rest
        a_cam, a_rand = float(drop[0].mean()), float(drop[1:].mean())
        return dict(score_unchanged=round(float(s0), 4), steps=list(steps), curve_heatmap=np.round(rest[0], 4).tolist(),
                    curve_random=np.round(rest[1:].mean(0), 4).tolist(), aopc_heatmap=round(a_cam, 4),
                    aopc_random=round(a_rand, 4),
                    scored_with=("the segmenter's cell-probability (same target)" if kind == "seg" else
                                 "the real cell-state tree model (other features kept)" if scorer is not None
                                 else "the stand-in head"))

    # ------------------------------------------------------------------ validation (sanity checks of the heatmaps)
    def _tile_labels(self, lab, geo):
        """the whole label image mapped onto the tile pixels (same nearest-pixel mapping as the cell mask)."""
        y0, x0, fy, fx = geo
        H0, W0 = lab.shape
        t = self.TILE
        oy = np.floor((y0 + np.arange(t) + 0.5) / fy).astype(int)
        ox = np.floor((x0 + np.arange(t) + 0.5) / fx).astype(int)
        oky, okx = (oy >= 0) & (oy < H0), (ox >= 0) & (ox < W0)
        L = np.zeros((t, t), np.int32)
        L[np.ix_(oky, okx)] = np.asarray(lab)[np.ix_(oy[oky], ox[okx])]
        return L

    def _occlusion(self, tile, valid, mask_t, small_t, kind, ge_t, b=2, batch=None):
        """reference map, faithful by construction: blur one block of b x b tokens at a time and record how much the
        target drops. (g/b, g/b) map; blocks that are only padding are NaN."""
        from scipy.ndimage import gaussian_filter
        torch, ps, g = self.torch, self.net.ps, self.TILE // self.net.ps
        blur = gaussian_filter(tile, 6)
        gb = g // b
        batch = batch or (32 if self.dev.type == "cuda" else 8)
        imgs, idx = [tile], []
        for i in range(gb):
            for j in range(gb):
                pm = np.zeros(tile.shape, bool)
                pm[i * b * ps:(i + 1) * b * ps, j * b * ps:(j + 1) * b * ps] = True
                pm &= valid
                if pm.any():
                    imgs.append(np.where(pm, blur, tile))
                    idx.append((i, j))
        sc = []
        with torch.no_grad():
            for k in range(0, len(imgs), batch):
                y, n = self._head(self._trunk(imgs[k:k + batch]))
                sc += self._target(y, n, mask_t, small_t, kind, ge_t).float().cpu().tolist()
        occ = np.full((gb, gb), np.nan)
        for (i, j), s in zip(idx, sc[1:]):
            occ[i, j] = sc[0] - s
        return occ

    @contextlib.contextmanager
    def _randomized(self, parts, seed=0):
        """model-parameter randomisation test (Adebayo et al. 2018): temporarily replace the top of the network by
        random weights with the same scale. parts: 'block' (last ViT block) and/or 'head' (neck + output layer)."""
        torch, net = self.torch, self.net
        enc = net.encoder
        gen = torch.Generator().manual_seed(seed)

        def rnd(mod):
            m2 = copy.deepcopy(mod)
            with torch.no_grad():
                for p in m2.parameters():
                    if p.ndim > 1:
                        p.copy_((torch.randn(p.shape, generator=gen) * float(p.float().std())).to(p.device, p.dtype))
                    elif p.numel() > 1:
                        p.copy_(p[torch.randperm(p.numel(), generator=gen).to(p.device)])
            return m2
        saved = {}
        try:
            if "block" in parts:
                saved["block"] = enc.blocks[-1]
                enc.blocks[-1] = rnd(saved["block"])
            if "head" in parts:
                saved["neck"], saved["out"] = enc.neck, net.out
                enc.neck, net.out = rnd(enc.neck), rnd(net.out)
            yield
        finally:
            if "block" in saved:
                enc.blocks[-1] = saved["block"]
            if "neck" in saved:
                enc.neck, net.out = saved["neck"], saved["out"]

    def check(self, img, lab, cids, diameter, seed=0, keep_maps=6):
        """Sanity checks of the segmentation heatmap (kind 'seg') for some cells of one frame.
        Per cell: localisation (heat on the cell vs its share of the area; peak on the cell), faithfulness
        (deletion vs random; agreement with an occlusion map), target specificity (map for a neighbour cell in the
        same view), model randomisation (map after randomising the last block / last block + head)."""
        from scipy.ndimage import distance_transform_edt, binary_dilation, sobel
        torch = self.torch
        lab = np.asarray(lab)
        x, rs = self._prep(img, diameter)
        g, ps = self.TILE // self.net.ps, self.net.ps
        out, maps = [], []
        for cid in cids:
            cell = lab == int(cid)
            if not cell.any():
                continue
            tile, valid, mask, geo = self._tile(x, cell.shape, cell, rs)
            vs, small = self._small(valid, g), self._small(mask, g)
            Lt = self._tile_labels(lab, geo)
            T = lambda a: torch.from_numpy(a.astype(np.float32)).to(self.dev)
            mask_t, small_t = T(mask), T(small)
            # neighbour: the other cell with the most pixels in this view (at least one token of its own)
            ids, cnt = np.unique(Lt[(Lt > 0) & (Lt != int(cid)) & valid], return_counts=True)
            nb = None
            for j in np.argsort(-cnt):
                mb = Lt == ids[j]
                sb = self._small(mb, g)
                if (sb & ~small).sum() >= 1:
                    nb = (int(ids[j]), mb, sb)
                    break
            with self.lock:
                cam, s0 = self._cam(tile, valid, mask_t, small_t, "seg", None)
                fa = self._deletion(tile, valid, cam, mask_t, small_t, "seg", None, None, seed)
                occ = self._occlusion(tile, valid, mask_t, small_t, "seg", None)
                camB = None
                if nb is not None:
                    camB, _ = self._cam(tile, valid, T(nb[1]), T(nb[2]), "seg", None)
                rnd = {}
                for name, parts in (("block", ("block",)), ("block+head", ("block", "head"))):
                    with self._randomized(parts, seed):
                        rnd[name], _ = self._cam(tile, valid, mask_t, small_t, "seg", None)
            # ---- metrics
            tot = cam[vs].sum()
            share = small[vs].mean()
            near = binary_dilation(small, iterations=1)
            heat_in = float(cam[small].sum() / tot) if tot > 0 else None
            peak = np.unravel_index(np.argmax(np.where(vs, cam, -1)), cam.shape) if tot > 0 else None
            b = g // occ.shape[0]
            pool = lambda a: a.reshape(occ.shape[0], b, occ.shape[1], b).sum((1, 3))
            okb = ~np.isnan(occ)
            dist = -distance_transform_edt(~small)
            edges = np.hypot(sobel(tile, 0), sobel(tile, 1)).reshape(g, ps, g, ps).mean((1, 3))
            r = dict(cell=int(cid), score=round(s0, 3), tile_scale=round(rs, 3),
                     heat_on_cell=None if heat_in is None else round(heat_in, 3), cell_area_share=round(float(share), 3),
                     lift=None if heat_in is None or share == 0 else round(heat_in / float(share), 2),
                     peak_on_cell=None if peak is None else bool(near[peak]),
                     deletion_heatmap=fa["aopc_heatmap"], deletion_random=fa["aopc_random"],
                     deletion_better=fa["aopc_heatmap"] > fa["aopc_random"] + 0.02 * abs(fa["score_unchanged"]) + 0.005,
                     rho_occlusion=_spear(pool(cam)[okb], occ[okb]),
                     rho_distance_prior=_spear(pool(dist * vs)[okb], occ[okb]),
                     rho_edges=_spear(cam[vs], edges[vs]),
                     neighbour=None if nb is None else nb[0],
                     rho_neighbour=None if camB is None else _spear(cam[vs], camB[vs]),
                     own_cell_preferred=None if camB is None else bool(
                         _frac(cam, small) > _frac(camB, small) and _frac(camB, nb[2]) > _frac(cam, nb[2])),
                     rho_random_block=_spear(cam[vs], rnd["block"][vs]),
                     rho_random_block_head=_spear(cam[vs], rnd["block+head"][vs]))
            out.append(r)
            if len(maps) < keep_maps:
                maps.append(dict(cell=int(cid), tile=tile, valid=valid, mask=mask, nb_mask=None if nb is None else nb[1],
                                 cam=cam, occ=occ, camB=camB, rnd=rnd["block+head"]))
        return out, maps


def state_scorer(model, xai: Explanations, row: int, key: str, sign: float):
    """encoder vectors (n, 256) -> the REAL tree model's pos-vs-neg log-odds for this cell's row (sign-adjusted),
    other features unchanged, averaged over the image-type assumptions as in the predictions."""
    T = TARGETS[key]
    head = model.heads[T["head"]]
    names, enc_idx = xai.names, xai.d["enc_idx"]
    x_row = xai.d["X"][row].astype(np.float64)
    mods = [str(m) for m in xai.d["mods"]]

    def f(E):
        Z = model.pca.transform(np.asarray(E, np.float64))
        X = np.repeat(x_row[None], len(Z), 0)
        X[:, enc_idx] = Z
        outs = []
        for md in mods:
            Xm = X.copy()
            for m2 in ("dic", "qpi", "phase"):
                if f"is_{m2}" in names:
                    Xm[:, names.index(f"is_{m2}")] = float(m2 == md)
            outs.append(head_raw_contrast(head, Xm, T["pos"], T["neg"]))
        return sign * np.mean(outs, 0)
    return f


# ----------------------------------------------------------------------------- validation report
CHECKS = [  # key, title, what it tests, rule
    ("localisation", "Localisation", "Is the heat on the cell that is explained?",
     "median lift ≥ 2 (heat on the cell ÷ its share of the view) and the peak on or next to the cell in ≥ 70% of cells"),
    ("deletion", "Faithfulness: deletion", "Does blurring the hottest areas first lower the score faster than random?",
     "heatmap order beats random order in ≥ 70% of cells"),
    ("occlusion", "Faithfulness: occlusion agreement", "Does the map agree with an occlusion map (blur each area, "
     "measure the drop) — slow but faithful by construction?", "median Spearman ρ ≥ 0.3"),
    ("specificity", "Target specificity", "Does the map change when a neighbouring cell is explained instead?",
     "each map prefers its own cell in ≥ 70% of cases and median ρ between the two maps < 0.8"),
    ("randomisation", "Model randomisation (Adebayo et al. 2018)", "Does the map depend on the trained weights? "
     "(last block + head replaced by random weights)", "median ρ with the original map < 0.5"),
]


def _med(v):
    v = [x for x in v if x is not None]
    return round(float(np.median(v)), 3) if v else None


def _pct(v):
    v = [bool(x) for x in v if x is not None]
    return round(100 * sum(v) / len(v)) if v else None


def check_summary(rows):
    g = lambda k: [r.get(k) for r in rows]
    lift, peak = _med(g("lift")), _pct(g("peak_on_cell"))
    dele = _pct(g("deletion_better"))
    occ, prior = _med(g("rho_occlusion")), _med(g("rho_distance_prior"))
    own, rnb = _pct(g("own_cell_preferred")), _med(g("rho_neighbour"))
    rr1, rr2 = _med(g("rho_random_block")), _med(g("rho_random_block_head"))
    ok = lambda c: None if c is None else bool(c)
    empty = sum(r.get("heat_on_cell") is None for r in rows)
    res = {
        "localisation": dict(value=f"median lift {lift} · peak on the cell in {peak}% · median heat on the cell "
                                   f"{_med(g('heat_on_cell'))} (cell = {_med(g('cell_area_share'))} of the view)",
                             passed=None if lift is None else bool(lift >= 2 and (peak or 0) >= 70)),
        "deletion": dict(value=f"better than random in {dele}% of cells", passed=ok(dele is not None and dele >= 70)),
        "occlusion": dict(value=f"median ρ {occ} (a map that is only 'distance to the cell': ρ {prior})",
                          passed=None if occ is None else bool(occ >= 0.3)),
        "specificity": dict(value=f"own cell preferred in {own}% · median ρ between the two maps {rnb}",
                            passed=None if rnb is None else bool((own or 0) >= 70 and rnb < 0.8)),
        "randomisation": dict(value=f"median ρ {rr2} (last block only: {rr1})",
                              passed=None if rr2 is None else bool(rr2 < 0.5)),
    }
    res["_info"] = dict(cells=len(rows), rho_edges=_med(g("rho_edges")), empty_maps=empty)
    return res


def check_figure(maps):
    """PNG (base64): one row per cell — image with the cell (red) and its neighbour (blue), Grad-CAM++, occlusion
    reference, Grad-CAM++ for the neighbour, Grad-CAM++ after randomising the model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    from skimage.transform import resize
    if not maps:
        return None
    cols = ["image · cell (red), neighbour (blue)", "Grad-CAM++", "occlusion (reference)",
            "Grad-CAM++ for the neighbour", "after randomising the model"]
    fig, ax = plt.subplots(len(maps), 5, figsize=(13, 2.7 * len(maps)), squeeze=False)
    for i, m in enumerate(maps):
        t = m["tile"]
        lo, hi = np.percentile(t[m["valid"]], [1, 99]) if m["valid"].any() else (0, 1)
        base = np.clip((t - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = np.dstack([base] * 3)
        for msk, col in ((m["mask"], (1, .2, .2)), (m["nb_mask"], (.2, .5, 1))):
            if msk is not None:
                rgb[find_boundaries(msk, mode="outer")] = col
        for j in range(5):
            a = ax[i, j]
            a.imshow(rgb if j == 0 else base, cmap="gray", vmin=0, vmax=1)
            mp = [None, m["cam"], m["occ"], m["camB"], m["rnd"]][j]
            if mp is not None:
                v = np.nan_to_num(np.asarray(mp, np.float64), nan=0.0)
                v = np.clip(v, 0, None)
                v = v / v.max() if v.max() > 0 else v
                up = resize(v, t.shape, order=1)
                up[~m["valid"]] = 0
                a.imshow(up, cmap="turbo", alpha=np.clip(up, 0, 1) * 0.65, vmin=0, vmax=1)
                if j:
                    a.contour(m["mask"] if j != 3 or m["nb_mask"] is None else m["nb_mask"], [0.5],
                              colors=["w"], linewidths=0.8)
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(cols[j], fontsize=9)
        ax[i, 0].set_ylabel(f"cell {m['cell']}", fontsize=9)
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def check_html(title, rows, summ, fig_b64, note=""):
    import html as H
    badge = lambda p: ('<span class="b na">n/a</span>' if p is None else
                       '<span class="b ok">PASS</span>' if p else '<span class="b no">FAIL</span>')
    trs = "".join(f"<tr><td>{badge(summ[k]['passed'])}</td><td><b>{H.escape(t)}</b><br><small>{H.escape(q)}</small></td>"
                  f"<td>{H.escape(summ[k]['value'])}</td><td><small>{H.escape(rule)}</small></td></tr>"
                  for k, t, q, rule in CHECKS)
    keys = ["cell", "heat_on_cell", "cell_area_share", "lift", "peak_on_cell", "deletion_heatmap", "deletion_random",
            "rho_occlusion", "rho_distance_prior", "neighbour", "rho_neighbour", "own_cell_preferred",
            "rho_random_block", "rho_random_block_head", "rho_edges", "frame"]
    keys = [k for k in keys if any(k in r for r in rows)]
    cell_rows = "".join("<tr>" + "".join(f"<td>{H.escape(str(r.get(k, '')))}</td>" for k in keys) + "</tr>" for r in rows)
    n_pass = sum(bool(summ[k]["passed"]) for k, *_ in CHECKS)
    return f"""<!doctype html><meta charset="utf-8"><title>{H.escape(title)}</title>
<style>body{{font:14px/1.5 system-ui,Segoe UI,sans-serif;margin:24px;max-width:1200px;color:#1d1b2c}}
h1{{font-size:22px;margin:0 0 4px}}table{{border-collapse:collapse;width:100%;margin:12px 0}}
td,th{{border-bottom:1px solid #e3e0f0;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#f4f2fb;font-size:12px}}small{{color:#6b6787}}.b{{font-weight:700;padding:2px 8px;border-radius:999px;font-size:12px}}
.ok{{background:#dff5e7;color:#16733a}}.no{{background:#fde2e2;color:#a11a1a}}.na{{background:#eee;color:#666}}
.cells td{{font-size:12px;font-variant-numeric:tabular-nums}}.sum{{font-size:16px;margin:8px 0}}</style>
<h1>{H.escape(title)}</h1><div class="sum"><b>{n_pass} of {len(CHECKS)}</b> checks passed on {summ['_info']['cells']} cells.
{H.escape(note)}</div>
{f'<p class="sum" style="color:#a11a1a"><b>{summ["_info"]["empty_maps"]} of {summ["_info"]["cells"]} maps were empty</b> '
 f'(no area pushes the score up): an empty map explains nothing and counts as a failure of the method for that cell.</p>'
 if summ["_info"].get("empty_maps") else ''}
<table><tr><th></th><th>check</th><th>result</th><th>rule</th></tr>{trs}</table>
<p><small>Correlation of the map with plain image edges (Sobel): median ρ {summ['_info']['rho_edges']} — a value near 1
would mean the map mostly follows edges in the image. Thresholds are practical rules of thumb, not statistical tests.</small></p>
{f'<img src="data:image/png;base64,{fig_b64}" style="width:100%;border:1px solid #e3e0f0;border-radius:8px">' if fig_b64 else ''}
<h3>Per cell</h3><table class="cells"><tr>{''.join(f'<th>{k}</th>' for k in keys)}</tr>{cell_rows}</table>"""