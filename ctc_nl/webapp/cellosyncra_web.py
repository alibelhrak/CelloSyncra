#!/usr/bin/env python
"""CelloSyncra web app: upload a 2D or 3D time-lapse, ask questions in plain English, see the answer.

Runs on the GPU node next to the Qwen server (see `sbatch run_hpc.sh web`) and reuses
ctc_nl_pipeline.py unchanged:
  upload -> Cellpose-SAM joint 2D+3D generalist (segmentation) -> IoU tracker -> cell table
  question -> Qwen2.5-72B planner (tool calls on the cell table) -> answer drawn on the image

Two layers (--model-nuclei given), fully automatic:
  "cells"  = whole cells (the main model)      "nuclei" = nuclei (Cellpose-SAM trained on ALFI DIC)
  After the whole cells of a 2D upload are found, the nucleus model is tried on two frames (nuclei_check):
  if its nuclei make sense (inside the cells, about one per cell, clearly smaller than the cell) the
  nuclei layer is built in the background; otherwise (e.g. not a DIC-like image) the video keeps whole
  cells only and answers say why. Old uploads get the same check when the server starts.
  For each question Qwen picks the layer (tool use_layer) following the rules in LAYER_NOTE; the user
  can fix the layer instead; every answer says which layer it used.

    python cellosyncra_web.py --out <CTC_OUT> --data <CTC_DATA> --work <upload folder> \
        --llm-url http://localhost:8123/v1 --port 8901

Open the printed URL (it contains an access token) through an SSH tunnel.
"""
from __future__ import annotations

import argparse
import csv
import io
import contextlib
import json
import math
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import uuid
import zipfile
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ctc_nl_pipeline as P  # noqa: E402
from cs_tracks import clean_divisions, res_track_text  # noqa: E402

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response  # noqa: E402

CFG: argparse.Namespace = None          # filled in main()

# Extra instructions for the planner in the web app (the benchmark prompt itself is unchanged).
WEB_NOTE = """

You are answering a person in the CelloSyncra web app.
- If the question asks for something the tools cannot measure (for example whether a cell is cancerous, healthy,
  its cell type, genes, or a diagnosis), do NOT guess and do not keep calling tools: call
  submit_answer right away with cell_ids=[] and count=0, and a short note saying that this cannot be measured here
  and what can be measured instead (size, shape, position, brightness, depth in 3D, tracks and divisions).
- If the question asks for a table, list or the measurements of cells, select those cells and call submit_answer
  with them; the app shows their measurements as a table, so you do not need to read every cell with get_cell.
- If no frame is named, use the frame the user is looking at or the first frame, and say which one in the note.
- Put a one-sentence summary of the answer in note."""
LAYER_NOTE = """

This video has TWO outline layers. Before the other tools, choose the layer with use_layer:
- "nuclei" (nuclei found by a model trained on DIC images): for questions about nuclei (nuclear size, shape,
  number of nuclei), for COUNTING cells, and for DIVISIONS, lineage and following objects over time
  (nuclei rarely touch, so counts and tracks are more reliable).
- "cells" (whole cells, the default): for cell size, area, shape, spreading, elongation, outlines, brightness
  and anything else. Cell size and cell shape ALWAYS come from the whole cells, never from nuclei.
If unsure, use "cells". Rules for the nuclei layer:
- Its features are named nucleus_* (nucleus_area ...). Use them only when the question is about nuclei.
- Its ids are NUCLEUS ids: nucleus 7 is not cell 7. Each nucleus has host_cell = the whole cell holding it.
  Call them nuclei in note, and give host cells when the user asks about cells.
- Its divisions are NUCLEAR divisions; each event says cell_divided (True = the cell split too, False = one cell
  with two nuclei). Say this in note, e.g. "4 nuclear divisions, 3 with the cell splitting".
- A count by nuclei counts nuclei (one per cell for most cells); say "counted by nuclei".
The whole-cell layer has n_nuclei per cell (2 or more = multinucleated).
Say in note which layer you used. Describe what is measured; never infer disease, cell type or cell state."""

STATE_NOTE = """

Cell states (cell_states tool): a trained model flags what each cell is DOING. Call cell_states for any question
about abnormal, dying or dead cells, apoptosis, necrosis, cell death, or which cells divided / are dividing.
- In CelloSyncra "abnormal" means abnormal BEHAVIOUR: signs of dying (apoptosis-like or necrosis-like).
  It never means cancerous, precancerous, malignant, diseased or unhealthy: if asked about these, say that
  CelloSyncra describes cell behaviour and cannot determine that, and offer the behaviour it can report.
- Each kind of state is only available for the image types it was validated on; the tool says which are
  available for this video and why not. If something is not available, say so with that reason; never guess.
- Dying times are the ONSET of visible signs of dying, not a confirmed time of death; say so, and that the
  minutes assume the frame interval the tool reports.
- Answer in your own words with the SAME numbers as the tool's summary_text (never your own count). Mention at
  most 3-5 example cells. The page adds the full table, a timeline and per-cell onset frames, so do not list them.
- Give cell ids from the tool; to show cells on the image, select them (they are whole-cell ids) and submit.
- For "why" questions about one cell's state use explain_cell_state: name the 2-3 largest factors with their
  direction (toward or away from the state), say they are contributions of the model (SHAP), not causes in the
  biology, and select that cell.
- If cell_states says states are not available (or still being computed), submit_answer with cell_ids [] and
  count 0 and the reason as the note: do not select or list cells for that answer."""
CELL_STATES_TOOL = {"name": "cell_states",
                    "description": "What cells are DOING (research flags, not diagnoses): which states are available for "
                                   "this video, counts at the current frame, cells with signs of dying (onset frame, "
                                   "apoptosis- or necrosis-like) and cells seen dividing. state (optional): interphase | "
                                   "mitosis | alive | apoptosis | necrosis selects the cells in that state at the current "
                                   "frame. cell_ids (optional): limit the per-cell events to these cells.",
                    "input_schema": {"type": "object", "properties": {
                        "state": {"enum": ["interphase", "mitosis", "alive", "apoptosis", "necrosis"]},
                        "cell_ids": {"type": "array", "items": {"type": "integer"}}}}}
EXPLAIN_TOOL = {"name": "explain_cell_state",
                "description": "WHY the cell-state model flags one cell (signs of dying / type of dying on QPI, dividing on "
                               "DIC or phase): exact SHAP contributions of the model's features, grouped into readable "
                               "factors (dry mass, density, size, shape, movement, image appearance ...). Use it for "
                               "'why is cell N dying / dividing?'. frame (optional): default the current frame.",
                "input_schema": {"type": "object", "properties": {"cell_id": {"type": "integer"},
                                                                   "frame": {"type": "integer"}},
                                 "required": ["cell_id"]}}
REPLY_TOOL = {"name": "reply",
              "description": "Answer the user in words when no cell selection is needed: questions about this "
                             "conversation or about what you did, what has been analysed in this video, what the app or "
                             "a method can do, explanations of an earlier answer, greetings and thanks. text: your "
                             "message (short, plain language; **bold** and '- ' lists are allowed). Numbers about the "
                             "cells must come from tool results or the VIDEO FACTS. cell_ids (optional): cells to "
                             "highlight in the current frame.",
              "input_schema": {"type": "object", "properties": {"text": {"type": "string"},
                                                                 "cell_ids": {"type": "array", "items": {"type": "integer"}}},
                               "required": ["text"]}}
ASK_TOOL = {"name": "ask_user",
            "description": "Ask the user ONE short clarifying question instead of answering, only when the request is "
                           "really ambiguous and a guess could give a misleading answer (e.g. which measurement, which "
                           "time span). options: 2-4 short answers the user can click.",
            "input_schema": {"type": "object", "properties": {"question": {"type": "string"},
                                                               "options": {"type": "array", "items": {"type": "string"}}},
                             "required": ["question"]}}
CHAT_NOTE = """

Talk with the user like a helpful research assistant: brief, friendly, precise, in plain language.
- Questions to measure or find cells: use the tools and finish with submit_answer. The note is your chat message
  to the user and is shown as the main text, so write it like a knowledgeable colleague talking to a biologist:
  2-4 sentences; answer the question directly first (yes / no and the key number), then one sentence of meaning
  or caution in biological terms, then, if useful, one short offer of a next step ("Want me to show when they
  started, or explain one of them?"). Use "I". No lists of ids, no copied tool text, no headings. Mention the
  frame only when it matters ("in the frame you are looking at").
- Questions about the conversation, about what you did or what was analysed ("did you analyse the whole video?",
  "how did you get that?", "what can you do?"), explanations, greetings and thanks: call reply. Use the VIDEO FACTS
  and the earlier turns; do not select cells or call submit_answer for these.
- Follow-up questions ("and in frame 30?", "which of them is the biggest?", "why?") refer to the earlier turns of
  this conversation: keep the same cells, measurement or frame unless the user changes them.
- If a request is truly ambiguous and a guess could mislead, call ask_user with one short question and 2-4 options.
  If a sensible default exists, use it and say so instead of asking.
- Never invent numbers: they come from tools or the VIDEO FACTS."""
HEATMAP_TOOL = {"name": "heatmap",
                "description": "Explain one cell with a heatmap (2D videos), drawn on the image for the user. "
                               "kind 'segmentation' (Grad-CAM++): the parts of the cell Cellpose-SAM relied on most to "
                               "find it (always available). kind 'cell_state' (patch attribution through the real "
                               "cell-state model): which parts of the cell push it towards its state (orange) or against "
                               "(blue) (only when cell states are available). Returns a deletion check (is the heatmap "
                               "better than random?) and, for cell_state, how well the map fits the model. One cell per "
                               "call, at most 4 per question: for 'all cells' pick a few (e.g. the selected, largest or "
                               "flagged ones) and say so, or ask which. frame: default the current frame.",
                "input_schema": {"type": "object", "properties": {
                    "cell_id": {"type": "integer"}, "kind": {"type": "string", "enum": ["segmentation", "cell_state"]},
                    "frame": {"type": "integer"}}, "required": ["cell_id"]}}
START_TOOL = {"name": "start_analysis",
              "description": "Start a background analysis for this video: 'cell_states' or 'nuclei' when VIDEO FACTS say "
                             "they were not computed yet; 'find_missed_cells' when the user shows a cell that was not "
                             "segmented (e.g. a box with no segmented cell, or a much bigger / smaller cell than the "
                             "others): it segments the video again in place, adding such cells (a few minutes; the chat "
                             "stays). Tell the user what started and how long it takes.",
              "input_schema": {"type": "object", "properties": {"what": {"type": "string", "enum": ["cell_states", "nuclei", "find_missed_cells"]}},
                               "required": ["what"]}}
CHAT_NOTE += """
- Answer EVERY part of the message. If a part cannot be done, say so plainly and offer what can be done.
- Explanations (XAI): use heatmap for heatmaps of a cell (segmentation: pixel-level Grad-CAM++; cell state: patch attribution), explain_cell_state for SHAP reasons of a cell's
  state. They work one cell at a time: for many cells, explain 2-3 representative ones (say which and why) and offer
  more. SHAP and the cell-state heatmap need cell states; the segmentation heatmap works for any 2D video.
- Export / download / save / table requests: ALWAYS call export_table (it adds the download button); never send the
  user to a menu instead, and never repeat an earlier answer that did not use it. Then finish with reply (not
  submit_answer): an export is not a cell selection, so do not highlight or list cells.
- After using a tool that draws something (heatmap), still finish with submit_answer (the cells involved) or reply,
  and describe in words what the heatmap and its deletion check say."""
TABLES = {"counts_per_frame": "number of cells, divisions, confluence and mean size in every frame",
          "divisions": "every division: parent cell, frame, daughter cells",
          "tracks": "every tracked cell: first and last frame, length, parent and daughters",
          "cells": "one row per cell per frame with all measurements",
          "cell_states": "per cell: dividing / signs of dying and when they start (when cell states exist)"}


EXPORT_TOOL = {"name": "export_table",
               "description": "Prepare a table the user can download (.csv), with a short summary for your answer. "
                              "table: counts_per_frame (cells, divisions, confluence %, mean size per frame; also "
                              "dividing / dying counts when cell states exist), divisions, tracks (one row per tracked "
                              "cell), cells (all measurements, one row per cell per frame), cell_states. Use it whenever "
                              "the user asks to export, download, save or tabulate something; a download button is "
                              "added to your answer. Then reply in words and summarise what is in it.",
               "input_schema": {"type": "object", "properties": {"table": {"type": "string", "enum": list(TABLES)}},
                                "required": ["table"]}}
for _t in (REPLY_TOOL, ASK_TOOL, HEATMAP_TOOL, START_TOOL, EXPORT_TOOL):
    if not any(t["name"] == _t["name"] for t in P.TOOLS):
        P.TOOLS.append(_t)
if not any(t["name"] == "cell_states" for t in P.TOOLS):
    P.TOOLS.insert(1, CELL_STATES_TOOL)
if not any(t["name"] == "explain_cell_state" for t in P.TOOLS):
    P.TOOLS.insert(2, EXPLAIN_TOOL)
LAYER_LABEL = {"cells": "whole cells", "nuclei": "nuclei (DIC)"}
USE_LAYER_TOOL = {"name": "use_layer",
                  "description": "Choose the outline layer for this question: 'cells' (whole cells) or 'nuclei' "
                                 "(nuclei, DIC videos only). Keeps the frame; the selection is reset to all objects "
                                 "of that frame in the chosen layer.",
                  "input_schema": {"type": "object", "properties": {"layer": {"enum": ["cells", "nuclei"]}},
                                   "required": ["layer"]}}
if not any(t["name"] == "use_layer" for t in P.TOOLS):
    P.TOOLS.insert(0, USE_LAYER_TOOL)
_BASE_PROMPT = P.system_prompt


def _web_prompt(sess):
    extra = LAYER_NOTE if getattr(sess, "route", "") == "auto" else ""
    if len(getattr(sess, "layers", {})) == 1 and getattr(sess, "nuclei_reason", ""):
        extra = ("\n\nThis video has whole-cell outlines only: " + sess.nuclei_reason +
                 " If the question asks about nuclei, say in note that the nuclei cannot be detected in this image type "
                 "(never say the cells have no nuclei), and answer with the whole cells if that still answers the "
                 "question (e.g. one nucleus per cell for counting); do not call use_layer.")
    fixed = getattr(sess, "route", "")
    if fixed in ("cells", "nuclei") and len(getattr(sess, "layers", {})) > 1:
        extra = (f"\n\nThe user chose the {LAYER_LABEL[fixed]} layer for this question; it is already active "
                 f"(do not call use_layer).")
    import cs_feedback
    facts = sess.__dict__.get("video_facts") or ""
    return (_BASE_PROMPT(sess) + WEB_NOTE + STATE_NOTE + CHAT_NOTE + extra + (("\n\nVIDEO FACTS\n" + facts) if facts else "") +
            cs_feedback.FeedbackStore.prompt_block(sess.__dict__.get("fewshot") or []))


P.system_prompt = _web_prompt
app = FastAPI(title="CelloSyncra", docs_url=None, redoc_url=None, openapi_url=None)


# =============================================================================
# access token (the node port is reachable by other cluster users)
# =============================================================================
def require_token(request: Request):
    tok = request.headers.get("x-token") or request.query_params.get("token")
    if not CFG.token or (tok and secrets.compare_digest(tok, CFG.token)):
        return True
    raise HTTPException(401, "missing or wrong access token: open the URL printed in the job log")


AUTH = [Depends(require_token)]


# =============================================================================
# sequences: uploads (seq folder "01") and the CTC test videos (seq "02", existing segmentation)
# =============================================================================
LOCK = threading.RLock()                 # protects SEQS / meta files
SEQS: dict[str, dict] = {}               # id -> meta
SESS: dict[str, tuple] = {}              # id -> (Session, Lock)
JOBS: deque = deque()                    # segmentation queue (ids)
JOB_EVENT = threading.Event()
BACKEND_NAMES = {"cpsam_joint": "our AI (Cellpose-SAM 2D+3D)", "oracle": "ground truth"}


def seq_root(sid):
    return Path(CFG.work) / "seqs" / sid


LIB_KEEP = ("layers", "res_dir_nuclei", "nuclei_per_frame", "divisions_nuclei", "diameter_nuclei", "nuclei_check",
            "states_check", "diameter", "n_frames", "frames")


def save_meta(m):
    if m["kind"] == "library":            # example videos: only what was added on top (nuclei, states) is kept
        d = seq_root(m["id"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "lib_meta.json").write_text(json.dumps({k: m[k] for k in LIB_KEEP if k in m}, indent=1, default=float))
        return
    if m["kind"] != "upload":
        return
    p = seq_root(m["id"]) / "meta.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=1, default=float))
    tmp.replace(p)


def load_uploads():
    for p in sorted((Path(CFG.work) / "seqs").glob("*/meta.json")):
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        if m.get("status") in ("queued", "segmenting", "converting", "sizing"):   # interrupted by a restart -> redo
            m["status"] = "queued"
            JOBS.append(m["id"])
        SEQS[m["id"]] = m
        st = (m.get("nuclei_check") or {}).get("state")
        if m.get("status") == "ready" and "nuclei" not in (m.get("layers") or []) and st in (None, "pending", "running"):
            queue_nuclei(m, force=bool((m.get("nuclei_check") or {}).get("force")))
        if m.get("status") == "ready" and (m.get("states_check") or {}).get("state") in (None, "pending", "running"):
            queue_states(m)
        if m.get("ndim") == 3 and m.get("spacing"):
            P.VOXEL_UM[m["ds_name"]] = tuple(m["spacing"])
    if JOBS:
        JOB_EVENT.set()


def library():
    """CTC test videos (sequence 02) that already have a segmentation in CTC_OUT."""
    out = []
    if not CFG.data:
        return out
    try:
        ds = P.find_datasets(CFG.data)
    except Exception:
        return out
    for name, d in sorted(ds.items()):
        if not (d / "02").is_dir():
            continue
        for be in ("cpsam_joint", "oracle"):
            rd = P.res_dir(CFG.out, be, name, "02")
            if rd.is_dir() and any(rd.glob("*.tif")):
                sid = f"lib-{be}-{name}"
                if sid not in SEQS:
                    m = dict(id=sid, kind="library", name=f"{name} (test video 02)", ds_name=name,
                             ds_dir=str(d), seq="02", res_dir=str(rd), backend=be, status="ready",
                             created=0, ndim=3 if "3D" in name else 2)
                    f = seq_root(sid) / "lib_meta.json"
                    if f.exists():
                        try:
                            m.update(json.loads(f.read_text()))
                        except Exception:
                            pass
                    for k in ("nuclei_check", "states_check"):      # a restart interrupted a job
                        if (m.get(k) or {}).get("state") in ("pending", "running"):
                            m.pop(k)
                    SEQS[sid] = m
                    if m["ndim"] == 2:
                        _lib_extras(m)
                out.append(SEQS[sid])
    return out


def lib_skip_reason(m, what):
    """Example videos where a model does not apply (known from the dataset name)."""
    if m.get("kind") != "library":
        return ""
    if m["ds_name"].startswith("Fluo-"):
        if what == "nuclei":
            return "these are fluorescence images of stained nuclei: the outlines already are the nuclei"
        return ("fluorescence images: the cell-state model was trained on label-free images "
                "(DIC, phase contrast, quantitative phase)")
    return ""


def _lib_extras(m):
    """2D example videos get the nuclei layer and cell states like uploads (in the background, once)."""
    if not m.get("diameter"):
        m["diameter"] = _diameter_from_masks(m)
    if CFG.model_nuclei and not m.get("nuclei_check") and not lib_skip_reason(m, "nuclei"):
        queue_nuclei(m)
    if CFG.model_states and not m.get("states_check") and not lib_skip_reason(m, "states"):
        queue_states(m)


def _diameter_from_masks(m):
    """typical cell diameter (px) of an existing segmentation: median equivalent diameter, first frame."""
    try:
        seg = P.SegResult(Path(m["res_dir"]))
        lab = np.asarray(seg.label(sorted(seg.files)[0]))
        _, n = np.unique(lab[lab > 0], return_counts=True)
        return float(round(np.median(2 * np.sqrt(n / np.pi)), 1)) if len(n) else 30.0
    except Exception:
        return 30.0


def get_meta(sid):
    with LOCK:
        m = SEQS.get(sid)
        if m is None and sid.startswith("lib-"):
            library()
            m = SEQS.get(sid)
    if m is None:
        raise HTTPException(404, "unknown sequence")
    return m


_SEQ_CACHE: dict = {}


def get_seq(m):
    key = (m["id"], m.get("status") == "ready")
    s = _SEQ_CACHE.get(key)
    if s is None:
        if not (Path(m["ds_dir"]) / m["seq"]).is_dir():
            raise HTTPException(409, "the images are still being prepared")
        s = P.Seq(Path(m["ds_dir"]), m["seq"])
        if not s.frames:
            raise HTTPException(409, "no frames found for this sequence")
        _SEQ_CACHE[key] = s
    return s


# ----------------------------------------------------------------------------- layer sessions
# Principle 1: whole-cell measurements come only from whole cells -> the nucleus layer has its OWN feature
#              names (nucleus_area ...); plain cell names (area, size, circularity ...) are refused there.
# Principle 2: a nucleus is not a cell -> nucleus ids are separate, each nucleus knows its HOST cell, each
#              cell knows how many nuclei it holds (n_nuclei), and a nuclear division says whether the cell
#              itself also divided (the two daughter nuclei end up in different cells).
NUC_RENAME = {"area": "nucleus_area", "perimeter": "nucleus_perimeter", "eccentricity": "nucleus_eccentricity",
              "solidity": "nucleus_solidity", "circularity": "nucleus_circularity",
              "irregularity": "nucleus_irregularity", "aspect_ratio": "nucleus_aspect_ratio",
              "mean_intensity": "nucleus_intensity", "texture_cv": "nucleus_texture",
              "frames_until_division": "frames_until_nuclear_division"}
NUC_DOC = {NUC_RENAME[k]: "NUCLEUS " + v for k, v in P.FEATURE_DOC.items() if k in NUC_RENAME}
NUC_DOC["frames_until_nuclear_division"] = ("frames until this NUCLEUS track ends in a nuclear division "
                                            "(0 = divides after this frame; see divisions for whether the cell split)")
NUC_DOC.update(cx="nucleus centroid x (px, left->right)", cy="nucleus centroid y (px, top->bottom)",
               host_cell="id of the whole cell that contains this nucleus (0 = none found)")
NUC_ALIAS = {"x": "cx", "y": "cy", "host": "host_cell", "cell": "host_cell", "cell_id": "host_cell"}
for _k, _v in NUC_RENAME.items():
    NUC_ALIAS[_k.replace("_cv", "").replace("mean_", "")] = _v
    NUC_ALIAS["nuclear_" + _k] = _v
    NUC_ALIAS["nucleus_" + _k] = _v
NUC_ALIAS.update(nucleus_size="nucleus_area", nuclear_size="nucleus_area", nucleus_brightness="nucleus_intensity",
                 nucleus_elongation="nucleus_aspect_ratio", nucleus_roundness="nucleus_circularity")
CELL_WORDS = set(NUC_RENAME) | {"size", "roundness", "elongation", "intensity", "brightness", "texture"}


def _host_map(nlab, clab):
    """{nucleus id: host cell id (0 = no cell holds more than half of it)}"""
    m = nlab > 0
    if clab is None or not m.any():
        return {}
    n, c = nlab[m].astype(np.int64), clab[m].astype(np.int64)
    pairs, cnt = np.unique(np.stack([n, c]), axis=1, return_counts=True)
    size = dict(zip(*np.unique(n, return_counts=True)))
    best = {}
    for (ni, ci), k in zip(pairs.T, cnt):
        if ci and k > best.get(int(ni), (0, 0))[1]:
            best[int(ni)] = (int(ci), int(k))
    return {int(ni): (best[int(ni)][0] if int(ni) in best and best[int(ni)][1] > 0.5 * size[ni] else 0) for ni in size}


class CellSession(P.Session):
    """Whole-cell layer; when a nucleus layer exists each cell also gets n_nuclei."""
    nuc = None

    def feats(self, t):
        f = super().feats(t)
        if self.nuc is not None and f and "n_nuclei" not in next(iter(f.values())):
            cnt = {}
            for r in self.nuc.feats(t).values():
                if r.get("host_cell"):
                    cnt[r["host_cell"]] = cnt.get(r["host_cell"], 0) + 1
            for i, r in f.items():
                r["n_nuclei"] = cnt.get(i, 0)
        return f

    def feature_doc(self):
        d = dict(super().feature_doc())
        if self.nuc is not None:
            d["n_nuclei"] = "number of nuclei inside this cell (from the nuclei layer; 2+ = multinucleated)"
        return d


class NucleusSession(P.Session):
    """Nucleus layer: nucleus ids, nucleus_* features, host cells, nuclear divisions."""
    cells = None
    alias_ok = False          # offline planner only: map plain names when the question is about nuclei

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._nf = {}

    def feats(self, t):
        if t not in self._nf:
            if len(self._nf) > 200:
                self._nf.pop(next(iter(self._nf)))
            base = super().feats(t)
            host = {}
            if self.cells is not None and t in self.cells.seg.files and t in self.seg.files:
                host = _host_map(self.seg.label(t), self.cells.seg.label(t))
            self._nf[t] = {i: {**{NUC_RENAME.get(k, k): v for k, v in r.items()}, "host_cell": host.get(i, 0)}
                           for i, r in base.items()}
        return self._nf[t]

    def feature_doc(self):
        return NUC_DOC

    def _feature(self, f):
        k = str(f).strip().lower().replace(" ", "_")
        if k in NUC_DOC:
            return k
        if k in CELL_WORDS and not self.alias_ok:
            raise ValueError(f"'{f}' is a WHOLE-CELL measurement and the active layer is NUCLEI. For the size or shape of "
                             f"cells call use_layer('cells'). Only if the question is about nuclei use the nucleus_* features "
                             f"(e.g. {NUC_RENAME.get(P.FEATURE_ALIAS.get(k, k), 'nucleus_area')}).")
        k = NUC_ALIAS.get(k, k)
        if k in CELL_WORDS and self.alias_ok:
            k = NUC_RENAME.get(P.FEATURE_ALIAS.get(k, k), k)
        if k not in NUC_DOC:
            raise ValueError(f"unknown feature '{f}' in the nuclei layer. Valid: {', '.join(NUC_DOC)}")
        return k

    def tool_select_all_cells(self):
        self.selection = sorted(self.feats(self.t))
        rows = self._rows()
        summ = {}
        for f in ("nucleus_area", "nucleus_aspect_ratio", "nucleus_circularity"):
            v = np.array([r[f] for r in rows if f in r])
            if len(v):
                summ[f] = {"p25": round(float(np.percentile(v, 25)), 3), "median": round(float(np.median(v)), 3),
                           "p75": round(float(np.percentile(v, 75)), 3)}
        return {"frame": self.t, "n_nuclei": len(self.selection), "summary": summ,
                "note": "ids are NUCLEUS ids (nucleus 7 is not cell 7); host_cell gives the whole cell"}

    def tool_count(self):
        hosts = {r.get("host_cell") for r in self._rows() if r.get("host_cell")}
        return {"frame": self.t, "count": len(self.selection), "n_nuclei": len(self.selection),
                "n_host_cells": len(hosts)}

    def tool_get_cell(self, cell_id):
        out = super().tool_get_cell(cell_id)
        if "error" not in out:
            out = {"object": "nucleus", **out}
            tr = out.pop("track", {})
            out["nucleus_track"] = {"begin": tr.get("begin"), "end": tr.get("end"), "parent_nucleus": tr.get("parent"),
                                    "daughter_nuclei": tr.get("daughters")}
        return out

    def tool_track(self, ids=None):
        out = super().tool_track(ids)["tracks"]
        f = self.feats(self.t)
        return {"nucleus_tracks": {i: {"begin": d["begin"], "end": d["end"], "parent_nucleus": d["parent"],
                                       "daughter_nuclei": d["daughters"], "nucleus_divides": d["divides"],
                                       "frames_until_nuclear_division": d["frames_until_division"],
                                       "host_cell_now": f.get(i, {}).get("host_cell")} for i, d in out.items()},
                "note": "nuclear tracks: a nuclear division is a cell division only if cell_divided (see divisions)"}

    def division_info(self, parent):
        """Did the CELL divide too?  True if the two daughter nuclei sit in different cells within 3 frames
        of their birth, False if they stay in one cell, None if no host cells were found."""
        tr = self.seg.tracks.get(int(parent)) or {}
        kids = [k for k in (tr.get("children") or []) if k in self.seg.tracks]
        if len(kids) < 2:
            return dict(cell_divided=None, host_cells=[])
        t0 = min(self.seg.tracks[k]["B"] for k in kids)
        seen_same, hosts_seen = False, []
        for t in range(t0, t0 + 4):
            if t not in self.seg.files:
                continue
            f = self.feats(t)
            hs = [f.get(k, {}).get("host_cell", 0) for k in kids[:2]]
            if all(hs):
                hosts_seen = hs
                if hs[0] != hs[1]:
                    return dict(cell_divided=True, host_cells=hs)
                seen_same = True
        return dict(cell_divided=False if seen_same else None, host_cells=hosts_seen)

    def tool_divisions(self, start, end):
        out = super().tool_divisions(start, end)
        ev = []
        for e in out["events"]:
            info = self.division_info(e["parent"])
            ev.append({"kind": "nuclear division", "parent": e["parent"], "daughters": e["daughters"],
                       "frame": e["frame"], "x": e["x"], "y": e["y"], **info})
        n_cell = sum(1 for e in ev if e["cell_divided"])
        return {"n": len(ev), "n_cell_also_divided": n_cell, "events": ev,
                "note": "NUCLEAR divisions: parent and daughters are NUCLEUS ids. cell_divided = the two daughter "
                        "nuclei ended up in different cells (False = still one cell, e.g. binucleated; None = cells not "
                        "found). Submit the parent ids as division_parent_ids."}


class LayeredSession:
    """One P.Session per outline layer ("cells", optionally "nuclei"), one of them active.
    Everything the pipeline and the planner use (tools, feats, seg, frames ...) goes to the active layer;
    steps, trace and the answer are kept here so a question can switch layers on the way."""

    def __init__(self, sessions: dict, nuclei_reason: str = "", states=None, states_reason: str = ""):
        self.__dict__["layers"] = sessions
        self.__dict__["nuclei_reason"] = nuclei_reason
        self.__dict__["states"] = states
        self.__dict__["states_reason"] = states_reason
        self.__dict__["active"] = "cells" if "cells" in sessions else next(iter(sessions))
        self.__dict__["route"] = "cells"
        self.reset()

    def __getattr__(self, k):                       # only called for names not set on this object
        return getattr(self.layers[self.active], k)

    def __setattr__(self, k, v):
        self.__dict__[k] = v

    def layer(self, name=None):
        name = name if name in self.layers else self.active if name in (None, "", "active") else "cells"
        return self.layers.get(name) or self.layers[self.active]

    def reset(self, route="cells"):
        """route: 'auto' (the planner chooses) or a fixed layer."""
        for s_ in self.layers.values():
            s_.reset()
            if isinstance(s_, NucleusSession):
                s_.alias_ok = False
        self.__dict__.update(steps=0, trace=[], answer=None, answer_layer=None, used_layers=[],
                             route=route if (route == "auto" and len(self.layers) > 1) or route in self.layers else "cells")
        self.__dict__["active"] = "cells" if self.route == "auto" else self.route
        if "cells" not in self.layers:
            self.__dict__["active"] = next(iter(self.layers))

    def tool_use_layer(self, layer):
        layer = str(layer).strip().lower()
        layer = {"cell": "cells", "whole": "cells", "whole cells": "cells", "nucleus": "nuclei"}.get(layer, layer)
        if layer not in ("cells", "nuclei"):
            raise ValueError("layer must be 'cells' or 'nuclei'")
        if layer not in self.layers:
            return {"layer": self.active, "note": "nuclei cannot be shown for this video: " +
                    (self.nuclei_reason or "only whole cells were found") + " Answer with the whole cells."}
        if self.route in ("cells", "nuclei") and layer != self.route:
            return {"layer": self.active, "note": f"the user fixed the {LAYER_LABEL[self.route]} layer"}
        t = self.layers[self.active].t
        self.__dict__["active"] = layer
        out = self.layers[layer].tool_set_frame(t)
        return {"layer": layer, **out}

    def tool_cell_states(self, state=None, cell_ids=None):
        if self.states is None:
            return {"available": False, "note": "cell states are not available for this video: " +
                    (self.states_reason or "they have not been computed")}
        if self.active != "cells":                       # states belong to whole cells
            t = self.layers[self.active].t
            self.__dict__["active"] = "cells"
            self.layers["cells"].tool_set_frame(t)
        cs = self.layers["cells"]
        out = self.states.tool(cs.t, state=state, cell_ids=cell_ids)
        if state and "cells_in_state_this_frame" in out:
            cs.selection = [i for i in out["cells_in_state_this_frame"] if i in cs.feats(cs.t)]
            out["selection"] = f"{len(cs.selection)} cells in state {state} are now selected"
        return out

    def tool_explain_cell_state(self, cell_id, frame=None):
        if self.states is None:
            return {"available": False, "note": "cell states are not available for this video: " +
                    (self.states_reason or "they have not been computed")}
        if self.active != "cells":
            t = self.layers[self.active].t
            self.__dict__["active"] = "cells"
            self.layers["cells"].tool_set_frame(t)
        cs = self.layers["cells"]
        t = cs.t if frame is None else int(frame)
        xai = load_xai(self.__dict__.get("sid"))
        if xai is None:
            return {"available": False, "note": "explanations were not made for this video (the cell states were "
                                                "computed before explanations existed): recompute the cell states"}
        out = xai.text(int(cell_id), t)
        if out is None:
            return {"available": False, "note": f"cell {cell_id} has no cell state (not tracked long enough)"}
        if int(cell_id) in cs.feats(cs.t):
            cs.selection = [int(cell_id)]
        return out

    def tool_heatmap(self, cell_id, kind="segmentation", frame=None):
        sid = self.__dict__.get("sid")
        k = "state" if str(kind).lower().startswith(("cell_state", "state", "cell state")) else "seg"
        cs = self.layers["cells"]
        t = cs.t if frame is None else int(frame)
        hs = self.__dict__.setdefault("heatmaps", [])
        if int(cell_id) <= 0:
            return {"error": "cell ids start at 1 (0 is the background): use the ids from your tool results"}
        if len(hs) >= 4:
            return {"error": "at most 4 heatmaps per question: tell the user which cells were explained and offer more"}
        try:
            r = compute_cam(sid, t, int(cell_id), k, 1, locked_session=self)
        except HTTPException as e:
            return {"available": False, "note": str(e.detail)}
        f = r.get("faithfulness") or {}
        better = (f.get("aopc_heatmap", 0) > f.get("aopc_random", 0) + 0.02 * abs(f.get("score_unchanged", 0)) + 0.005) if f else None
        e = dict(cell=int(cell_id), frame=t, kind=k, method=r.get("method"), meaning=r.get("meaning"),
                 deletion_heatmap=f.get("aopc_heatmap"), deletion_random=f.get("aopc_random"), better_than_random=better,
                 explained_state=r.get("explained_state"))
        if k == "seg":
            e.update(heat_on_cell=r.get("heat_inside_cell"))
        else:
            e.update(share_of_cell_pushing_towards_state=r.get("share_towards"), patches=r.get("n_patches"),
                     map_fit_r2=r.get("surrogate_r2"))
        hs.append(e)
        if int(cell_id) in cs.feats(cs.t):
            cs.selection = [int(cell_id)]
        return dict(e, drawn_on_image=True,
                    how_to_read=("segmentation (Grad-CAM++): the brighter (yellow-white) a part of the cell, the more "
                                 "Cellpose-SAM relied on it to find this cell; heat_on_cell = share of the heat on the "
                                 "cell itself. "
                                 "cell_state (patch attribution through the real tree model): orange parts of the cell push "
                                 "towards explained_state, blue parts against; map_fit_r2 = how well the patch map explains "
                                 "the model's score when parts of the cell are hidden (low = read with caution). "
                                 "better_than_random = blurring the strongest areas first changes the model's score more "
                                 "than random blurring. It explains the MODEL, not the biology."))

    def tool_export_table(self, table):
        sid = self.__dict__.get("sid")
        if table not in TABLES:
            return {"error": "table must be one of " + ", ".join(TABLES)}
        ex = self.__dict__.setdefault("exports", [])
        e = dict(table=table, what=TABLES[table], url=f"api/seqs/{sid}/table/{table}.csv")
        if table == "cell_states" and self.states is None:
            return {"available": False, "note": "no cell states for this video: " + (self.states_reason or "")}
        if table in ("counts_per_frame", "divisions", "tracks"):
            cols, rows = build_table(sid, table, self.layers["cells"])
            e.update(columns=cols, rows=len(rows))
            summ = {}
            if table == "counts_per_frame" and rows:
                c = [int(r["cells"]) for r in rows]
                cf = [float(r["confluence_percent"]) for r in rows]
                summ = dict(frames=len(rows), cells_first=c[0], cells_last=c[-1], cells_min=min(c), cells_max=max(c),
                            divisions_total=sum(int(r["divisions"]) for r in rows),
                            confluence_first=cf[0], confluence_last=cf[-1])
                e["series"] = dict(frame=[int(r["frame"]) for r in rows], cells=c)
            elif table == "divisions":
                summ = dict(divisions=len(rows), first_frame=rows[0]["frame"] if rows else None)
            elif table == "tracks":
                summ = dict(tracks=len(rows), whole_video=sum(1 for r in rows if r["first_frame"] == self.frames[0]
                                                              and r["last_frame"] == self.frames[-1]))
            e["summary"] = summ
        ex.append(e)
        return {k: v for k, v in e.items() if k != "series"}

    def tool_start_analysis(self, what):
        m = SEQS.get(self.__dict__.get("sid"))
        if m is None:
            return {"error": "unknown video"}
        if what == "nuclei":
            if "nuclei" in (m.get("layers") or []):
                return {"note": "the nuclei are already available"}
            why = lib_skip_reason(m, "nuclei")
            if why or m.get("ndim") != 2 or not CFG.model_nuclei:
                return {"started": False, "note": why or nuclei_reason(m)}
            queue_nuclei(m, force=True)
            return {"started": True, "note": "finding the nuclei in the background (several minutes)"}
        if what == "cell_states":
            if (m.get("states_check") or {}).get("state") in ("pending", "running"):
                return {"started": False, "note": "the cell states are already being computed"}
            why = lib_skip_reason(m, "states")
            if why or m.get("ndim") != 2 or not CFG.model_states:
                return {"started": False, "note": why or states_reason(m)}
            queue_states(m)
            return {"started": True, "note": "computing the cell states in the background (10-20 minutes)"}
        if what == "find_missed_cells":
            return find_missed_cells(m["id"])
        return {"error": "what must be cell_states, nuclei or find_missed_cells"}

    def tool_reply(self, text, cell_ids=None):
        cs = self.layers[self.active]
        ids = [int(i) for i in (cell_ids or []) if int(i) in cs.feats(cs.t)]
        self.__dict__.update(answer=dict(kind="reply", text=str(text), frame=cs.t, cell_ids=ids, count=len(ids),
                                         division_parent_ids=[], note=str(text)), answer_layer=self.active)
        return {"ok": True}

    def tool_ask_user(self, question, options=None):
        cs = self.layers[self.active]
        opts = [str(o)[:120] for o in (options or [])][:4]
        self.__dict__.update(answer=dict(kind="ask", text=str(question), options=opts, frame=cs.t, cell_ids=[], count=0,
                                         division_parent_ids=[], note=str(question)), answer_layer=self.active)
        return {"ok": True}

    def call(self, name, args):
        chk = self.__dict__.get("stop_check")
        if chk and chk():
            raise StoppedByUser()
        self.__dict__["steps"] = self.steps + 1
        try:
            fn = (self.tool_use_layer if name == "use_layer" else self.tool_cell_states if name == "cell_states"
                  else self.tool_explain_cell_state if name == "explain_cell_state"
                  else self.tool_reply if name == "reply" else self.tool_ask_user if name == "ask_user"
                  else self.tool_heatmap if name == "heatmap" else self.tool_start_analysis if name == "start_analysis"
                  else self.tool_export_table if name == "export_table"
                  else getattr(self.layers[self.active], f"tool_{name}"))
            out = fn(**(args or {}))
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}"}
        if self.active not in self.used_layers:
            self.used_layers.append(self.active)
        if name == "submit_answer" and self.layers[self.active].answer is not None:
            self.__dict__.update(answer=self.layers[self.active].answer, answer_layer=self.active)
        self.trace.append({"tool": name, "args": args or {}, "layer": self.active,
                           "out": json.dumps(out, default=float)[:300]})
        return out


def route_offline(q: str):
    """Keyword version of LAYER_NOTE, for the offline planner (tests)."""
    s = q.lower()
    if re.search(r"\bnucle", s):
        return "nuclei"
    if re.search(r"\b(how many|count|number of)\b|divid|division|mitos|lineage|track|follow", s) and \
            not re.search(r"size|area|shape|elongat|round|spread|outline|perimeter|length|width", s):
        return "nuclei"
    return "cells"


def nuclei_reason(m):
    """Why a video has no nuclei layer (for the planner and the page)."""
    if not CFG.model_nuclei:
        return "the nucleus model is not loaded on this server."
    if m.get("ndim") != 2:
        return "nuclei are only found for 2D videos."
    if lib_skip_reason(m, "nuclei"):
        return lib_skip_reason(m, "nuclei") + "."
    c = m.get("nuclei_check") or {}
    st = c.get("state")
    if st in (None, "pending", "running"):
        return "the nuclei of this video are still being found (try again in a minute or two)."
    if st == "no":
        return ("the cells do have nuclei, but the nucleus model (trained on DIC images only) cannot find them in "
                f"this type of image ({c.get('why', 'its nuclei did not match the cells')}), so they are not shown.")
    if st == "error":
        return "finding the nuclei failed for this video."
    return ""


def states_reason(m):
    if not CFG.model_states:
        return "the cell-state model is not loaded on this server."
    if m.get("kind") not in ("upload", "library"):
        return "cell states are computed for uploaded videos."
    if m.get("ndim") != 2:
        return "cell states are only available for 2D time-lapse videos."
    if lib_skip_reason(m, "states"):
        return "cell states are not computed for this video: " + lib_skip_reason(m, "states") + "."
    c = m.get("states_check") or {}
    st = c.get("state")
    if st in (None, "pending", "running"):
        return "the cell states of this video are still being computed (try again in a minute or two)."
    if st == "error":
        return "computing the cell states failed for this video."
    return ""


def load_states(m):
    if (m.get("states_check") or {}).get("state") != "ok":
        return None
    try:
        import cs_states
        return cs_states.States(seq_root(m["id"]) / "states")
    except Exception:
        traceback.print_exc()
        return None


FB = None                                  # cs_feedback.FeedbackStore, set in main()
_XAI: dict = {}
_CAMS: dict = {}
_CAM_LOCK = threading.Lock()


def load_xai(sid):
    """SHAP explanations of the cell states (states/xai.npz), cached until the file changes."""
    if not sid:
        return None
    m = SEQS.get(sid)
    if not m or (m.get("states_check") or {}).get("state") != "ok":
        return None
    f = seq_root(sid) / "states" / "xai.npz"
    if not f.exists():
        return None
    key = f.stat().st_mtime
    if sid not in _XAI or _XAI[sid][0] != key:
        import cs_xai
        _XAI[sid] = (key, cs_xai.Explanations(f.parent))
    return _XAI[sid][1]


def cam_engine(model_path):
    debug = bool(CFG.debug_segmenter or not model_path)
    k = "debug" if debug else str(model_path)
    with _CAM_LOCK:
        if k not in _CAMS:
            import cs_xai
            eng = cs_xai.CamEngine(model_path, debug=debug)
            if debug:                                # tests on CPU: random weights, a few blocks are enough
                eng.net.encoder.blocks = eng.net.encoder.blocks[:3]
            _CAMS[k] = eng
        return _CAMS[k]


def layer_dirs(m):
    """{layer: result folder}"""
    d = {"cells": m["res_dir"]}
    if "nuclei" in (m.get("layers") or []) and m.get("res_dir_nuclei"):
        d["nuclei"] = m["res_dir_nuclei"]
    return d


def get_session(sid):
    m = get_meta(sid)
    if m["status"] != "ready":
        raise HTTPException(409, f"sequence is not ready yet ({m['status']})")
    with LOCK:
        if sid not in SESS:
            seq = get_seq(m)
            subs = {}
            for layer, rd in layer_dirs(m).items():
                seg = P.SegResult(Path(rd))
                if seg.files:
                    cls = NucleusSession if layer == "nuclei" else CellSession if seq.ndim == 2 else P.Session
                    subs[layer] = cls(seq, seg, None, m["ds_name"])
            if "nuclei" in subs and "cells" in subs:          # link the layers (host cells, n_nuclei)
                subs["nuclei"].cells, subs["cells"].nuc = subs["cells"], subs["nuclei"]
            if "cells" not in subs:
                raise HTTPException(409, "no segmentation found")
            st = load_states(m)
            SESS[sid] = (LayeredSession(subs, nuclei_reason(m) if "nuclei" not in subs else "", st,
                                        "" if st is not None else states_reason(m)), threading.Lock())
            SESS[sid][0].__dict__["sid"] = sid
            if "ndim" not in m or "shape" not in m:
                m.update(ndim=seq.ndim, shape=list(seq.shape), n_frames=len(seq.frames),
                         frames=[min(seq.frames), max(seq.frames)])
        return SESS[sid]


def public(m):
    keys = ("id", "kind", "name", "ndim", "shape", "n_frames", "frames", "status", "progress", "error",
            "created", "backend", "spacing", "cells_per_frame", "seg_seconds", "ds_name", "source", "video",
            "cell_size", "diameter", "diameter_extra", "extra_model", "multiscale", "ms_version", "seg_version", "size_report", "layers", "nuclei_per_frame", "diameter_nuclei",
            "divisions", "divisions_nuclei", "layer_busy", "nuclei_check", "states_check", "ndim_reason",
            "spacing_source")
    d = {k: m.get(k) for k in keys if k in m}
    d["layers"] = m.get("layers") or ["cells"]
    if m.get("kind") in ("upload", "library") and m.get("status") == "ready" and "nuclei" not in d["layers"]:
        d["nuclei_reason"] = nuclei_reason(m)
        d["nuclei_skip"] = bool(lib_skip_reason(m, "nuclei"))
    if m.get("status") == "ready" and (m.get("states_check") or {}).get("state") != "ok":
        d["states_reason"] = states_reason(m)
    d["backend_label"] = BACKEND_NAMES.get(m.get("backend"), m.get("backend"))
    if m["kind"] == "upload":
        d["model"] = model_key(m)
        d["model_label"] = CFG.models[d["model"]]["label"]
        d["backend_label"] = d["model_label"]
    return d


# =============================================================================
# upload -> CTC-style folder -> segmentation + tracking (one GPU job at a time)
# =============================================================================
_NUM = re.compile(r"(\d+)")


def natural(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in _NUM.split(p.name)]


VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".ogv", ".m4v", ".gif", ".mpg", ".mpeg", ".wmv")
TIF_EXT = (".tif", ".tiff")
IMG_EXT = TIF_EXT + (".png", ".jpg", ".jpeg", ".bmp")    # still images: one per time point (or one multi-page .tif)
UPLOAD_EXT = IMG_EXT + (".zip",) + VIDEO_EXT


def read_image(f: Path) -> np.ndarray:
    """.tif/.tiff with tifffile (keeps 16-bit, z-stacks, multi-page); .png/.jpg/.bmp with imageio or Pillow."""
    if f.suffix.lower() in TIF_EXT:
        return tifffile.imread(f)
    try:
        import imageio.v2 as iio
        a = iio.imread(str(f))
    except Exception:
        from PIL import Image
        with Image.open(f) as im:
            a = np.asarray(im)
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[-1] in (3, 4):        # colour picture: average R,G,B (drop alpha)
        a = a[..., :3].astype(np.float32).mean(-1)
    return a


def frames_from_video(path: Path, max_frames: int, max_side: int):
    """A normal video file -> list of 2D grayscale frames (at most max_frames, evenly spaced;
    frames larger than max_side are shrunk). Returns (frames, info)."""
    try:
        import imageio.v2 as iio
    except ImportError:
        raise ValueError("video support needs the packages imageio and imageio-ffmpeg "
                         "(pip install imageio imageio-ffmpeg in the web app's Python environment)")
    reader = iio.get_reader(str(path))
    meta = reader.get_meta_data() or {}
    try:
        n = reader.count_frames()
    except Exception:
        n = None
    if not n or n == float("inf"):
        n = None
    step = max(1, math.ceil(n / max_frames)) if n else 1
    out, total = [], 0
    for i, fr in enumerate(reader):
        total = i + 1
        if i % step:
            continue
        a = np.asarray(fr)
        a = a[..., :3].astype(np.float32).mean(-1) if a.ndim == 3 else a.astype(np.float32)
        h, w = a.shape
        if max(h, w) > max_side:
            from skimage.transform import resize
            sc = max_side / max(h, w)
            a = resize(a, (round(h * sc), round(w * sc)), anti_aliasing=True, preserve_range=True).astype(np.float32)
        out.append(a)
        if n is None and len(out) >= 4 * max_frames:          # unknown length: keep reading, thin later
            out, step = out[::2], step * 2
    reader.close()
    if len(out) > max_frames:
        k = math.ceil(len(out) / max_frames)
        out, step = out[::k], step * k
    if not out:
        raise ValueError("no frames could be read from the video")
    info = dict(file=path.name, frames_in_video=total, used_frames=len(out), every_nth_frame=step,
                fps=meta.get("fps"), size=list(out[0].shape))
    return out, info


def gray(a):
    """RGB(A) -> first channel, like the pipeline's Seq.image."""
    return a[..., 0] if a.ndim >= 3 and a.shape[-1] in (3, 4) and a.shape[0] > 4 else a


def _tif_info(f: Path):
    """axes, shape (colour samples dropped) and ImageJ / resolution metadata of a .tif, without reading pixels."""
    with tifffile.TiffFile(f) as tf:
        s = tf.series[0]
        axes, shape = s.axes, list(s.shape)
        for ax in ("S", "C"):                     # colour channels are not time or depth
            if ax in axes and shape[axes.index(ax)] <= 4:
                i = axes.index(ax)
                axes, shape = axes[:i] + axes[i + 1:], shape[:i] + shape[i + 1:]
        if len(shape) >= 3 and shape[-1] in (3, 4) and axes[-1] not in "XY":
            axes, shape = axes[:-1], shape[:-1]
        ij = tf.imagej_metadata or {}
        spacing = None
        try:
            p0 = tf.pages[0]
            xr = p0.tags.get("XResolution")
            unit = str(ij.get("unit", "")).lower()
            if xr and unit in ("micron", "um", "µm", "\\u00b5m"):
                num, den = xr.value
                px = den / num if num else None
                if px:
                    spacing = [float(ij.get("spacing", px)), px, px]
        except Exception:
            spacing = None
    return axes, shape, ij, spacing


def detect_ndim(files: list[Path]):
    """2D or 3D time-lapse, read from the files: (ndim, reason, voxel size in um or None).
    videos / .png / .jpg -> 2D. .tif: the axes saved in the file (ImageJ / OME: T = time, Z = depth) decide;
    without them: one image per file -> 2D; one stack per file -> 3D; a single stack -> 2D frames (most common),
    a single 4-D array -> 3D (T, Z, Y, X)."""
    if any(f.suffix.lower() in VIDEO_EXT for f in files):
        return 2, "a video", None
    tifs = [f for f in files if f.suffix.lower() in TIF_EXT]
    if not tifs:
        return 2, "still images (.png / .jpg)", None
    axes, shape, ij, sp = _tif_info(tifs[0])
    nd = len(shape)
    if len(tifs) > 1:                             # one file per time point
        if nd == 2:
            return 2, "one 2D image per file", None
        if nd == 3:
            return 3, f"one z-stack per file ({shape[0]} slices)", sp
        raise ValueError(f"{tifs[0].name}: cannot tell a time point from an array of shape {shape}")
    if "Z" in axes and nd >= 3:                   # metadata says depth
        return 3, f"the file says it has {shape[axes.index('Z')]} z-slices" + (" per time point" if "T" in axes else ""), sp
    slices, frames_ = int(ij.get("slices", 1) or 1), int(ij.get("frames", 1) or 1)
    if slices > 1 and nd >= 3:
        return 3, f"ImageJ: {slices} slices x {frames_} frames", sp
    if nd == 2:
        return 2, "a single image", None
    if nd == 3:
        return 2, f"{shape[0]} frames in one file" + (" (time axis in the file)" if "T" in axes else ""), None
    if nd == 4:
        return 3, f"a 4-D array {shape} read as time, depth, y, x", sp
    raise ValueError(f"cannot read a time-lapse from an array of shape {shape}")


def frames_from_files(files: list[Path], ndim: int) -> list[np.ndarray]:
    """One file per time point, or one multi-page file holding the whole time-lapse."""
    if len(files) == 1:
        a = read_image(files[0])
        if ndim == 2:
            if a.ndim == 2 or (a.ndim == 3 and a.shape[-1] in (3, 4)):
                return [gray(a)]
            if a.ndim == 3:
                return [a[t] for t in range(a.shape[0])]
            if a.ndim == 4 and a.shape[-1] in (3, 4):
                return [gray(a[t]) for t in range(a.shape[0])]
            raise ValueError(f"cannot read a 2D time-lapse from an array of shape {a.shape}")
        a = gray(a)
        if a.ndim == 3:
            return [a]
        if a.ndim == 4:
            return [a[t] for t in range(a.shape[0])]
        raise ValueError(f"cannot read a 3D time-lapse from an array of shape {a.shape} (expected Z,Y,X or T,Z,Y,X)")
    out = [gray(read_image(f)) for f in files]
    want = 2 if ndim == 2 else 3
    for f, a in zip(files, out):
        if a.ndim != want:
            raise ValueError(f"expected one {'2D image' if ndim == 2 else '3D z-stack (Z,Y,X)'} per file; "
                             f"{f.name} has shape {a.shape}")
    return out


def expand_uploads(raw: Path) -> list[Path]:
    files = []
    for f in sorted(raw.iterdir(), key=natural):
        if f.suffix.lower() == ".zip":
            dest = raw / (f.stem + "_unzipped")
            dest.mkdir(exist_ok=True)
            with zipfile.ZipFile(f) as z:
                for info in z.infolist():
                    nm = Path(info.filename).name                  # flatten: no path traversal
                    if info.is_dir() or nm.startswith(".") or Path(nm).suffix.lower() not in IMG_EXT:
                        continue
                    with z.open(info) as src, open(dest / nm, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            files += sorted(dest.iterdir(), key=natural)
        elif f.suffix.lower() in IMG_EXT + VIDEO_EXT:
            files.append(f)
    return files


def model_key(m):
    k = m.get("model") or "thesis"
    return k if k in CFG.models else CFG.default_model


# Cell size ("diameter" in Cellpose, in pixels): the image is rescaled so cells are ~30 px for the model.
# 30 = Cellpose default.  "auto": try every candidate on a few frames and keep the size the result agrees
# with (median diameter of the cells found closest to the diameter asked for).
SIZE_CHOICES = {"small": 15, "medium": 30, "large": 60, "xlarge": 120, "huge": 240}
SIZE_CANDIDATES = (15, 30, 60, 120, 240)          # 240: one big cell filling the view (e.g. 42166)


def _median_diameter(lab):
    ids, areas = np.unique(lab[lab > 0], return_counts=True)
    areas = areas[areas >= 9]
    return float(np.median(2 * np.sqrt(areas / np.pi))) if len(areas) else 0.0


MS_VERSION = 2   # v2: the extra size covers a missed cell with ONE object (no fragments); fragments never added


def _keep(a, d_big):
    """an object found at cell size d_big that really has that size (>= half d_big wide); smaller bits are
    usually pieces of a larger cell cut up because the size was wrong"""
    return 2 * np.sqrt(a / np.pi) >= 0.5 * d_big


def merge_big(base, big, max_overlap=0.2, d_big=None):
    """add the objects of a second-size segmentation that the main one missed: an object of `big` is added when
    at most max_overlap of it is already covered by `base` cells and (if d_big is given) it really has that size;
    ids continue after base's."""
    out = base.copy()
    nxt = int(base.max()) + 1
    ids, areas = np.unique(big[big > 0], return_counts=True)
    for i, a in zip(ids, areas):
        if d_big and not _keep(a, d_big):
            continue
        mk = big == i
        if (base[mk] > 0).sum() <= max_overlap * a:
            out[mk & (out == 0)] = nxt
            nxt += 1
    return out


def _missed_big(base, big, d_big, max_overlap=0.2):
    """(count, area) of the objects of `big` (found at cell size d_big) that `base` misses and that really have
    that size"""
    n, area = 0, 0
    ids, areas = np.unique(big[big > 0], return_counts=True)
    for i, a in zip(ids, areas):
        if not _keep(a, d_big):
            continue
        if (base[big == i] > 0).sum() <= max_overlap * a:
            n += 1
            area += int(a)
    return n, area


def choose_size(run, seq, alt_run=None, relaxed=False):
    """run(image, diameter) -> labels.  Returns (diameter, report, extra) for 2D; 3D keeps the default.
    extra: a second cell size, much larger or much smaller than the main one, when cells of very different sizes
    share the video (e.g. one large cell among small ones): the main size misses them, so both sizes are used and
    merged (merge_big: objects of the second size are added where the main size found nothing)."""
    fr = sorted(seq.frames)
    pick = sorted({fr[0], fr[len(fr) // 2]})
    report, best, labs = [], None, {}
    for d in SIZE_CANDIDATES:
        meds, ns = [], []
        for t in pick:
            lab = run(seq.image(t).astype(np.float32), d)
            labs[(d, t)] = lab
            ns.append(int(lab.max() > 0 and len(np.unique(lab)) - 1))
            meds.append(_median_diameter(lab))
        md = float(np.median([x for x in meds if x > 0])) if any(meds) else 0.0
        score = abs(np.log(md / d)) if md > 0 else float("inf")
        report.append(dict(diameter=d, cells=ns, median_cell_px=round(md, 1),
                           score=round(score, 3) if np.isfinite(score) else None))   # inf is not valid JSON
        if best is None or score < best[1] - 1e-9:
            best = (d, score)
    d0 = best[0]
    cands = []                                          # (d2, model, missed count, missed area)
    tries = [("same", run)] + ([("general", alt_run)] if alt_run else [])
    for name, rr in tries:
        for d2 in SIZE_CANDIDATES:
            if name == "same" and d0 / 3 < d2 < 3 * d0:     # same model: only clearly different sizes
                continue
            if name == "general" and d2 == d0:
                continue
            miss = []
            for t in pick:
                lab2 = labs[(d2, t)] if name == "same" else rr(seq.image(t).astype(np.float32), d2)
                miss.append(_missed_big(labs[(d0, t)], lab2, d2))
            ns = [x[0] for x in miss]
            ok = (max(ns) >= 1) if relaxed else (min(ns) >= 1)   # normally: seen in every test frame
            if ok:
                cands.append((d2, name, sum(ns), sum(x[1] for x in miss)))
    report.append(dict(extra_candidates=[dict(diameter=c[0], model=c[1], missed_cells=c[2], missed_area_px=c[3])
                                         for c in cands]))
    if not cands:
        return d0, report, None
    # the size that covers (nearly) all the missed area with the FEWEST objects: a big cell found whole, not
    # cut into pieces by a too-small size; ties -> the same model, then the larger size
    top = max(c[3] for c in cands)
    good = [c for c in cands if c[3] >= 0.6 * top]
    d2, which, n2, a2 = min(good, key=lambda c: (c[2] / max(len(pick), 1), c[1] != "same", -c[0]))
    report.append(dict(extra_diameter=d2, extra_model=which, cells_missed_by_main_size=n2))
    return d0, report, (d2, which)


def make_segmenter(seq, key, m=None, layer="cells"):
    """Returns fn(t) -> labels.  m (the sequence's meta) carries cell_size and gets the chosen diameter.
    layer "nuclei" uses the nucleus model (--model-nuclei); its size is stored as diameter_nuclei."""
    m = m if m is not None else {}
    path = CFG.model_nuclei if layer == "nuclei" else CFG.models[key]["path"]
    dkey = "diameter_nuclei" if layer == "nuclei" else "diameter"
    want = str(m.get("cell_size") or "auto")
    if CFG.debug_segmenter or not path:  # testing without Cellpose weights / GPU
        from skimage import filters, measure as sm, morphology, transform
        def run(im, d):
            f = 30.0 / d                         # same rescaling idea as Cellpose's diameter
            small = transform.rescale(im, f, preserve_range=True, anti_aliasing=True) if f != 1 else im
            b = small > filters.threshold_otsu(small)
            if layer == "nuclei":                # tests: smaller objects than the cell layer
                b = morphology.binary_erosion(b, morphology.disk(6))
            b = morphology.remove_small_objects(b, 10)
            lab = sm.label(b)
            if f != 1:
                lab = transform.resize(lab, im.shape, order=0, preserve_range=True, anti_aliasing=False)
            return lab.astype(np.int32)
        gen_run = None
        threed = False
    else:
        from cellpose import models as cpm
        model = cpm.CellposeModel(gpu=CFG.gpu, pretrained_model=path)
        threed = seq.ndim == 3
        kw3 = dict(do_3D=True, z_axis=0, anisotropy=seq.anisotropy) if threed else {}
        def run(im, d):
            kw = dict(kw3)
            if d and d != 30:
                kw["diameter"] = d
            return np.asarray(model.eval(im, batch_size=CFG.batch_size, **kw)[0]).astype(np.int32)
        _gen = {}

        def gen_run(im, d):
            # the general Cellpose-SAM (not fine-tuned): finds very large / unusual cells the fine-tuned model misses
            if "m" not in _gen:
                try:
                    _gen["m"] = cpm.CellposeModel(gpu=CFG.gpu, pretrained_model="cpsam")
                except Exception as e:
                    P.log(f"web: general Cellpose-SAM not available ({type(e).__name__}: {e})")
                    _gen["m"] = None
            if _gen["m"] is None:
                return np.zeros(im.shape, np.int32)
            kw = {"diameter": d} if d and d != 30 else {}
            return np.asarray(_gen["m"].eval(im, batch_size=CFG.batch_size, **kw)[0]).astype(np.int32)
    if threed or want == "medium":
        d = 30
    elif want in SIZE_CHOICES:
        d = SIZE_CHOICES[want]
    elif want.replace(".", "", 1).isdigit():
        d = max(5.0, min(400.0, float(want)))
    else:                                      # auto
        ms = m.get("multiscale", "auto") if layer == "cells" else "off"
        d, rep, ex = choose_size(run, seq, alt_run=gen_run if ms != "off" else None, relaxed=ms == "on")
        m["size_report" if layer == "cells" else "size_report_nuclei"] = rep
        P.log(f"web: {m.get('name', '')} [{layer}]: automatic size {d} px  " +
              " | ".join(f"{r['diameter']}: median {r['median_cell_px']} px, score {r['score']}" for r in rep if "diameter" in r)
              + (f"  + missed cells found at {ex[0]} px ({ex[1]} model)" if ex else ""))
        m.pop("diameter_extra", None); m.pop("extra_model", None)
        if ex and layer == "cells" and ms != "off":
            m["diameter_extra"], m["extra_model"] = ex
    m[dkey] = d
    d2 = m.get("diameter_extra") if layer == "cells" and not threed else None
    if d2:
        run2 = gen_run if (m.get("extra_model") == "general" and gen_run) else run

        def seg2(t):
            im = seq.image(t).astype(np.float32)
            return merge_big(run(im, d), run2(im, d2), d_big=d2)
        return seg2
    return lambda t: run(seq.image(t).astype(np.float32), d)


def process(sid):
    m = SEQS[sid]
    root = seq_root(sid)
    try:
        if m.pop("redo_cells", None):       # same video, cells found again in place (the chat stays)
            seq = P.Seq(Path(m["ds_dir"]), "01")
            m.update(status="segmenting", progress=[0, len(seq.frames)])
            save_meta(m)
            return _segment(sid, m, seq)
        if m.get("source"):                 # re-run of another upload with a different model: images exist
            if m["ndim"] == 3:
                P.VOXEL_UM[m["ds_name"]] = tuple(m["spacing"])
            seq = P.Seq(Path(m["ds_dir"]), "01")
            m.update(status="segmenting", progress=[0, len(seq.frames)])
            save_meta(m)
            return _segment(sid, m, seq)
        m.update(status="converting", progress=[0, 0], error=None)
        save_meta(m)
        files = expand_uploads(root / "raw")
        if not files:
            raise ValueError("no images or video found in the upload")
        if m.get("ndim") not in (2, 3):                 # automatic: read 2D / 3D from the files
            nd, why, sp = detect_ndim(files)
            m.update(ndim=nd, ndim_reason=why)
            if nd == 3:
                user = m.get("spacing_user")
                m["spacing"] = user if user and any(v != 1.0 for v in user) else (sp or [1.0, 1.0, 1.0])
                m["spacing_source"] = "you" if user and any(v != 1.0 for v in user) else ("file" if sp else "default 1 µm")
            else:
                m["spacing"] = None
            P.log(f"web: {m['name']}: detected {nd}D ({why})")
            save_meta(m)
        vids = [f for f in files if f.suffix.lower() in VIDEO_EXT]
        if vids:
            if len(files) > 1:
                raise ValueError("upload one video at a time (without other files)")
            if m["ndim"] != 2:
                raise ValueError("a video is a 2D time-lapse: choose '2D time-lapse'")
            frames, m["video"] = frames_from_video(vids[0], CFG.video_max_frames, CFG.video_max_side)
            P.log(f"web: video {vids[0].name}: {m['video']}")
        else:
            frames = frames_from_files(files, m["ndim"])
        shapes = {f.shape for f in frames}
        if len(shapes) > 1:
            raise ValueError(f"all time points must have the same size; found {sorted(shapes)}")
        dsd = root / "data" / m["ds_name"] / "01"
        if dsd.exists():
            shutil.rmtree(dsd)
        dsd.mkdir(parents=True)
        digits = max(3, len(str(len(frames) - 1)))
        for t, a in enumerate(frames):
            tifffile.imwrite(dsd / f"t{t:0{digits}d}.tif", a)
        del frames
        if m["ndim"] == 3:
            P.VOXEL_UM[m["ds_name"]] = tuple(m["spacing"])
        seq = P.Seq(root / "data" / m["ds_name"], "01")
        m.update(shape=list(seq.shape), n_frames=len(seq.frames), frames=[min(seq.frames), max(seq.frames)],
                 status="segmenting", progress=[0, len(seq.frames)])
        save_meta(m)
        return _segment(sid, m, seq)
    except Exception as e:
        traceback.print_exc()
        m.update(status="error", error=f"{type(e).__name__}: {e}")
        save_meta(m)


def _seg_layer(m, seq, layer, fn, rd, on_frame=None):
    """Segment + track every frame of one layer into rd, then the division clean-up.
    Returns (objects per frame, clean-up report)."""
    rd = Path(rd)
    tmp = rd.with_name(rd.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    trk = P.OverlapTracker(spacing=seq.spacing)
    counts = []
    for k, t in enumerate(sorted(seq.frames)):          # same loop as P.write_res, with progress
        tl = trk.update(t, fn(t))
        tifffile.imwrite(tmp / f"mask{t:0{seq.digits}d}.tif", tl.astype(np.uint16))
        counts.append(int(len(np.unique(tl)) - (1 if (tl == 0).any() else 0)))
        if on_frame:
            on_frame(k)
    # tracker fix: undo "divisions" whose daughters do not live on (outline flicker)
    tracks, relabel, rep = clean_divisions(trk.tracks, max(seq.frames), CFG.min_daughter_life)
    if relabel:
        for t in sorted(seq.frames):
            f = tmp / f"mask{t:0{seq.digits}d}.tif"
            lab = tifffile.imread(f)
            if np.isin(lab, list(relabel)).any():
                lut = np.arange(int(lab.max()) + 1, dtype=lab.dtype)
                for old, new in relabel.items():
                    if old < len(lut):
                        lut[old] = new
                tifffile.imwrite(f, lut[lab])
    (tmp / "res_track.txt").write_text(res_track_text(tracks))
    if rd.exists():
        shutil.rmtree(rd)
    tmp.rename(rd)                                       # a layer appears complete or not at all
    P.log(f"web: {m['name']} [{layer}]: {len(counts)} frames, {np.mean(counts):.1f} objects/frame, "
          f"divisions kept {rep['divisions_kept']}, undone {rep['divisions_undone']}")
    return counts, rep


def _segment(sid, m, seq):
    """Whole cells (the sequence is ready after this); the nuclei follow in the background."""
    try:
        n = len(seq.frames)
        m.update(progress=[0, n], status="sizing" if str(m.get("cell_size") or "auto") == "auto"
                 and m.get("ndim", 2) == 2 else "segmenting")
        save_meta(m)
        fn = make_segmenter(seq, model_key(m), m, "cells")
        m["status"] = "segmenting"
        save_meta(m)

        def prog(k):
            m["progress"] = [k + 1, n]
            if k % 5 == 0 or k + 1 == n:
                save_meta(m)
        t0 = time.time()
        counts, rep = _seg_layer(m, seq, "cells", fn, m["res_dir"], prog)
        m.update(status="ready", cells_per_frame=counts, divisions=rep, seg_seconds=round(time.time() - t0, 1),
                 layers=["cells"])
        for k in ("res_dir_nuclei", "nuclei_per_frame", "divisions_nuclei", "diameter_nuclei"):
            m.pop(k, None)
        m["seg_version"] = int(m.get("seg_version") or 0) + 1      # cached frames / heatmaps are stale now
        for k in ("states_check", "states_summary", "dividing_cells"):
            if m.get("redone_inplace"):
                m.pop(k, None)
        m.pop("redone_inplace", None)
        SESS.pop(sid, None)
        queue_nuclei(m)
        queue_states(m)
    except Exception as e:
        traceback.print_exc()
        m.update(status="error", error=f"{type(e).__name__}: {e}")
    save_meta(m)


# ----------------------------------------------------------------------------- nuclei (automatic)
NUC_JOBS: deque = deque()                # background: nuclei check + layer, after the GPU queue is empty


def queue_nuclei(m, force=False):
    if not CFG.model_nuclei or m.get("kind") not in ("upload", "library") or m.get("ndim") != 2:
        return
    if "nuclei" in (m.get("layers") or []) and not force:
        return
    m["nuclei_check"] = dict(state="pending", force=force)
    save_meta(m)
    if m["id"] not in NUC_JOBS:
        NUC_JOBS.append(m["id"])
    JOB_EVENT.set()


def nuclei_fit(cells, nuclei):
    """Do the nuclei make sense for these cells?  For each nucleus, the cell holding most of it.
    Returns metrics: inside (share of nuclei > 50% inside one cell), one_per_cell (share of cells with
    exactly one nucleus), ratio (median nucleus area / its cell's area)."""
    ins, per_cell, ratios = 0, {}, []
    ids = [i for i in np.unique(nuclei) if i]
    for i in ids:
        mk = nuclei == i
        v = cells[mk]
        lab, cnt = np.unique(v, return_counts=True)
        kk = int(np.argmax(cnt))
        if lab[kk] and cnt[kk] > 0.5 * mk.sum():
            ins += 1
            per_cell[int(lab[kk])] = per_cell.get(int(lab[kk]), 0) + 1
            ratios.append(mk.sum() / max(1, int((cells == lab[kk]).sum())))
    n_cells = len([i for i in np.unique(cells) if i])
    return dict(nuclei=len(ids), cells=n_cells, n_inside=ins,
                n_one=sum(1 for v in per_cell.values() if v == 1), ratios=ratios)


def pool_fits(fits):
    """Pool several frames (counts are added, not averaged: one empty frame must not decide a video with
    one or two cells)."""
    nu = sum(f["nuclei"] for f in fits)
    ce = sum(f["cells"] for f in fits)
    ra = [r for f in fits for r in f["ratios"]]
    return dict(frames=len(fits), nuclei=nu, cells=ce,
                inside=sum(f["n_inside"] for f in fits) / nu if nu else 0.0,
                one_per_cell=sum(f["n_one"] for f in fits) / ce if ce else 0.0,
                ratio=float(np.median(ra)) if ra else 0.0)


def nuclei_ok(f):
    if f["nuclei"] == 0 or f["cells"] == 0:
        return False, "no nuclei were found"
    if f["inside"] < 0.7:
        return False, f"only {f['inside']:.0%} of the nuclei lie inside a cell"
    if f["ratio"] > 0.8:
        return False, "the 'nuclei' are as large as the whole cells"
    if f["ratio"] < 0.05:
        return False, f"the 'nuclei' are tiny fragments (median {f['ratio']:.0%} of their cell's area)"
    if f["nuclei"] > 1.6 * f["cells"]:
        return False, f"there are {f['nuclei'] / f['cells']:.1f} 'nuclei' per cell (expected about one)"
    if f["one_per_cell"] < 0.5:
        return False, f"only {f['one_per_cell']:.0%} of the cells hold exactly one nucleus"
    return True, ""


def build_nuclei(sid):
    m = SEQS.get(sid)
    if not m or m.get("status") != "ready":
        return
    chk = m.get("nuclei_check") or {}
    try:
        chk.update(state="running")
        m["nuclei_check"] = chk
        save_meta(m)
        seq = P.Seq(Path(m["ds_dir"]), m.get("seq", "01"))
        fn = make_segmenter(seq, model_key(m), m, "nuclei")
        cells = P.SegResult(Path(m["res_dir"]))
        fr = sorted(seq.frames)
        pick = sorted({fr[int(round(q * (len(fr) - 1)))] for q in (0, 0.33, 0.66, 1.0)})     # 4 frames, pooled
        fits = [nuclei_fit(cells.label(t), fn(t)) for t in pick if t in cells.files]
        f = pool_fits(fits) if fits else dict(nuclei=0, cells=0)
        ok, why = nuclei_ok(f) if fits else (False, "no frames to check")
        chk.update({k: round(v, 3) for k, v in f.items()})
        P.log(f"web: {m['name']}: nuclei check {'OK' if ok else 'NO'} {why} {f}")
        if not ok and not chk.get("force"):
            chk.update(state="no", why=why)
            return
        rd = seq_root(sid) / "res_nuclei"
        counts, rep = _seg_layer(m, seq, "nuclei", fn, rd)
        with LOCK:
            m.update(res_dir_nuclei=str(rd), nuclei_per_frame=counts, divisions_nuclei=rep,
                     layers=["cells", "nuclei"])
            chk.update(state="ok")
            SESS.pop(sid, None)                    # next question / image uses both layers
    except Exception as e:
        traceback.print_exc()
        chk.update(state="error", why=f"{type(e).__name__}: {e}")
    finally:
        m["nuclei_check"] = chk
        save_meta(m)


# ----------------------------------------------------------------------------- cell states (automatic)
STATE_JOBS: deque = deque()              # background: cell-state classifier, after uploads and nuclei


def queue_states(m):
    if not CFG.model_states or m.get("kind") not in ("upload", "library") or m.get("ndim") != 2:
        return
    m["states_check"] = dict(state="pending")
    save_meta(m)
    if m["id"] not in STATE_JOBS:
        STATE_JOBS.append(m["id"])
    JOB_EVENT.set()


def build_states(sid):
    m = SEQS.get(sid)
    if not m or m.get("status") != "ready":
        return
    chk = dict(state="running", progress=[0, m.get("n_frames") or 0])
    try:
        import cs_states
        m["states_check"] = chk
        save_meta(m)
        seq = P.Seq(Path(m["ds_dir"]), m.get("seq", "01"))
        seg = P.SegResult(Path(m["res_dir"]))
        model_path = CFG.models[model_key(m)]["path"]

        def prog(k, n):
            chk["progress"] = [k + 1, n]
            if k % 10 == 0:
                save_meta(m)
        summ = cs_states.compute(seq, seg, CFG.model_states, model_path, m.get("diameter") or 30,
                                 seq_root(sid) / "states", debug=CFG.debug_segmenter or not model_path, on_frame=prog)
        chk.update(state="ok", **{k: summ[k] for k in ("modality", "ood", "ood_score", "shown_states",
                                                         "n_dying", "n_dividing", "availability")})
        P.log(f"web: {m['name']}: cell states {summ}")
        with LOCK:
            SESS.pop(sid, None)
    except Exception as e:
        traceback.print_exc()
        chk.update(state="error", why=f"{type(e).__name__}: {e}")
    finally:
        m["states_check"] = chk
        save_meta(m)


def worker():
    while True:
        JOB_EVENT.wait()
        while JOBS:
            sid = JOBS.popleft()
            if sid in SEQS and SEQS[sid].get("status") == "queued":
                process(sid)
        while NUC_JOBS and not JOBS:                 # uploads first, nuclei when the queue is empty
            build_nuclei(NUC_JOBS.popleft())
        while STATE_JOBS and not JOBS and not NUC_JOBS:
            build_states(STATE_JOBS.popleft())
        if not JOBS and not NUC_JOBS and not STATE_JOBS:
            JOB_EVENT.clear()


# =============================================================================
# rendering
# =============================================================================
_RENDER: OrderedDict = OrderedDict()
_U8: OrderedDict = OrderedDict()
COL_OUT, COL_HL, COL_DIV, COL_SEL = (0, 200, 230), (60, 220, 110), (255, 150, 40), (255, 215, 0)


def _lru(cache, key, make, size):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    v = make()
    cache[key] = v
    if len(cache) > size:
        cache.popitem(last=False)
    return v


def ids_param(s):
    return [int(x) for x in (s or "").split(",") if x.strip().lstrip("-").isdigit()]


def render_png(sid, t, view="max", outlines=True, hl=(), div=(), sel=None, labels=True, maxw=None, layer="cells",
               selfill=True):
    from PIL import Image, ImageDraw
    from skimage.segmentation import find_boundaries
    m = get_meta(sid)
    seq = get_seq(m)
    if t not in seq.frames:
        raise HTTPException(404, f"frame {t} does not exist")
    z = None if view in (None, "", "max") else int(view)

    def u8():
        img = seq.image(t)
        if img.ndim == 3:
            img = img.max(0) if z is None else img[min(max(z, 0), img.shape[0] - 1)]
        return P.normalize_u8(img)
    g = _lru(_U8, (sid, t, z), u8, 64)
    rgb = np.stack([g] * 3, -1).astype(np.float32)
    lab2 = None
    if m["status"] == "ready" and (outlines or hl or div or sel is not None):
        seg = get_session(sid)[0].layer(layer).seg
        lab = seg.label(t) if t in seg.files else None
        if lab is not None:
            if lab.ndim == 3:
                zz = None if z is None else min(max(z, 0), lab.shape[0] - 1)
                lab2 = lab.max(0) if zz is None else lab[zz]
                fp = (lambda ids: np.isin(lab, ids).any(0)) if zz is None else (lambda ids: np.isin(lab[zz], ids))
            else:
                lab2 = lab
                fp = lambda ids: np.isin(lab, ids)
            if outlines:
                rgb[find_boundaries(lab2, mode="inner")] = COL_OUT
            for ids, col, fill in ((list(hl), COL_HL, 0.42), (list(div), COL_DIV, 0.5),
                                   ([sel] if sel is not None else [], COL_SEL, 0.3 if selfill else 0.0)):
                if not ids:
                    continue
                mk = fp(ids)
                if fill:
                    rgb[mk] = (1 - fill) * rgb[mk] + fill * np.array(col, np.float32)
                bd = find_boundaries(np.where(mk, lab2, 0), mode="inner")
                if not fill:                     # heatmap shown: a thick outline (inside the cell) instead of a fill
                    from scipy.ndimage import binary_dilation as _bdil
                    bd = (_bdil(bd, iterations=max(2, round(min(rgb.shape[:2]) / 250))) & mk) | bd
                rgb[bd] = col
    im = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    if labels and lab2 is not None and (hl or div):
        d = ImageDraw.Draw(im)
        seg = get_session(sid)[0].layer(layer).seg
        lab = seg.label(t)
        for i in list(hl) + list(div):
            mk = (lab == i).any(0) if lab.ndim == 3 else lab == i
            ys, xs = np.nonzero(mk)
            if len(xs):
                x, y = int(xs.mean()), int(ys.min())
                txt = str(i)
                w = 7 * len(txt) + 6
                d.rectangle([x - w // 2, y - 16, x + w // 2, y - 3], fill=(20, 24, 40))
                d.text((x - w // 2 + 3, y - 15), txt, fill=(255, 255, 255))
    if maxw and im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)))
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=False, compress_level=3)
    return buf.getvalue()


# =============================================================================
# API
# =============================================================================
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((HERE / "web" / "index.html").read_text(), headers={"Cache-Control": "no-store"})


def llm_status():
    if CFG.planner == "offline":
        return {"ok": True, "model": "offline keyword planner (testing)", "url": None}
    try:
        with urllib.request.urlopen(CFG.llm_url.rstrip("/") + "/models", timeout=2) as r:
            names = [d.get("id") for d in json.loads(r.read()).get("data", [])]
        return {"ok": CFG.llm_model in names, "model": CFG.llm_model, "url": CFG.llm_url, "served": names}
    except Exception as e:
        return {"ok": False, "model": CFG.llm_model, "url": CFG.llm_url, "error": f"{type(e).__name__}: {e}"}


@app.get("/api/status", dependencies=AUTH)
def status():
    gpu = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return {"llm": llm_status(), "gpu": gpu,
            "models": [dict(key=k, label=v["label"], path=v["path"] or "(debug segmenter)") for k, v in CFG.models.items()],
            "default_model": CFG.default_model,
            "nuclei_model": CFG.model_nuclei or None,
            "states_model": CFG.model_states or None,
            "seg_model": CFG.models[CFG.default_model]["path"] or "(debug segmenter)",
            "work": str(CFG.work), "out": str(CFG.out), "planner": CFG.planner,
            "queue": list(JOBS)}


@app.get("/api/seqs", dependencies=AUTH)
def list_seqs():
    with LOCK:
        lib = [public(m) for m in library()]
        ups = [public(m) for m in SEQS.values() if m["kind"] == "upload"]
    return {"uploads": sorted(ups, key=lambda m: -m.get("created", 0)), "library": lib}


@app.post("/api/seqs", dependencies=AUTH)
async def upload(files: list[UploadFile] = File(...), name: str = Form(""), ndim: str = Form("auto"),
                 vz: float = Form(1.0), vy: float = Form(1.0), vx: float = Form(1.0), model: str = Form(""),
                 cell_size: str = Form("auto"), dic: str = Form("0")):
    model = model or CFG.default_model
    # dic: accepted for old pages and ignored - the nuclei layer is now decided automatically
    cell_size = (cell_size or "auto").strip().lower()
    if cell_size not in ("auto", *SIZE_CHOICES) and not cell_size.replace(".", "", 1).isdigit():
        raise HTTPException(400, "cell_size must be auto, small, medium, large, xlarge or a number of pixels")
    if model not in CFG.models:
        raise HTTPException(400, f"unknown model {model!r}; available: {list(CFG.models)}")
    ndim = {"2": 2, "3": 3}.get(str(ndim).strip().lower(), 0)      # 0 = detect from the files
    if min(vz, vy, vx) <= 0:
        raise HTTPException(400, "voxel sizes must be positive")
    sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    root = seq_root(sid)
    (root / "raw").mkdir(parents=True)
    n = 0
    for f in files:
        nm = Path(f.filename or "").name
        if Path(nm).suffix.lower() not in UPLOAD_EXT:
            continue
        with open(root / "raw" / nm, "wb") as dst:
            while chunk := await f.read(1 << 22):
                dst.write(chunk)
        n += 1
    if not n:
        shutil.rmtree(root)
        raise HTTPException(400, "upload a video (.mp4, .avi, .mov, .gif ...), images (.tif, .png, .jpg: "
                                 "one per time point, or one multi-page .tif) or a .zip of them")
    clean = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name.strip())[:60] or "sequence"
    m = dict(id=sid, kind="upload", name=name.strip() or clean, ds_name=f"upload_{sid}_{clean}",
             ndim=ndim, spacing=[vz, vy, vx] if ndim == 3 else None, spacing_user=[vz, vy, vx],
             status="queued", progress=[0, 0],
             created=time.time(), backend="cpsam_joint", seq="01", n_files=n, model=model, cell_size=cell_size,
             layers=["cells"])
    m["ds_dir"] = str(root / "data" / m["ds_name"])
    m["res_dir"] = str(root / "res")
    with LOCK:
        SEQS[sid] = m
        save_meta(m)
        JOBS.append(sid)
        JOB_EVENT.set()
    return public(m)


def find_missed_cells(sid):
    """Re-segment this upload IN PLACE (same id, same chat): main cell size + the cells it misses (much bigger or
    smaller ones), looked for with the fine-tuned and the general Cellpose-SAM. Nuclei and states follow."""
    m = get_meta(sid)
    if m.get("kind") != "upload":
        return {"started": False, "note": "only uploaded videos can be segmented again (the examples are fixed)"}
    if m.get("ndim") != 2:
        return {"started": False, "note": "this works for 2D videos"}
    if m.get("status") != "ready":
        return {"started": False, "note": f"the video is busy ({m.get('status')}); try again when it is ready"}
    if m.get("multiscale") == "on" and m.get("diameter_extra") and m.get("ms_version") == MS_VERSION:
        return {"started": False, "note": f"already done: cells at {m['diameter_extra']} px ({m.get('extra_model')} "
                                          f"model) were added to the {m.get('diameter')} px cells"}
    with LOCK:
        m.update(multiscale="on", redo_cells=True, redone_inplace=True, status="queued", progress=[0, 0],
                 ms_version=MS_VERSION)
        m.pop("diameter_extra", None); m.pop("extra_model", None)
        save_meta(m)
        JOBS.append(sid)
        JOB_EVENT.set()
    return {"started": True, "note": "finding the cells again, including much bigger or smaller ones (a few minutes); "
                                     "the chat stays, the image updates when it is done; nuclei and cell states are "
                                     "recomputed after"}


@app.post("/api/seqs/{sid}/find_missed_cells", dependencies=AUTH)
def find_missed_cells_ep(sid: str):
    r = find_missed_cells(sid)
    if not r.get("started") and "already" not in r.get("note", ""):
        raise HTTPException(409, r["note"])
    return dict(r, meta=public(get_meta(sid)))


@app.post("/api/seqs/{sid}/resegment", dependencies=AUTH)
def resegment(sid: str, body: dict):
    """Same images, other cell-finding model -> a new sequence next to the original (to compare)."""
    src = get_meta(sid)
    if src["kind"] != "upload" or src["status"] != "ready":
        raise HTTPException(400, "only finished uploads can be re-run")
    key = body.get("model") or src.get("model")
    if key not in CFG.models:
        raise HTTPException(400, f"unknown model {key!r}")
    ms = str(body.get("multiscale") or "auto").lower()
    csz = str(body.get("cell_size") or src.get("cell_size", "auto")).lower()
    if csz not in ("auto", *SIZE_CHOICES) and not csz.replace(".", "", 1).isdigit():
        raise HTTPException(400, "cell_size must be auto, small, medium, large, xlarge, huge or a number of pixels")
    base = SEQS.get(src.get("source")) or src
    nid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    root = seq_root(nid)
    root.mkdir(parents=True)
    plain = re.sub(r"\s+·\s+.*$", "", base["name"])
    tag = CFG.models[key]["label"] if key != src.get("model") else ("big + small cells" if ms == "on" else f"cell size {csz}")
    m = dict(id=nid, kind="upload", name=f"{plain} · {tag}", ds_name=base["ds_name"],
             ds_dir=base["ds_dir"], ndim=base["ndim"], spacing=base.get("spacing"), shape=base.get("shape"),
             n_frames=base.get("n_frames"), frames=base.get("frames"), status="queued", progress=[0, 0],
             created=time.time(), backend="cpsam_joint", seq="01", model=key, source=base["id"],
             cell_size=csz, multiscale=ms, layers=["cells"],
             res_dir=str(root / "res"))
    with LOCK:
        SEQS[nid] = m
        save_meta(m)
        JOBS.append(nid)
        JOB_EVENT.set()
    return public(m)


@app.post("/api/seqs/{sid}/add_nuclei", dependencies=AUTH)
def add_nuclei(sid: str):
    """Build the nuclei layer even if the automatic check said no (API only; the page does not need it)."""
    m = get_meta(sid)
    if not CFG.model_nuclei or m["kind"] not in ("upload", "library") or m.get("ndim") != 2 or m["status"] != "ready":
        raise HTTPException(400, "nuclei can be added to finished 2D uploads when the nucleus model is loaded")
    queue_nuclei(m, force=True)
    return public(m)


@app.post("/api/seqs/{sid}/add_states", dependencies=AUTH)
def add_states(sid: str):
    """(Re)compute the cell states of a finished 2D upload."""
    m = get_meta(sid)
    if not CFG.model_states or m["kind"] not in ("upload", "library") or m.get("ndim") != 2 or m["status"] != "ready":
        raise HTTPException(400, "cell states can be computed for finished 2D uploads when the state model is loaded")
    queue_states(m)
    return public(m)


@app.get("/api/seqs/{sid}/states.csv", dependencies=AUTH)
def states_csv(sid: str):
    m = get_meta(sid)
    f = seq_root(sid) / "states" / "states.csv"
    if (m.get("states_check") or {}).get("state") != "ok" or not f.exists():
        raise HTTPException(409, states_reason(m) or "no cell states for this video")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", m["name"])
    df_txt = f.read_text().replace("track,", "cell,", 1)
    return Response(df_txt, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}_cell_states.csv"'})


@app.get("/api/seqs/{sid}/explain", dependencies=AUTH)
def explain_cell(sid: str, t: int, cid: int):
    """SHAP explanation of one cell's state (exact TreeSHAP, grouped into readable factors)."""
    m = get_meta(sid)
    if (m.get("states_check") or {}).get("state") != "ok":
        raise HTTPException(409, states_reason(m) or "no cell states for this video")
    xai = load_xai(sid)
    if xai is None:
        raise HTTPException(409, "explanations are made together with the cell states: recompute the cell states of "
                                 "this video to get them")
    r = xai.explain(cid, t)
    if r is None:
        raise HTTPException(404, f"cell {cid} has no cell state")
    return r


_CAM_CACHE: "OrderedDict" = None


@app.get("/api/seqs/{sid}/cam", dependencies=AUTH)
def cam_cell(sid: str, t: int, cid: int, kind: str = "seg", faith: int = 1, detail: int = 0):
    """Heatmap for one cell: kind=seg (pixel-level Grad-CAM++: the pixels Cellpose-SAM relied on) or kind=state (image evidence
    for its cell state: patch attribution through the real tree model) + deletion test."""
    return compute_cam(sid, t, cid, kind, faith, detail=detail)


def compute_cam(sid, t, cid, kind="seg", faith=1, locked_session=None, detail=0):
    """the heatmap computation (page and assistant); recent results are cached."""
    global _CAM_CACHE
    from collections import OrderedDict
    if _CAM_CACHE is None:
        _CAM_CACHE = OrderedDict()
    key = (sid, int(t), int(cid), kind, int(bool(faith)), int(bool(detail)), (get_meta(sid).get("states_check") or {}).get("state"),
           get_meta(sid).get("seg_version"))
    if key in _CAM_CACHE:
        _CAM_CACHE.move_to_end(key)
        return _CAM_CACHE[key]
    r = _compute_cam(sid, int(t), int(cid), kind, faith, locked_session, detail)
    _CAM_CACHE[key] = r
    while len(_CAM_CACHE) > 64:
        _CAM_CACHE.popitem(last=False)
    return r


def _compute_cam(sid, t, cid, kind, faith, locked_session=None, detail=0):
    import cs_xai
    m = get_meta(sid)
    if m["status"] != "ready" or m.get("ndim", 2) != 2:
        raise HTTPException(400, "heatmaps are available for finished 2D videos")
    if kind not in ("seg", "state"):
        raise HTTPException(400, "kind must be seg or state")
    if locked_session is not None:             # called by the assistant, which already holds the session lock
        full, lk = locked_session, contextlib.nullcontext()
    else:
        full, lk = get_session(sid)
    sess = full.layer("cells")
    with lk:
        if t not in sess.seg.files:
            raise HTTPException(404, "no segmentation for this frame")
        img = np.asarray(sess.seq.image(t), np.float32)
        lab = np.asarray(sess.seg.label(t))
    if img.ndim != 2:
        img = img[..., 0] if img.shape[-1] in (3, 4) else img
    if cid <= 0 or not (lab == cid).any():
        raise HTTPException(404, f"cell {cid} is not in frame {t}" + (" (cell ids start at 1; 0 is the background)" if cid <= 0 else ""))
    model_path = CFG.models[model_key(m)]["path"] if m.get("kind") == "upload" else CFG.models[CFG.default_model]["path"]
    d = float(m.get("diameter") or 30)
    eng = cam_engine(model_path)
    info = dict(cell=cid, frame=t, kind=kind, method="Grad-CAM++" if kind == "seg" else "patch attribution")
    try:
        if kind == "seg":
            # pixel-level Grad-CAM++ (neck layer x SmoothGrad), shown on the cell; see CamEngine.gradcam_pixels
            r = eng.gradcam_pixels(img, lab, cid, d, faith=bool(faith), detail=bool(detail))
            info.update(method="Grad-CAM++", target="the segmenter's cell probability inside this cell",
                        meaning=("the pixels where Cellpose-SAM decided the outline of this cell" if detail else
                                 "the parts of this cell Cellpose-SAM relied on most to find it"))
        else:
            xai = load_xai(sid)
            if xai is None:
                raise HTTPException(409, states_reason(m) or "explanations are made together with the cell states: "
                                                            "recompute the cell states of this video")
            tm = next(iter(xai.meta["targets"]), None)
            if tm is None:
                raise HTTPException(409, "no explainable cell-state target for this video")
            row = xai.row(cid, t)
            if row is None:
                raise HTTPException(404, f"cell {cid} has no cell state")
            import cs_states
            model = cs_states.load_model(CFG.model_states)
            T = cs_xai.TARGETS[tm["key"]]
            raw = cs_xai.head_raw_contrast(model.heads[T["head"]], xai.d["X"][row:row + 1].astype(np.float64),
                                           T["pos"], T["neg"])
            sign = 1.0 if raw is None or raw[0] >= 0 else -1.0
            scorer = cs_xai.state_scorer(model, xai, row, tm["key"], sign)
            # exact patch attribution through the real tree model (no stand-in, no gradients): see CamEngine.patches
            r = eng.patches(img, lab, cid, d * 0.5, scorer, faith=bool(faith))
            shown = T["label"] if sign > 0 else T["neg_label"]
            other = T["neg_label"] if sign > 0 else T["label"]
            info.update(method="patch attribution (real cell-state model)",
                        target=f"evidence for '{shown}' (the state the model leans to at frame {int(xai.d['frame'][row])})",
                        meaning=f"which parts of the cell push the model towards '{shown}' (orange) or towards "
                                f"'{other}' (blue)", explained_state=shown, other_state=other)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"heatmap failed: {type(e).__name__}: {e}")
    r.update(info)
    return r


@app.get("/api/seqs/{sid}/cam_check", response_class=HTMLResponse, dependencies=AUTH)
def cam_check(sid: str, n: int = 12, seed: int = 0, frame: int = -1):
    """Sanity checks of the segmentation heatmaps on n cells spread over the video (or one frame): localisation,
    deletion, agreement with occlusion, target specificity, model randomisation. Opens as a report page; the same
    report is saved in the sequence's exports folder."""
    import cs_xai
    m = get_meta(sid)
    if m["status"] != "ready" or m.get("ndim", 2) != 2:
        raise HTTPException(400, "the check runs on finished 2D videos")
    n = max(1, min(int(n), 60))
    full, lk = get_session(sid)
    sess = full.layer("cells")
    frames = sorted(sess.seg.files)
    pick = [frame] if frame >= 0 else [frames[int(i * (len(frames) - 1) / max(1, min(n, len(frames)) - 1))]
                                       for i in range(min(n, len(frames)))] if len(frames) > 1 else frames
    pick = [t for t in dict.fromkeys(pick) if t in sess.seg.files]
    if not pick:
        raise HTTPException(404, "frame not found")
    rng = np.random.default_rng(seed)
    per = [n // len(pick) + (1 if i < n % len(pick) else 0) for i in range(len(pick))]
    model_path = CFG.models[model_key(m)]["path"] if m.get("kind") == "upload" else CFG.models[CFG.default_model]["path"]
    d = float(m.get("diameter") or 30)
    eng = cam_engine(model_path)
    rows, maps = [], []
    t0 = time.time()
    try:
        for t, k in zip(pick, per):
            if k <= 0:
                continue
            with lk:
                img = np.asarray(sess.seq.image(t), np.float32)
                lab = np.asarray(sess.seg.label(t))
            if img.ndim != 2:
                img = img[..., 0] if img.shape[-1] in (3, 4) else img
            ids, area = np.unique(lab[lab > 0], return_counts=True)
            ids = ids[area >= 30]
            if not len(ids):
                continue
            cids = rng.choice(ids, size=min(k, len(ids)), replace=False)
            r, mp = eng.check(img, lab, [int(c) for c in cids], d, seed=seed, keep_maps=max(0, 6 - len(maps)))
            for x in r:
                x["frame"] = int(t)
            rows += r
            maps += mp
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"check failed: {type(e).__name__}: {e}")
    if not rows:
        raise HTTPException(404, "no cells to check")
    summ = cs_xai.check_summary(rows)
    note = (f"Video {m.get('name', sid)} · cell size {d:g} px (the model sees a {cs_xai.CamEngine.TILE}-px view "
            f"rescaled by {30.0 / d:.2f}) · {len(pick)} frame(s) · {time.time() - t0:.0f} s"
            + (" · DEBUG model with random weights: the numbers mean nothing" if CFG.debug_segmenter else ""))
    page = cs_xai.check_html("Grad-CAM++ sanity checks (segmentation heatmap)", rows, summ,
                             cs_xai.check_figure(maps), note)
    try:
        ex = seq_root(sid) / "exports"
        ex.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (ex / f"cam_check_{stamp}.html").write_text(page)
        (ex / f"cam_check_{stamp}.json").write_text(json.dumps(dict(summary=summ, cells=rows, note=note), default=float, indent=1))
    except Exception:
        pass
    return HTMLResponse(page)


@app.get("/api/seqs/{sid}", dependencies=AUTH)
def seq_info(sid: str):
    m = get_meta(sid)
    if m["status"] == "ready":
        sess = get_session(sid)[0]
        m.setdefault("n_frames", len(sess.frames))
        m.setdefault("frames", [sess.frames[0], sess.frames[-1]])
        m.setdefault("shape", list(sess.seq.shape))
        m["ndim"] = sess.ndim
        if m["ndim"] == 3:
            m["spacing"] = list(sess.spacing)
    return public(m)


@app.delete("/api/seqs/{sid}", dependencies=AUTH)
def delete_seq(sid: str):
    m = get_meta(sid)
    if m["kind"] != "upload":
        raise HTTPException(400, "test videos cannot be removed")
    if m["status"] in ("converting", "segmenting"):
        raise HTTPException(409, "wait until segmentation has finished")
    users = [o["name"] for o in SEQS.values() if o.get("source") == sid]
    if users:
        raise HTTPException(409, f"remove the re-runs that use these images first: {', '.join(users)}")
    with LOCK:
        SEQS.pop(sid, None)
        SESS.pop(sid, None)
        trash = Path(CFG.work) / "trash"            # moved, not deleted: can be restored by hand
        trash.mkdir(parents=True, exist_ok=True)
        shutil.move(str(seq_root(sid)), str(trash / sid))
    return {"ok": True, "moved_to": str(trash / sid)}


@app.get("/api/seqs/{sid}/frame/{t}.png", dependencies=AUTH)
def frame_png(sid: str, t: int, view: str = "max", outlines: int = 1, hl: str = "", div: str = "",
              sel: int | None = None, labels: int = 1, maxw: int | None = None, layer: str = "cells", selfill: int = 1):
    key = (sid, t, view, outlines, hl, div, sel, labels, maxw, layer, selfill, SEQS.get(sid, {}).get("status"),
           SEQS.get(sid, {}).get("seg_version"))
    png = _lru(_RENDER, key, lambda: render_png(sid, t, view, bool(outlines), ids_param(hl), ids_param(div),
                                                sel, bool(labels), maxw, layer, bool(selfill)), 128)
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=600"})


def _cell_record(sess, t, cid):
    f = sess.feats(t).get(cid)
    if f is None:
        return None
    tr = sess.seg.tracks.get(cid, {})
    rec = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items() if k not in ("bbox", "zrange")}
    if isinstance(sess, NucleusSession):
        rec["object"] = "nucleus"
    rec.update(frame=t, ndim=sess.ndim,
               track=dict(begin=tr.get("B"), end=tr.get("E"), parent=tr.get("P") or None,
                          daughters=tr.get("children") or None))
    return rec


@app.get("/api/seqs/{sid}/cell", dependencies=AUTH)
def pick_cell(sid: str, t: int, x: float, y: float, view: str = "max", cid: int | None = None,
              layer: str = "cells"):
    full, lk = get_session(sid)
    sess = full.layer(layer)
    with lk:
        lab = sess.seg.label(t) if t in sess.seg.files else None
        if lab is None:
            raise HTTPException(404, "no segmentation for this frame")
        if cid is None:
            xi, yi = int(round(x)), int(round(y))
            H, W = lab.shape[-2:]
            if not (0 <= xi < W and 0 <= yi < H):
                return {"cell": None}
            col = (lab[:, yi, xi] if view in ("", "max") else lab[int(view), yi, xi][None]) if lab.ndim == 3 \
                else lab[yi, xi][None]
            col = col[col > 0]
            if not col.size:                                       # near miss: closest cell within 6 px
                l2 = (lab.max(0) if view in ("", "max") else lab[int(view)]) if lab.ndim == 3 else lab
                y0, y1, x0, x1 = max(yi - 6, 0), min(yi + 7, H), max(xi - 6, 0), min(xi + 7, W)
                win = l2[y0:y1, x0:x1]
                ys, xs = np.nonzero(win)
                if not len(xs):
                    return {"cell": None}
                k = int(np.argmin((ys + y0 - yi) ** 2 + (xs + x0 - xi) ** 2))
                cid = int(win[ys[k], xs[k]])
            else:
                cid = int(np.bincount(col).argmax())
        rec = _cell_record(sess, t, int(cid))
        if rec is not None:
            rec["layer"] = layer if layer in full.layers else "cells"
        return {"cell": rec}


def _reply(full, ans):
    if ans is None:
        return None
    layer = full.answer_layer or full.active
    sess = full.layer(layer)
    nuc = layer == "nuclei"
    t = ans["frame"]
    feats = sess.feats(t)
    size = "nucleus_area" if nuc else P.size_key(sess.ndim)
    axes = {}
    if sess.ndim == 2 and ans["cell_ids"] and t in sess.seg.files:      # length / width of each cell (px)
        from skimage.measure import regionprops
        want = set(ans["cell_ids"][:300])
        for r in regionprops(sess.seg.label(t)):
            if r.label in want:
                axes[r.label] = (r.axis_major_length, r.axis_minor_length)
    keys = (["nucleus_area", "nucleus_perimeter", "nucleus_circularity", "nucleus_aspect_ratio", "nucleus_solidity",
             "nucleus_intensity", "host_cell"] if nuc else
            ["area", "perimeter", "circularity", "aspect_ratio", "solidity", "mean_intensity", "n_nuclei"]
            if sess.ndim == 2 else
            ["volume", "surface_area", "sphericity", "aspect_ratio", "solidity", "mean_intensity"])
    states_off = any(s.get("tool") == "cell_states" and '"available": false' in str(s.get("out", "")).lower()
                     for s in getattr(full, "trace", []))
    if states_off and not ans.get("count"):          # "not available" answers: no cell list (it confused users)
        ans = dict(ans, cell_ids=[])
    cells = []
    st = getattr(full, "states", None) if not nuc else None
    for i in ans["cell_ids"][:300]:
        f = feats.get(i) or {}
        c = dict(id=i, size=round(f[size], 2) if size in f else None)
        if st is not None:
            s_ = st.at(t, i)
            if s_ and s_["state"] != "unknown":
                c["state"], c["state_confidence"] = s_["state"], round(s_["confidence"], 2)
            ev = st.cells.get(i) or {}
            if "dying_from_frame" in ev:
                c["dying_from_frame"], c["death_type"] = ev["dying_from_frame"], ev["death_type"]
        for k in keys:
            if k in f and f[k] is not None:
                c[k] = round(float(f[k]), 3)
        if i in axes:
            c["length"], c["width"] = round(float(axes[i][0]), 1), round(float(axes[i][1]), 1)
        for k in ("cx", "cy", "cz"):
            if k in f:
                c[k] = round(float(f[k]), 1)
        cells.append(c)
    divs = []
    for p in ans.get("division_parent_ids") or []:
        tr = sess.seg.tracks.get(p)
        if tr:
            kids = tr.get("children") or []
            d = dict(parent=p, last_frame=tr["E"], daughters=kids,
                     frame=min((sess.seg.tracks[k]["B"] for k in kids if k in sess.seg.tracks), default=tr["E"] + 1))
            if nuc:
                d.update(kind="nuclear division", **sess.division_info(p))
            divs.append(d)
    used_states = any(s.get("tool") == "cell_states" for s in getattr(full, "trace", []))
    return dict(kind=ans.get("kind", "answer"), text=ans.get("text"), options=ans.get("options"),
                frame=t, cell_ids=ans["cell_ids"], count=ans["count"], cells=cells, divisions=divs,
                states_used=used_states, states_info=(st.info if st is not None and used_states else None),
                states_summary=(st.summary() if st is not None and used_states else None),
                note=ans.get("note") or "", size_unit="um^3" if sess.ndim == 3 else "px^2",
                layer=layer, layer_label=LAYER_LABEL.get(layer, layer), object="nucleus" if nuc else "cell")


STATE_WORDS = re.compile(r"abnormal|dying|dead|death|died|apopto|necro|cancer|malignan|healthy", re.I)


def plan_states_offline(sess, q):
    """keyword stand-in for Qwen (tests): cell_states, then submit the cells with signs of dying."""
    out = sess.call("cell_states", {})
    if re.search(r"cancer|malignan|healthy", q, re.I):
        note = ("CelloSyncra describes cell behaviour (division, signs of dying) and cannot tell whether cells are "
                "cancerous or healthy.")
        sess.call("submit_answer", {"cell_ids": [], "count": 0, "note": note})
        return {}
    if out.get("available") is False:
        sess.call("submit_answer", {"cell_ids": [], "count": 0, "note": out["note"]})
        return {}
    dy = out.get("cells_with_signs_of_dying")
    cs = sess.layers["cells"]
    if dy is None:
        sess.call("submit_answer", {"cell_ids": [], "count": 0,
                                    "note": "Signs of dying are not available: " +
                                            str(out["availability"]["dying_apoptosis_necrosis"])})
        return {}
    ids = [d["cell"] for d in dy if d["cell"] in cs.feats(cs.t)]
    note = out.get("summary_text") or f"{len(dy)} cells show signs of dying."
    sess.call("submit_answer", {"cell_ids": ids, "count": len(dy), "note": note})
    return {}


def history_path(sid):
    m = get_meta(sid)
    if m["kind"] == "upload":
        return seq_root(sid) / "history.jsonl"
    p = Path(CFG.work) / "library_history" / f"{sid}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _turn_text(r):
    """an earlier answer, written back for the LLM (short)."""
    a = r.get("answer") or {}
    if r.get("error") and not a:
        return "(that question failed: " + str(r["error"])[:200] + ")"
    if a.get("kind") in ("reply", "ask"):
        txt = a.get("text") or ""
        return txt + (("\nOptions offered: " + "; ".join(a.get("options") or [])) if a.get("options") else "")
    if not a:
        return "(no answer was given)"
    ids = a.get("cell_ids") or []
    parts = [f"[answer shown to the user] frame {a.get('frame')}: {len(ids)} {a.get('object', 'cell')}s "
             f"{ids[:25]}{' ...' if len(ids) > 25 else ''} (count {a.get('count')}), layer {a.get('layer')}"]
    if a.get("divisions"):
        parts.append("divisions: " + ", ".join(f"{d['parent']}->{d['daughters']} at frame {d['frame']}" for d in a["divisions"][:15]))
    if a.get("note"):
        parts.append("note: " + a["note"])
    calls = [s.get("tool") for s in r.get("trace") or []]
    if calls:
        parts.append("tools used: " + ", ".join(calls))
    return "\n".join(parts)[:1500]


def prior_turns(sid, conv, max_turns=6):
    if not conv:
        return []
    p = history_path(sid)
    if not p.exists():
        return []
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("conversation") == conv and not r.get("stopped") and not r.get("hidden")][-max_turns:]
    msgs = []
    for r in recs:
        msgs += [{"role": "user", "content": r.get("question", "")}, {"role": "assistant", "content": _turn_text(r)}]
    return msgs


def video_facts(sid, sess, body):
    m = get_meta(sid)
    cpf = m.get("cells_per_frame") or []
    fr = m.get("frames") or [sess.frames[0], sess.frames[-1]]
    lines = [f"- video: {m.get('name')} ({'3D' if m.get('ndim') == 3 else '2D'} time-lapse), frames {fr[0]}-{fr[1]} "
             f"({m.get('n_frames') or len(sess.frames)} frames)",
             "- the WHOLE video was segmented and every cell tracked over all frames (sizes, shapes, positions, "
             "brightness, tracks and divisions are available for every frame, not only the one on screen)"]
    lines.append("- tracking convention (Cell Tracking Challenge): a cell keeps its id until it divides; its two "
                 "daughters get new ids (their parent is recorded). So a cell that divides is not 'followed for the "
                 "whole video' under one id, but its lineage is; say this when it matters")
    if cpf:
        lines.append(f"- cells per frame: about {round(sum(cpf) / len(cpf))} (min {min(cpf)}, max {max(cpf)})")
    lines.append("- outline layers: " + ", ".join(LAYER_LABEL.get(k, k) for k in sess.layers) +
                 ("" if "nuclei" in sess.layers else f" (no nuclei: {sess.nuclei_reason})" if sess.nuclei_reason else ""))
    st = sess.states
    if st is not None:
        try:
            lines.append("- cell states (dividing / signs of dying) were computed for the whole video: " +
                         (st.summary().get("text") or ""))
        except Exception:
            lines.append("- cell states were computed for the whole video")
    else:
        lines.append("- cell states: not available (" + (sess.states_reason or "not computed") + ")")
    if m.get("ndim") == 2:
        xs = ["segmentation heatmap (pixel-level Grad-CAM++, tool heatmap) for any cell"]
        if st is not None:
            xs.append("SHAP reasons (explain_cell_state) and cell-state heatmap (patch attribution, tool heatmap)" +
                      ("" if load_xai(sid) is not None else " - not made yet for these states: start_analysis cell_states "
                                                             "recomputes them with explanations"))
        lines.append("- explanations available: " + "; ".join(xs))
    if "nuclei" not in sess.layers and m.get("ndim") == 2 and not lib_skip_reason(m, "nuclei") and CFG.model_nuclei \
            and (m.get("nuclei_check") or {}).get("state") not in ("pending", "running", "no"):
        lines.append("- nuclei were not searched yet (start_analysis nuclei)")
    lines.append("- downloads: the menu above the image has the cell table (one row per cell per frame), masks and "
                 "cell states; the export_table tool makes a download button in your answer for: " +
                 "; ".join(f"{k} ({v})" for k, v in TABLES.items()) + ". Do not claim a table contains something it "
                 "does not.")
    lines.append("- the user can rate answers (thumbs up/down) and correct a cell's state on its card")
    vt = body.get("view_frame")
    if vt is not None:
        lines.append(f"- the user is looking at frame {vt}")
    if body.get("selected_cell") is not None:
        lines.append(f"- the user has selected cell {body.get('selected_cell')}")
    roi = body.get("roi")
    if roi and isinstance(roi, dict) and len(roi.get("box") or []) == 4:
        lines.append(box_fact(sess, m, roi))
    return "\n".join(lines)


def box_fact(sess, m, roi):
    """what the user's drawn box contains, so the assistant can answer 'this cell' / 'in the box' directly."""
    x0, y0, x1, y1 = (int(round(float(v))) for v in roi["box"])
    t = int(roi.get("t", 0))
    head = f"- the user drew a box x {x0}-{x1}, y {y0}-{y1} (pixels) on frame {t}; 'the box', 'this cell', 'here' mean it"
    try:
        s = sess.layer("cells") if hasattr(sess, "layer") else sess
        lab = np.asarray(s.seg.label(t))
        H, W = lab.shape
        x0, x1, y0, y1 = max(0, min(x0, x1)), min(W, max(x0, x1)), max(0, min(y0, y1)), min(H, max(y0, y1))
        sub = lab[y0:y1, x0:x1]
        ids, cnt = np.unique(sub[sub > 0], return_counts=True)
        if not len(ids):
            ex = (" Missed cells were not looked for yet in this video: tell the user and call start_analysis with "
                  "what='find_missed_cells' right away (it segments the video again in place, also with the general "
                  "Cellpose-SAM model, a few minutes; the chat stays)." if m.get("kind") == "upload" and not
                  (m.get("multiscale") == "on" and m.get("ms_version") == MS_VERSION) else
                  " The search for missed cells was already done for this video: the models cannot find this cell."
                  if m.get("multiscale") == "on" else " This is a fixed example video: it cannot be segmented again.")
            return (head + f". NO segmented cell lies in this box (0% of it is covered): if the user sees a cell "
                    f"there, the segmentation MISSED it (e.g. it is much larger or smaller than the typical cell size "
                    f"of {m.get('diameter', '?')} px used here). Say so plainly; do not invent a cell id." + ex)
        parts = []
        for i, c in sorted(zip(ids.tolist(), cnt.tolist()), key=lambda z: -z[1])[:8]:
            full = int((lab == i).sum())
            parts.append(f"cell {i} ({round(100 * c / full)}% of it inside the box)")
        cov = round(100 * float((sub > 0).mean()))
        whole = sum(1 for i, c in zip(ids.tolist(), cnt.tolist()) if c >= 0.9 * int((lab == i).sum()))
        frag = ""
        if whole >= 4 and m.get("kind") == "upload" and not (m.get("multiscale") == "on" and m.get("ms_version") == MS_VERSION):
            frag = (f" {whole} small objects lie entirely inside the box: if the user sees ONE big cell there, it was cut "
                    "into pieces; say so and call start_analysis what='find_missed_cells' (segments again in place, "
                    "keeps a big cell whole).")
        return head + f". Segmented cells in it: {', '.join(parts)}; cells cover {cov}% of the box. Use these ids." + frag
    except Exception as e:
        return head + f" (could not read the masks: {type(e).__name__})"


STOPPED = {}            # run id -> "stop" | "discard" (the user pressed Stop, or edited the question while it ran)
HIDE_LATER = set()      # ids hidden by the user before their (stopped) record was written


@app.post("/api/seqs/{sid}/answers/hide", dependencies=AUTH)
def hide_answers(sid: str, body: dict):
    """Retry / edit / 'ask again' replace earlier turns, like a chat assistant: the given question ids (and in the
    page: everything after them) disappear from the chat and from the assistant's memory. Kept in the file with
    hidden=true, so nothing is lost for the thesis logs."""
    ids = {str(x)[:40] for x in (body.get("ids") or []) if x}
    if not ids:
        return {"hidden": 0}
    p = history_path(sid)
    n = 0
    with LOCK:
        if p.exists():
            lines = []
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    if r.get("id") in ids and not r.get("hidden"):
                        r["hidden"] = True
                        n += 1
                    lines.append(json.dumps(r, default=float))
            p.write_text("\n".join(lines) + ("\n" if lines else ""))
        HIDE_LATER.update(ids)
        while len(HIDE_LATER) > 2000:
            HIDE_LATER.pop()
    return {"hidden": n}


class StoppedByUser(Exception):
    pass


@app.post("/api/seqs/{sid}/ask/{run_id}/stop", dependencies=AUTH)
def ask_stop(sid: str, run_id: str, body: dict | None = None):
    """Stop a running question. The assistant stops before its next model call or tool step.
    discard=true (the question was edited): the stopped run is not kept in the history."""
    STOPPED[str(run_id)[:40]] = "discard" if (body or {}).get("discard") else "stop"
    while len(STOPPED) > 500:
        STOPPED.pop(next(iter(STOPPED)))
    return {"ok": True}


@app.post("/api/seqs/{sid}/ask", dependencies=AUTH)
def ask(sid: str, body: dict):
    q = str(body.get("question", "")).strip()
    if not q:
        raise HTTPException(400, "empty question")
    if len(q) > 2000:
        raise HTTPException(400, "question too long (max 2000 characters)")
    planner = body.get("planner") or CFG.planner
    route = str(body.get("layer") or "auto").strip().lower()
    run_id = str(body.get("run_id") or uuid.uuid4().hex[:10])[:40]
    sess, lk = get_session(sid)
    if run_id in STOPPED:                              # stopped while it was waiting for the previous question
        return dict(id=run_id, question=q, stopped=True, discarded=STOPPED.get(run_id) == "discard", answer=None)
    with lk:                                           # a Session holds state: one question at a time
        sess.reset(route if route in ("auto", "cells", "nuclei") else "auto")
        examples = []
        if planner != "offline" and CFG.examples and FB is not None:
            try:
                examples = FB.examples(q, ndim=sess.ndim)
            except Exception:
                traceback.print_exc()
        sess.__dict__["fewshot"] = examples
        conv = str(body.get("conversation") or "")[:40]
        sess.__dict__["prior_msgs"] = prior_turns(sid, conv) if planner != "offline" else []
        sess.__dict__["video_facts"] = video_facts(sid, sess, body)
        sess.__dict__["allow_text_reply"] = True
        sess.__dict__["stop_check"] = lambda: run_id in STOPPED
        sess.__dict__["heatmaps"] = []
        sess.__dict__["exports"] = []
        try:                                              # start where the user is looking
            vt = int(body.get("view_frame"))
            if vt in sess.layer("cells").frames:
                for s_ in sess.layers.values():
                    s_.tool_set_frame(vt)
        except (TypeError, ValueError):
            pass
        t0, err, usage = time.time(), None, {}
        try:
            if planner == "offline" and STATE_WORDS.search(q):
                usage = plan_states_offline(sess, q)
            elif planner == "offline":
                if sess.route == "auto":
                    sess.call("use_layer", {"layer": route_offline(q)})
                if "nuclei" in sess.layers and re.search(r"\bnucle", q.lower()):
                    sess.layers["nuclei"].alias_ok = True       # keyword planner says "area": means nucleus area
                usage = P.plan_offline(sess, q) or {}
            else:
                usage = P.plan_openai(sess, q, CFG.llm_model, CFG.llm_url, max_steps=CFG.max_steps)
                if sess.answer is None and (usage or {}).get("final_text", "").strip():
                    sess.call("reply", {"text": usage["final_text"].strip()})   # it answered in plain words
        except StoppedByUser:
            usage = usage or getattr(sess, "usage", {}) or {}
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            usage = usage or getattr(sess, "usage", {}) or {}
            if "Connection" in err or "connect" in err.lower():
                err = f"the assistant (Qwen) is not reachable at {CFG.llm_url} - is the LLM server running? ({err})"
        stopped = STOPPED.pop(run_id, None)
        sess.__dict__["stop_check"] = None
        if stopped:
            rec = dict(id=run_id, time=time.time(), sequence=sid, question=q, conversation=conv, stopped=True,
                       discarded=stopped == "discard", answer=None, error=None, steps=sess.steps,
                       seconds=round(time.time() - t0, 2), trace=sess.trace,
                       planner=CFG.llm_model if planner != "offline" else "offline")
            if run_id in HIDE_LATER:
                rec["hidden"] = True
            if stopped != "discard":                   # an edited question replaces the stopped one: not kept
                with LOCK, open(history_path(sid), "a") as f:
                    f.write(json.dumps(rec, default=float) + "\n")
            return rec
        rec = dict(id=run_id, time=time.time(), sequence=sid, question=q, conversation=conv,
                   planner=CFG.llm_model if planner != "offline" else "offline",
                   seconds=round(time.time() - t0, 2), steps=sess.steps, error=err,
                   answer=(lambda a: (a.update(heatmaps=list(sess.__dict__.get("heatmaps") or []),
                                               exports=list(sess.__dict__.get("exports") or [])) or a) if a else a)(
                       _reply(sess, sess.answer)), trace=sess.trace,
                   layer_choice=sess.route, layers_used=list(sess.used_layers),
                   examples_used=[dict(answer_id=e["answer_id"], similarity=e["similarity"]) for e in examples],
                   usage={k: v for k, v in (usage or {}).items() if isinstance(v, (int, float))})
    # a question about nuclei on a video without them: always give the exact reason (not a paraphrase)
    if "nuclei" not in sess.layers and re.search(r"nucl|nucle", q.lower()) and sess.nuclei_reason:
        rec["nuclei_reason"] = sess.nuclei_reason[:1].upper() + sess.nuclei_reason[1:]
    with LOCK, open(history_path(sid), "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    return rec


@app.get("/api/seqs/{sid}/history", dependencies=AUTH)
def history(sid: str):
    p = history_path(sid)
    items = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    items = [r for r in items if not r.get("hidden")]
    if FB is not None and items:
        fb = FB.ratings({r["id"] for r in items})
        for r in items:
            if r["id"] in fb:
                r["feedback"] = fb[r["id"]]
    return {"items": items}


# =============================================================================
# feedback: thumbs on answers, corrected cell states (cs_feedback.py)
# =============================================================================
def _find_answer(sid, aid):
    p = history_path(sid)
    if p.exists():
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("id") == aid:
                    return r
    raise HTTPException(404, "answer not found")


@app.post("/api/seqs/{sid}/answers/{aid}/feedback", dependencies=AUTH)
def answer_feedback(sid: str, aid: str, body: dict):
    if FB is None:
        raise HTTPException(503, "feedback is off")
    try:
        rating = int(body.get("rating", 0))
    except (TypeError, ValueError):
        rating = 0
    if rating not in (-1, 0, 1):
        raise HTTPException(400, "rating must be -1, 0 or 1")
    m = get_meta(sid)
    rec = FB.add_answer(_find_answer(sid, aid), rating, str(body.get("comment", "")), str(body.get("reviewer", "")), m)
    return {"ok": True, "rating": rec["rating"], "trusted": rec["trusted"]}


@app.post("/api/seqs/{sid}/state_label", dependencies=AUTH)
def state_label(sid: str, body: dict):
    import cs_feedback
    if FB is None:
        raise HTTPException(503, "feedback is off")
    m = get_meta(sid)
    label = str(body.get("label", ""))
    if label not in cs_feedback.STATE_LABELS and label != "retract":
        raise HTTPException(400, "unknown state: " + label)
    try:
        cid, t = int(body["cell"]), int(body["t"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "cell and t are needed")
    scope = "from_here" if body.get("scope") == "from_here" else "frame"
    st = load_states(m)
    now = st.at(t, cid) if st is not None else None
    rec = FB.add_state(dict(sequence=sid, sequence_name=m.get("name"), cell=cid, frame=t, scope=scope, label=label,
                            model_state=(now or {}).get("state"), model_confidence=(now or {}).get("confidence"),
                            modality=(st.info.get("modality") if st is not None else None),
                            comment=str(body.get("comment", ""))[:2000], reviewer=str(body.get("reviewer", ""))[:80]))
    return {"ok": True, "trusted": rec["trusted"], "model_state": rec["model_state"]}


@app.get("/api/seqs/{sid}/state_labels", dependencies=AUTH)
def state_labels(sid: str):
    get_meta(sid)
    return {"items": FB.state_corrections(sid) if FB is not None else []}


@app.get("/api/feedback/summary", dependencies=AUTH)
def feedback_summary():
    return FB.summary() if FB is not None else {"off": True}


@app.get("/api/feedback/states.csv", dependencies=AUTH)
def feedback_states_csv():
    if FB is None:
        raise HTTPException(503, "feedback is off")
    return Response(FB.states_csv(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="cellosyncra_state_corrections.csv"'})


@app.get("/api/feedback/answers.jsonl", dependencies=AUTH)
def feedback_answers():
    if FB is None:
        raise HTTPException(503, "feedback is off")
    return Response(FB.answers_jsonl(), media_type="application/x-ndjson",
                    headers={"Content-Disposition": 'attachment; filename="cellosyncra_answer_feedback.jsonl"'})


@app.get("/api/history", dependencies=AUTH)
def all_history():
    items = []
    with LOCK:
        metas = list(SEQS.values())
    for m in metas:
        try:
            p = history_path(m["id"])
        except HTTPException:
            continue
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    r["sequence_name"] = m["name"]
                    r["backend_label"] = (CFG.models[model_key(m)]["label"] if m["kind"] == "upload"
                                          else BACKEND_NAMES.get(m.get("backend"), ""))
                    r.pop("trace", None)
                    if not r.get("hidden"):
                        items.append(r)
    return {"items": sorted(items, key=lambda r: -r["time"])}


def build_table(sid, kind, sess=None):
    """-> (columns, rows) for the exports; per-frame counts are cached on disk (they read every mask once)."""
    m = get_meta(sid)
    if sess is None:
        full, _ = get_session(sid)
        sess = full.layer("cells")
    seg = sess.seg
    if kind == "counts_per_frame":
        sf = seq_root(sid) / "states" / "states.csv"        # recomputed states -> a new table
        f = seq_root(sid) / "exports" / (f"counts_per_frame_{int(sf.stat().st_mtime)}.csv" if sf.exists() else "counts_per_frame.csv")
        if f.exists():
            with open(f) as fh:
                rows = list(csv.DictReader(fh))
            return list(rows[0].keys()) if rows else [], rows
        div_at = {}
        for i, tr in seg.tracks.items():
            if tr.get("children"):
                div_at[int(tr["E"])] = div_at.get(int(tr["E"]), 0) + 1
        st = load_states(m)
        rows = []
        for t in sess.frames:
            lab = np.asarray(seg.label(t))
            ids, area = np.unique(lab[lab > 0], return_counts=True)
            r = dict(frame=t, cells=len(ids), divisions=div_at.get(t, 0),
                     confluence_percent=round(100.0 * float((lab > 0).mean()), 2),
                     mean_size=round(float(area.mean()), 1) if len(area) else 0)
            if st is not None:
                try:
                    now = [st.at(t, int(c)) or {} for c in ids]
                    r["dividing"] = sum(1 for x in now if x.get("state") == "mitosis")
                    r["signs_of_dying"] = sum(1 for x in now if x.get("state") in ("apoptosis", "necrosis"))
                except Exception:
                    pass
            rows.append(r)
        f.parent.mkdir(parents=True, exist_ok=True)
        cols = list(rows[0].keys()) if rows else []
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        return cols, rows
    if kind == "divisions":
        rows = [dict(parent=i, frame=int(tr["E"]), daughters=" ".join(map(str, sorted(tr["children"]))))
                for i, tr in sorted(seg.tracks.items()) if tr.get("children")]
        return ["parent", "frame", "daughters"], sorted(rows, key=lambda r: r["frame"])
    if kind == "tracks":
        rows = [dict(cell=i, first_frame=int(tr["B"]), last_frame=int(tr["E"]), frames=int(tr["E"]) - int(tr["B"]) + 1,
                     parent=tr.get("P") or 0, daughters=" ".join(map(str, sorted(tr.get("children") or []))))
                for i, tr in sorted(seg.tracks.items())]
        return ["cell", "first_frame", "last_frame", "frames", "parent", "daughters"], rows
    raise HTTPException(400, "unknown table: " + kind)


@app.get("/api/seqs/{sid}/table/{kind}.csv", dependencies=AUTH)
def table_csv(sid: str, kind: str):
    if kind == "cells":
        return cells_csv(sid)
    if kind == "cell_states":
        return states_csv(sid)
    full, lk = get_session(sid)
    with lk:
        cols, rows = build_table(sid, kind, full.layer("cells"))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", get_meta(sid)["name"])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}_{kind}.csv"'})


@app.get("/api/seqs/{sid}/cells.csv", dependencies=AUTH)
def cells_csv(sid: str, layer: str = "cells"):
    full, lk = get_session(sid)
    sess = full.layer(layer)
    layer = layer if layer in full.layers else "cells"
    rows, keys = [], []
    with lk:
        for t in sess.frames:
            for cid in sorted(sess.feats(t)):
                r = _cell_record(sess, t, cid)
                tr = r.pop("track")
                r.pop("ndim", None)
                r.update(track_begin=tr["begin"], track_end=tr["end"], parent=tr["parent"] or 0,
                         daughters=" ".join(map(str, tr["daughters"] or [])))
                rows.append(r)
                keys += [k for k in r if k not in keys]
    first = ["frame", "id"]
    keys = first + [k for k in keys if k not in first]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", get_meta(sid)["name"])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}_{"nuclei" if layer == "nuclei" else "cells"}.csv"'})


@app.get("/api/seqs/{sid}/masks.zip", dependencies=AUTH)
def masks_zip(sid: str, layer: str = "cells"):
    m = get_meta(sid)
    rd = Path(layer_dirs(m).get(layer) or m["res_dir"])
    if m["status"] != "ready" or not rd.is_dir():
        raise HTTPException(409, "no segmentation yet")
    tmp = Path(tempfile.mkstemp(suffix=".zip", dir=CFG.work)[1])
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(rd.iterdir()):
            z.write(f, f"RES/{f.name}")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", m["name"])
    from starlette.background import BackgroundTask
    return FileResponse(tmp, media_type="application/zip",
                        filename=f"{name}_{'nuclei' if layer == 'nuclei' and 'nuclei' in layer_dirs(m) else 'cells'}_masks_CTC.zip",
                        background=BackgroundTask(lambda: tmp.unlink(missing_ok=True)))


@app.get("/api/seqs/{sid}/history.json", dependencies=AUTH)
def history_json(sid: str):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", get_meta(sid)["name"])
    return JSONResponse(history(sid), headers={"Content-Disposition": f'attachment; filename="{name}_answers.json"'})


# =============================================================================
def find_joint_model(out, backup):
    try:
        return P._model_file(out, "cpsam_ft_joint")
    except FileNotFoundError:
        pass
    if backup:
        hits = sorted(p for p in Path(backup).rglob("cpsam_ft_joint*") if p.is_file())
        if hits:
            return str(hits[-1])
    return None


def main(argv=None):
    global CFG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("CTC_OUT", ""), help="results folder of the 2D+3D experiment (models/, seg/)")
    ap.add_argument("--data", default=os.environ.get("CTC_DATA", ""), help="CTC data roots (for the test-video library)")
    ap.add_argument("--work", default=os.environ.get("WEB_WORK", ""), help="where uploads and their results are stored")
    ap.add_argument("--model", default=os.environ.get("WEB_SEG_MODEL", ""), help="thesis Cellpose-SAM model (default: cpsam_ft_joint in --out or --backup)")
    ap.add_argument("--model-qpi", default=os.environ.get("WEB_SEG_MODEL_QPI", ""), help="model fine-tuned with QPI (cpsam_ft_joint_qpi); if given it is the default")
    ap.add_argument("--model-nuclei", default=os.environ.get("WEB_SEG_MODEL_NUCLEI", ""),
                    help="nucleus model for DIC (cpsam_alfi_nuclei): adds the 'DIC video' option at upload")
    ap.add_argument("--model-states", default=os.environ.get("WEB_STATES_MODEL", ""),
                    help="cell-state classifier (CellState/out/train/v7/final/model.pkl): dividing / dying cells")
    ap.add_argument("--min-daughter-life", type=int, default=3,
                    help="tracker fix: a division is kept only if both daughters live this many frames (0 = off)")
    ap.add_argument("--backup", default=os.environ.get("BACKUP", ""))
    ap.add_argument("--llm-url", default=os.environ.get("QWEN_URL", "http://localhost:8000/v1"))
    ap.add_argument("--llm-model", default=os.environ.get("OPEN_LLM", "Qwen/Qwen2.5-72B-Instruct-AWQ"))
    ap.add_argument("--planner", choices=["qwen", "offline"], default="qwen")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--trusted-reviewers", default=os.environ.get("WEB_TRUSTED", ""),
                    help="comma-separated names whose feedback becomes worked examples / labels (default: everyone)")
    ap.add_argument("--no-examples", action="store_true", default=os.environ.get("WEB_EXAMPLES", "1") == "0",
                    help="do not show the planner worked examples from thumbs-up answers (to measure their effect)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--token", default=os.environ.get("WEB_TOKEN", ""), help="access token (default: random)")
    ap.add_argument("--no-token", action="store_true", help="no access token (only on a private machine)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--video-max-frames", type=int, default=200, help="longer videos are thinned to this many frames")
    ap.add_argument("--video-max-side", type=int, default=1024, help="larger video frames are shrunk to this size")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--debug-segmenter", action="store_true", help=argparse.SUPPRESS)   # Otsu threshold, tests only
    CFG = ap.parse_args(argv)
    CFG.gpu = not CFG.cpu
    CFG.examples = not CFG.no_examples
    if not CFG.work:
        CFG.work = str(Path(CFG.out or ".") / "web")
    Path(CFG.work, "seqs").mkdir(parents=True, exist_ok=True)
    global FB
    import cs_feedback
    FB = cs_feedback.FeedbackStore(CFG.work, [x for x in CFG.trusted_reviewers.split(",") if x.strip()])
    if not CFG.debug_segmenter and not CFG.model:
        CFG.model = find_joint_model(CFG.out, CFG.backup)
        if not CFG.model:
            raise SystemExit("ERROR: joint model cpsam_ft_joint not found in --out or --backup; pass --model <file>")
    CFG.models = OrderedDict()
    if CFG.model_qpi:
        if not CFG.debug_segmenter and not Path(CFG.model_qpi).is_file():
            raise SystemExit(f"ERROR: --model-qpi file not found: {CFG.model_qpi}")
        CFG.models["qpi"] = dict(label="With QPI (recommended)", path=CFG.model_qpi)
    CFG.models["thesis"] = dict(label="Original (thesis)" if CFG.model_qpi else "Cellpose-SAM 2D+3D", path=CFG.model)
    CFG.default_model = next(iter(CFG.models))
    if CFG.model_states and not Path(CFG.model_states).is_file():
        raise SystemExit(f"ERROR: --model-states file not found: {CFG.model_states}")
    if CFG.model_nuclei and not CFG.debug_segmenter and not Path(CFG.model_nuclei).is_file():
        raise SystemExit(f"ERROR: --model-nuclei file not found: {CFG.model_nuclei}")
    CFG.token = "" if CFG.no_token else (CFG.token or secrets.token_urlsafe(16))
    load_uploads()
    library()
    threading.Thread(target=worker, daemon=True).start()
    host = os.uname().nodename
    q = f"/?token={CFG.token}" if CFG.token else "/"
    print("=" * 78, flush=True)
    print(f"CelloSyncra web app on {host}:{CFG.port}", flush=True)
    for k, v in CFG.models.items():
        print(f"  cell finder {k:7s}: {v['path'] or 'debug (Otsu)'}  [{v['label']}]{' (default)' if k == CFG.default_model else ''}", flush=True)
    if CFG.model_nuclei:
        print(f"  nucleus finder     : {CFG.model_nuclei}  [DIC videos: 'nuclei' layer]", flush=True)
    if CFG.model_states:
        print(f"  cell states        : {CFG.model_states}  [dividing: DIC/phase; dying: QPI]", flush=True)
    print(f"  tracker fix        : divisions kept only if daughters live >= {CFG.min_daughter_life} frames", flush=True)
    print(f"  assistant          : {CFG.llm_model if CFG.planner == 'qwen' else 'offline'} @ {CFG.llm_url}", flush=True)
    print(f"  uploads            : {CFG.work}", flush=True)
    print(f"  open               : http://localhost:{CFG.port}{q}", flush=True)
    print("=" * 78, flush=True)
    import uvicorn
    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level="warning")


if __name__ == "__main__":
    main()