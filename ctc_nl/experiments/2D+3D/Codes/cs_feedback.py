"""CelloSyncra: learning from the people who use it.

What users give (stored as JSON lines in <web work>/feedback/, never sent anywhere):
  answers.jsonl   thumbs up / down (+ optional comment) on an assistant answer, with the question and the tool calls
                  that produced it (taken from the server's own history, not from the browser)
  states.jsonl    a corrected cell state for one cell ("this cell is dividing", "not dying" ...) at one frame or from
                  that frame on, next to what the model said there

How it is used:
  1. now   - retrieval: for a new question, the most similar questions that a (trusted) user marked as answered
             well are shown to the LLM planner as worked examples (question -> tool calls). No training: the model
             is unchanged; it just sees good past approaches. Every answer records which examples it was given,
             so the effect can be measured (benchmark with and without examples).
  2. later - the state corrections are new labels for retraining the cell-state classifier (export: states.csv),
             always checked on the locked test before a new model replaces the old one. Thumbs-up/-down pairs can
             later fine-tune the planner (SFT / DPO).
Only the latest rating of an answer counts (a user can change their mind). If a list of trusted reviewers is set,
only their feedback becomes examples or labels.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import threading
import time
from collections import Counter
from pathlib import Path

STATE_LABELS = {"interphase": "not dividing (interphase)", "mitosis": "dividing (mitosis)", "alive": "alive",
                "apoptosis": "apoptosis-like (dying)", "necrosis": "necrosis-like (dying)", "unsure": "not sure"}
STOP = set("""a an the of in on at to for from by with and or is are was were be been it its this that these those
which what who whom how many much does do did show me give list find tell please cell cells frame frames there
any all i you we can could would should between into than then""".split())


def _toks(q: str):
    out = []
    for w in re.findall(r"[a-z]+", q.lower()):
        if w in STOP or len(w) < 2:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                                   # crude plural folding: divisions -> division
        out.append(w)
    return out


def _compact_calls(trace, max_args=140):
    calls = []
    for s in trace or []:
        name, args = s.get("tool"), s.get("args") or {}
        if not name:
            continue
        if name == "submit_answer":
            calls.append("submit_answer(...)")
            continue
        a = ", ".join(f"{k}={json.dumps(v, default=float)}" for k, v in args.items())
        calls.append(f"{name}({a[:max_args]}{'…' if len(a) > max_args else ''})")
    return calls


class FeedbackStore:
    def __init__(self, root: Path, trusted: list[str] | None = None):
        self.dir = Path(root) / "feedback"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fa, self.fs = self.dir / "answers.jsonl", self.dir / "states.jsonl"
        self.lock = threading.Lock()
        self.trusted = {t.strip().lower() for t in (trusted or []) if t.strip()}

    # ------------------------------------------------------------------ writing
    def _append(self, f: Path, rec: dict):
        with self.lock, open(f, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")

    def is_trusted(self, reviewer: str) -> bool:
        return not self.trusted or (reviewer or "").strip().lower() in self.trusted

    def add_answer(self, hist_rec: dict, rating: int, comment: str, reviewer: str, meta: dict):
        rec = dict(time=time.time(), answer_id=hist_rec["id"], sequence=hist_rec.get("sequence"),
                   sequence_name=meta.get("name"), ndim=meta.get("ndim", 2), rating=int(rating),
                   comment=(comment or "")[:2000], reviewer=(reviewer or "")[:80],
                   trusted=self.is_trusted(reviewer), question=hist_rec.get("question"),
                   planner=hist_rec.get("planner"), error=hist_rec.get("error"),
                   had_answer=hist_rec.get("answer") is not None, calls=_compact_calls(hist_rec.get("trace")),
                   trace=[{"tool": s.get("tool"), "args": s.get("args")} for s in hist_rec.get("trace") or []],
                   examples_used=hist_rec.get("examples_used", []))
        self._append(self.fa, rec)
        return rec

    def add_state(self, rec: dict):
        rec = dict(rec, time=time.time(), trusted=self.is_trusted(rec.get("reviewer", "")))
        self._append(self.fs, rec)
        return rec

    # ------------------------------------------------------------------ reading
    def _read(self, f: Path):
        if not f.exists():
            return []
        with self.lock:
            lines = f.read_text().splitlines()
        out = []
        for l in lines:
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
        return out

    def answers(self):
        """latest rating per answer (rating 0 = taken back)."""
        last = {}
        for r in self._read(self.fa):
            last[r["answer_id"]] = r
        return last

    def ratings(self, answer_ids=None):
        a = self.answers()
        return {k: dict(rating=v["rating"], comment=v.get("comment", "")) for k, v in a.items()
                if answer_ids is None or k in answer_ids}

    def state_corrections(self, sid=None):
        last = {}
        for r in self._read(self.fs):
            if sid is None or r.get("sequence") == sid:
                last[(r.get("sequence"), r.get("cell"), r.get("frame"))] = r
        return [r for r in last.values() if r.get("label") != "retract"]

    # ------------------------------------------------------------------ retrieval of worked examples
    def examples(self, question: str, ndim: int = 2, k: int = 3, min_sim: float = 0.3):
        pool = [r for r in self.answers().values()
                if r["rating"] > 0 and r.get("trusted", True) and r.get("had_answer") and not r.get("error")
                and int(r.get("ndim") or 2) == int(ndim) and r.get("calls")]
        if not pool:
            return []
        docs = [_toks(r["question"]) for r in pool]
        df = Counter(w for d in docs for w in set(d))
        n = len(docs)
        idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}

        def vec(ts):
            tf = Counter(ts)
            v = {w: c * idf.get(w, math.log(1 + n) + 1) for w, c in tf.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            return {w: x / norm for w, x in v.items()}
        qv = vec(_toks(question))
        scored = []
        for r, d in zip(pool, docs):
            dv = vec(d)
            s = sum(qv[w] * dv.get(w, 0.0) for w in qv)
            if s >= min_sim:
                scored.append((s, r))
        scored.sort(key=lambda x: -x[0])
        seen, out = set(), []
        for s, r in scored:
            key = r["question"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(answer_id=r["answer_id"], question=r["question"], calls=r["calls"], similarity=round(s, 3)))
            if len(out) >= k:
                break
        return out

    @staticmethod
    def prompt_block(examples):
        if not examples:
            return ""
        lines = ["\n\nWorked examples: earlier questions (often on other videos) that users marked as answered well, with the tool "
                 "calls that answered them. Reuse the approach when it fits this question; the frames, cell ids and "
                 "numbers of THIS video must come from your own tool calls, never from these examples."]
        for i, e in enumerate(examples, 1):
            lines.append(f"{i}. Q: {e['question']}\n   calls: " + " -> ".join(e["calls"]))
        return "\n".join(lines)

    # ------------------------------------------------------------------ summaries / exports
    def summary(self):
        a = list(self.answers().values())
        s = self.state_corrections()
        return dict(answers_rated=len([r for r in a if r["rating"] != 0]), thumbs_up=sum(r["rating"] > 0 for r in a),
                    thumbs_down=sum(r["rating"] < 0 for r in a), usable_examples=sum(
                        r["rating"] > 0 and r.get("trusted", True) and r.get("had_answer") for r in a),
                    state_corrections=len(s), state_corrections_trusted=sum(bool(r.get("trusted")) for r in s),
                    by_label=dict(Counter(r["label"] for r in s)), trusted_reviewers=sorted(self.trusted) or "everyone")

    def states_csv(self):
        cols = ["sequence", "sequence_name", "cell", "frame", "scope", "label", "model_state", "model_confidence",
                "modality", "reviewer", "trusted", "comment", "time"]
        b = io.StringIO()
        w = csv.DictWriter(b, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(self.state_corrections(), key=lambda r: (r.get("sequence", ""), r.get("cell", 0), r.get("frame", 0))):
            w.writerow(r)
        return b.getvalue()

    def answers_jsonl(self):
        return "\n".join(json.dumps(r, default=float) for r in self.answers().values() if r["rating"] != 0) + "\n"
