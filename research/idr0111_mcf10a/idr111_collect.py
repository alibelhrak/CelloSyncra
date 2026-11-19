"""idr0111: gather the per-video results (cells/<video>/) into one table + one feature file, and report what is missing.

  python idr111_collect.py --cells cells --out features
  -> features/tracks.csv.gz  (all tracked cells, with uid = video|track, day, line, series, well)
     features/emb.npz        (frozen-encoder features per track, same uid)
     features/summary.md     (videos done / missing, tracks per line and day)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default="cells")
ap.add_argument("--out", default="features")
ap.add_argument("--expected", type=int, default=198)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
T, U, M, S, ST, N, info = [], [], [], [], [], [], []
for d in sorted(Path(a.cells).iterdir()):
    if not (d / "info.json").exists():
        continue
    info.append(json.loads((d / "info.json").read_text()))
    t = pd.read_csv(d / "tracks.csv.gz")
    t["uid"] = t["video"] + "|" + t["track"].astype(str)
    T.append(t)
    if (d / "emb.npz").exists():
        z = np.load(d / "emb.npz")
        U += [f"{d.name}|{k}" for k in z["track"]]
        M.append(z["emb_mean"]); S.append(z["emb_std"]); ST.append(z["emb_step"]); N.append(z["n"])
tr = pd.concat(T, ignore_index=True)
tr.to_csv(out / "tracks.csv.gz", index=False)
np.savez_compressed(out / "emb.npz", uid=np.array(U), emb_mean=np.concatenate(M), emb_std=np.concatenate(S),
                    emb_step=np.concatenate(ST), n=np.concatenate(N))
inf = pd.DataFrame(info)
L = [f"# idr0111 cells: {len(inf)} of {a.expected} videos done", "",
     f"tracked cells (>= min length): {len(tr):,}; with encoder features: {len(U):,}", "",
     "Tracked cells per line and series:", "", tr.groupby(["series", "line"]).size().to_string(), "",
     "Tracked cells per day and line:", "", tr.pivot_table(index="day", columns="line", values="uid", aggfunc="count").fillna(0).astype(int).to_string(), "",
     f"median cell diameter (px) per line: {inf.groupby('line')['diameter_px'].median().round(1).to_dict()}",
     f"median minutes per video: {round(inf['seconds_total'].median() / 60, 1)}"]
(out / "summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
