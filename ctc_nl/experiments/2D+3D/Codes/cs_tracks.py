"""Division clean-up for CelloSyncra tracks (used by the web app and the ALFI evaluation).

The IoU tracker records a division whenever one cell is followed by two nearby cells. A cell outline that
flickers (found as two pieces for a frame or two, then as one again) therefore looks like a division.
A real division gives two daughters that both live on. Rule:

    keep a division only if EVERY daughter lives at least `min_life` frames
    (a daughter that reaches the last frame of the video is given the benefit of the doubt).

A rejected division is undone: the longest-living "daughter" continues the parent's track (same id, so the
cell keeps its identity), the other pieces become tracks of their own without a parent.
Tracks are CTC style: {id: dict(B=first frame, E=last frame, P=parent id or 0)}.
"""
from __future__ import annotations


def clean_divisions(tracks: dict, last_frame: int, min_life: int = 3):
    """Returns (new_tracks, relabel, report).
    relabel: {old_id: new_id} to apply to the masks of frames [old B, old E] (daughter -> parent id)."""
    tr = {int(L): dict(B=int(d["B"]), E=int(d["E"]), P=int(d.get("P") or 0)) for L, d in tracks.items()}
    relabel: dict[int, int] = {}
    undone = 0

    def kids_of(j):
        return sorted((L for L, d in tr.items() if d["P"] == j), key=lambda L: tr[L]["B"])

    def life(L):
        return tr[L]["E"] - tr[L]["B"] + 1

    changed = True
    while changed:
        changed = False
        parents = sorted({d["P"] for d in tr.values() if d["P"] and d["P"] in tr},
                         key=lambda j: tr[j]["E"])
        for j in parents:
            kids = kids_of(j)
            if not kids:
                continue
            ok = len(kids) >= 2 and all(life(k) >= min_life or tr[k]["E"] >= last_frame for k in kids)
            if ok:
                continue
            # undo: the longest-living piece continues the parent
            cont = max(kids, key=lambda k: (life(k), -k))
            for k in kids:
                if k != cont:
                    tr[k]["P"] = 0
            tr[j]["E"] = tr[cont]["E"]
            for g in kids_of(cont):              # the continuation's own daughters now belong to j
                tr[g]["P"] = j
            relabel[cont] = j
            del tr[cont]
            undone += 1
            changed = True
            break                                # the lineage changed: start again in time order
    # follow chains (a -> b -> c becomes a -> c)
    for k in list(relabel):
        v = relabel[k]
        while v in relabel:
            v = relabel[v]
        relabel[k] = v
    kept = len({d["P"] for d in tr.values() if d["P"] and d["P"] in tr})
    return tr, relabel, dict(divisions_kept=kept, divisions_undone=undone, min_life=min_life)


def res_track_text(tracks: dict) -> str:
    return "".join(f"{L} {d['B']} {d['E']} {d['P']}\n" for L, d in sorted(tracks.items()))
