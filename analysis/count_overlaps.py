"""Count pairs of wedges whose impressions touch. Usage: python analysis/count_overlaps.py OUT_DIR"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

from common import find_tablets, load_labels, load_mesh, unique_edges


def pairs_for_tablet(faces, inst, base_label, step=None):
    """Return rows (a, b, shared_edges, n_a, n_b, first) for one tablet."""
    edges = unique_edges(faces)
    la, lb = inst[edges[:, 0]], inst[edges[:, 1]]
    touch = (la != lb) & (la != base_label) & (lb != base_label)
    if not np.any(touch):
        return []
    ab = np.sort(np.stack([la[touch], lb[touch]], axis=1), axis=1)
    pairs, counts = np.unique(ab, axis=0, return_counts=True)

    ids, sizes = np.unique(inst, return_counts=True)
    size_of = dict(zip(ids.tolist(), sizes.tolist()))

    rows = []
    for (a, b), k in zip(pairs.tolist(), counts.tolist()):
        if step is None:
            first = a if a < b else b
        else:
            sa, sb = int(step[inst == a].max()), int(step[inst == b].max())
            first = a if sa < sb else (b if sb < sa else -1)  # -1 = same step
        rows.append((a, b, k, size_of[a], size_of[b], first))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", nargs="?", help="SynTab output folder containing plys/ and labels/")
    ap.add_argument("--meshes", help="folder of .ply meshes (instead of out_dir)")
    ap.add_argument("--labels", help="folder of instance label files")
    ap.add_argument("--steps", help="folder of step label files giving impression order")
    ap.add_argument("--label-suffix", default=".txt")
    ap.add_argument("--step-suffix", default="_step.txt")
    ap.add_argument("--base-label", type=int, default=1,
                    help="label of unwritten surface (SynTab uses 1)")
    ap.add_argument("--min-edges", type=int, nargs="+", default=[1, 5, 20],
                    help="report counts at these shared-edge thresholds")
    ap.add_argument("-o", "--output", default="overlap_pairs.csv")
    args = ap.parse_args()

    if args.out_dir:
        items = [(t["name"], t["mesh"], t["labels"], None) for t in find_tablets(args.out_dir)]
    elif args.meshes and args.labels:
        items = []
        for ply in sorted(glob.glob(os.path.join(args.meshes, "*.ply"))):
            name = os.path.splitext(os.path.basename(ply))[0]
            lab = os.path.join(args.labels, name + args.label_suffix)
            stp = os.path.join(args.steps, name + args.step_suffix) if args.steps else None
            items.append((name, ply, lab if os.path.exists(lab) else None, stp))
    else:
        ap.error("give out_dir, or both --meshes and --labels")

    all_rows = []
    print(f"{'tablet':<24}{'wedges':>8}" + "".join(f"{'pairs>=' + str(m):>12}" for m in args.min_edges))
    for name, ply, lab, stp in items:
        if lab is None:
            print(f"{name:<24} (no label file, skipped)")
            continue
        _, faces = load_mesh(ply)
        inst = load_labels(lab)
        step = load_labels(stp) if stp and os.path.exists(stp) else None
        rows = pairs_for_tablet(faces, inst, args.base_label, step)
        n_wedges = len(np.unique(inst[inst != args.base_label]))
        shared = np.array([r[2] for r in rows], dtype=int)
        print(f"{name:<24}{n_wedges:>8}" + "".join(f"{int((shared >= m).sum()):>12}" for m in args.min_edges))
        all_rows += [(name,) + r for r in rows]

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tablet", "wedge_a", "wedge_b", "shared_edges", "verts_a", "verts_b", "first"])
        w.writerows(all_rows)

    shared = np.array([r[3] for r in all_rows], dtype=int)
    print("-" * 44)
    print(f"{'TOTAL':<24}{'':>8}" + "".join(f"{int((shared >= m).sum()):>12}" for m in args.min_edges))
    print(f"wrote {len(all_rows)} pairs to {args.output}")


if __name__ == "__main__":
    main()
