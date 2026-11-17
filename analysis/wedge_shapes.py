"""Measure wedge length, width and depth. Usage: python analysis/wedge_shapes.py OUT_DIR -o A.csv | --compare A.csv B.csv"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

from common import find_tablets, load_labels, load_mesh


def measure_tablet(verts, labels, base_label, min_vertices):
    """Return rows (label, n_vertices, length, width, depth) for one tablet."""
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    ids, starts = np.unique(sorted_labels, return_index=True)
    ends = np.append(starts[1:], len(labels))
    rows = []
    for lid, s, e in zip(ids.tolist(), starts.tolist(), ends.tolist()):
        if lid == base_label or e - s < min_vertices:
            continue
        pts = verts[order[s:e]]
        centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vt.T
        ext = proj.max(axis=0) - proj.min(axis=0)
        rows.append((lid, e - s, float(ext[0]), float(ext[1]), float(ext[2])))
    return rows


def summarize(name, arr):
    q = np.percentile(arr, [5, 25, 50, 75, 95])
    return f"{name:<10}" + "".join(f"{v:>9.2f}" for v in q)


def compare(path_a, path_b, plot_path):
    def read(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return {k: np.array([float(r[k]) for r in rows]) for k in ("length", "width", "depth")}, len(rows)

    a, na = read(path_a)
    b, nb = read(path_b)
    name_a, name_b = os.path.basename(path_a), os.path.basename(path_b)
    print(f"A = {name_a} ({na} wedges)\nB = {name_b} ({nb} wedges)\n")
    print(f"{'':<10}" + "".join(f"{p:>9}" for p in ("p5", "p25", "median", "p75", "p95")))
    for k in ("length", "width", "depth"):
        print(summarize(k + " A", a[k]))
        print(summarize(k + " B", b[k]))
        ratio = np.median(a[k]) / np.median(b[k])
        print(f"  median A / median B = {ratio:.2f}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping histograms")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, k in zip(axes, ("length", "width", "depth")):
        hi = np.percentile(np.concatenate([a[k], b[k]]), 99)
        bins = np.linspace(0, hi, 40)
        ax.hist(a[k], bins=bins, alpha=0.55, density=True, label=name_a)
        ax.hist(b[k], bins=bins, alpha=0.55, density=True, label=name_b)
        ax.set_xlabel(f"wedge {k} (mesh units)")
        ax.set_ylabel("density")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"saved histograms to {plot_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", nargs="?", help="SynTab output folder containing plys/ and labels/")
    ap.add_argument("--meshes", help="folder of .ply meshes (instead of out_dir)")
    ap.add_argument("--labels", help="folder of instance label files")
    ap.add_argument("--label-suffix", default=".txt")
    ap.add_argument("--base-label", type=int, default=1)
    ap.add_argument("--min-vertices", type=int, default=10,
                    help="ignore wedges with fewer labelled vertices than this")
    ap.add_argument("-o", "--output", default="wedge_shapes.csv")
    ap.add_argument("--compare", nargs=2, metavar=("A.csv", "B.csv"))
    ap.add_argument("--plot", default="wedge_shapes.png")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1], args.plot)
        return

    if args.out_dir:
        items = [(t["name"], t["mesh"], t["labels"]) for t in find_tablets(args.out_dir)]
    elif args.meshes and args.labels:
        items = []
        for ply in sorted(glob.glob(os.path.join(args.meshes, "*.ply"))):
            name = os.path.splitext(os.path.basename(ply))[0]
            items.append((name, ply, os.path.join(args.labels, name + args.label_suffix)))
    else:
        ap.error("give out_dir, or both --meshes and --labels, or --compare")

    all_rows = []
    for name, ply, lab in items:
        if not lab or not os.path.exists(lab):
            print(f"{name}: no label file, skipped")
            continue
        verts, _ = load_mesh(ply)
        rows = measure_tablet(verts, load_labels(lab), args.base_label, args.min_vertices)
        all_rows += [(name,) + r for r in rows]
        print(f"{name}: measured {len(rows)} wedges")

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tablet", "label", "n_vertices", "length", "width", "depth"])
        w.writerows(all_rows)

    if all_rows:
        arr = np.array([r[3:] for r in all_rows])
        print(f"\n{'':<10}" + "".join(f"{p:>9}" for p in ("p5", "p25", "median", "p75", "p95")))
        for i, k in enumerate(("length", "width", "depth")):
            print(summarize(k, arr[:, i]))
    print(f"wrote {len(all_rows)} wedges to {args.output}")


if __name__ == "__main__":
    main()
