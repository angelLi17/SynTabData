"""Check size, point spacing, wedges per face and missing wedge ids. Usage: python analysis/check_tablets.py OUT_DIR"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from common import find_tablets, load_annotations, load_labels, load_mesh, unique_edges


def _parse_range(text):
    lo, _, hi = text.partition("-")
    return float(lo), float(hi or lo)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("--expect", nargs=3, metavar=("WIDTH", "LENGTH", "THICK"),
                    default=["40", "50-70", "20"], help="expected size in mm, each a value or lo-hi")
    ap.add_argument("--spacing", type=float, default=0.1, help="expected mean edge length in mm")
    ap.add_argument("--tolerance", type=float, default=0.10, help="allowed relative deviation")
    ap.add_argument("--base-label", type=int, default=1)
    ap.add_argument("--label-offset", type=int, default=1, help="label = wedge_id + offset")
    args = ap.parse_args()

    exp = sorted([_parse_range(args.expect[2]), _parse_range(args.expect[0]), _parse_range(args.expect[1])])
    tol = args.tolerance
    face_total = Counter()
    problems = 0

    for t in find_tablets(args.out_dir):
        print(f"\n=== {t['name']} ===")
        verts, faces = load_mesh(t["mesh"])
        extents = np.sort(verts.max(axis=0) - verts.min(axis=0))
        edges = unique_edges(faces)
        spacing = float(np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1).mean())

        size_ok = all(lo * (1 - tol) <= e <= hi * (1 + tol) for e, (lo, hi) in zip(extents, exp))
        spacing_ok = abs(spacing - args.spacing) <= tol * args.spacing * 2
        print(f"size mm (thick, wide, long): {extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f}"
              f"   {'OK' if size_ok else 'OUTSIDE EXPECTED'}")
        print(f"vertices: {len(verts):,}   mean point spacing: {spacing:.3f} mm"
              f"   {'OK' if spacing_ok else 'CHECK'}")
        problems += (not size_ok) + (not spacing_ok)

        if t["labels"] is None:
            print("no label file")
            continue
        labels = load_labels(t["labels"])
        if len(labels) != len(verts):
            print(f"LABEL COUNT MISMATCH: {len(labels)} labels for {len(verts)} vertices")
            problems += 1
            continue
        present = np.unique(labels[labels != args.base_label]) - args.label_offset
        max_id = int(present.max()) if len(present) else 0
        missing = sorted(set(range(1, max_id + 1)) - set(present.tolist()))
        wedge_frac = float((labels != args.base_label).mean())
        print(f"wedges with a labelled region: {len(present)} of at least {max_id}"
              f"   ({wedge_frac:.1%} of vertices are wedge)")
        print(f"missing wedge ids (up to the highest id seen): {len(missing)}")

        if t["annotations"] is None:
            continue
        wedges, signs = load_annotations(t["annotations"])
        faces_here = Counter(w["face"] for w in wedges)
        face_total.update(faces_here)
        n = sum(faces_here.values())
        print("wedges per face: " + ", ".join(f"{k} {v} ({v / n:.0%})" for k, v in sorted(faces_here.items())))

        if missing:
            sign_of = {}
            for s in signs:
                for wid in s["wedge_ids"]:
                    sign_of[wid] = s
            ordered = sorted(signs, key=lambda s: min(s["wedge_ids"]))
            partial, whole, near_face = 0, 0, Counter()
            for wid in missing:
                owner = next((s for s in ordered if min(s["wedge_ids"]) < wid < max(s["wedge_ids"])), None)
                if owner is not None:
                    partial += 1
                    near_face.update(owner["faces"][:1])
                else:
                    whole += 1
                    before = [s for s in ordered if max(s["wedge_ids"]) < wid]
                    if before:
                        near_face.update(before[-1]["faces"][:1])
            print(f"  inside a sign that is otherwise present: {partial}")
            print(f"  at a sign boundary or in a sign with no labels at all: {whole}")
            print("  face of the affected or preceding sign: "
                  + ", ".join(f"{k} {v}" for k, v in near_face.most_common()))

    if face_total:
        n = sum(face_total.values())
        print("\n=== all tablets ===")
        print("wedges per face: " + ", ".join(f"{k} {v} ({v / n:.0%})" for k, v in sorted(face_total.items())))
    print(f"\n{problems} size or spacing problem(s) found")


if __name__ == "__main__":
    main()
