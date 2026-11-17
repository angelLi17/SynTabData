"""Cut tablets into fixed-size training patches. Usage: python analysis/make_patches.py OUT_DIR -o patches"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from scipy.spatial import cKDTree

from common import find_tablets, load_labels, load_mesh


def vertex_normals(verts, faces):
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(norm, 1e-12)


def sample_patches(verts, normals, labels, *, base_label, radius, n_points, n_patches, wedge_frac, rng):
    tree = cKDTree(verts)
    is_wedge = labels != base_label
    n_wedge = int(round(n_patches * wedge_frac)) if is_wedge.any() else 0
    pools = [(np.flatnonzero(is_wedge), n_wedge), (np.flatnonzero(~is_wedge), n_patches - n_wedge)]

    xyz, nrm, inst, centers = [], [], [], []
    for pool, count in pools:
        if count <= 0 or len(pool) == 0:
            continue
        for c in rng.choice(pool, size=count, replace=len(pool) < count):
            idx = np.asarray(tree.query_ball_point(verts[c], radius))
            if len(idx) < n_points // 4:
                continue
            idx = rng.choice(idx, size=n_points, replace=len(idx) < n_points)
            xyz.append(((verts[idx] - verts[c]) / radius).astype(np.float32))
            nrm.append(normals[idx].astype(np.float32))
            inst.append(labels[idx].astype(np.int32))
            centers.append(verts[c].astype(np.float32))
    if not xyz:
        return None
    inst = np.stack(inst)
    return {
        "xyz": np.stack(xyz),
        "normals": np.stack(nrm),
        "instance": inst,
        "wedge": (inst != base_label).astype(np.uint8),
        "centers": np.stack(centers),
    }


def write_preview(path, xyz, wedge):
    colors = np.where(wedge[:, None] > 0, [220, 60, 40], [170, 170, 170])
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n"
                f"element vertex {len(xyz)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(xyz, colors):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", nargs="?", help="SynTab output folder containing plys/ and labels/")
    ap.add_argument("--meshes", help="folder of .ply meshes (instead of out_dir)")
    ap.add_argument("--labels", help="folder of instance label files")
    ap.add_argument("--label-suffix", default=".txt")
    ap.add_argument("--base-label", type=int, default=1)
    ap.add_argument("--radius", type=float, default=5.0, help="patch radius in mesh units (mm)")
    ap.add_argument("--points", type=int, default=4096, help="points kept per patch")
    ap.add_argument("--per-tablet", type=int, default=200, help="patches per tablet")
    ap.add_argument("--wedge-frac", type=float, default=0.5, help="share of patches centred on a wedge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="also write the first patch as a coloured PLY")
    ap.add_argument("-o", "--output", default="patches")
    args = ap.parse_args()

    if args.out_dir:
        items = [(t["name"], t["mesh"], t["labels"]) for t in find_tablets(args.out_dir)]
    elif args.meshes and args.labels:
        items = []
        for ply in sorted(glob.glob(os.path.join(args.meshes, "*.ply"))):
            name = os.path.splitext(os.path.basename(ply))[0]
            items.append((name, ply, os.path.join(args.labels, name + args.label_suffix)))
    else:
        ap.error("give out_dir, or both --meshes and --labels")

    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    total = 0
    for name, ply, lab in items:
        if not lab or not os.path.exists(lab):
            print(f"{name}: no label file, skipped")
            continue
        verts, faces = load_mesh(ply)
        labels = load_labels(lab)
        if len(labels) != len(verts):
            print(f"{name}: {len(labels)} labels for {len(verts)} vertices, skipped")
            continue
        patches = sample_patches(
            verts, vertex_normals(verts, faces), labels,
            base_label=args.base_label, radius=args.radius, n_points=args.points,
            n_patches=args.per_tablet, wedge_frac=args.wedge_frac, rng=rng,
        )
        if patches is None:
            print(f"{name}: no patches")
            continue
        np.savez_compressed(os.path.join(args.output, name + ".npz"), radius=args.radius, **patches)
        n = len(patches["xyz"])
        total += n
        print(f"{name}: {n} patches, {patches['wedge'].mean():.0%} of points are wedge")
        if args.preview:
            write_preview(os.path.join(args.output, name + "_preview.ply"), patches["xyz"][0], patches["wedge"][0])
    print(f"wrote {total} patches to {args.output}/")


if __name__ == "__main__":
    main()
