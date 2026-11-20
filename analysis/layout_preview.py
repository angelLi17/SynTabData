"""Preview where a config puts the writing, without carving. Usage: python analysis/layout_preview.py W3DGE_EDGE_CONFIG [W3DGE_CURVED_CONFIG ...]"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfgmod
from directions import DIRECTION
from layout3d import layout_signs_on_superquadric
from paleocodage import parse_signs
from signs import drop_signs_without_wedges, resolve_sign_codes


def simulate(cfg, n_tablets, seed, paleocodes):
    """Lay out signs on n_tablets random tablets; return face counts, per-wedge sign bend, wedges per tablet."""
    rng = np.random.default_rng(seed)
    g = cfg.geometry
    faces, bend, counts = Counter(), [], []
    for _ in range(n_tablets):
        h, w, d = (float(rng.uniform(*r)) for r in (g.height_range, g.width_range, g.depth_range))
        eps, eta = float(rng.uniform(*g.epsilon_range)), float(rng.uniform(*g.eta_range))
        codes = resolve_sign_codes("paleocode", None, paleocodes, rng, cfg.generation)
        signs, _ = drop_signs_without_wedges(parse_signs(codes, config=cfg.paleocodage))
        placed = layout_signs_on_superquadric(
            signs, a=h, b=w, c=d, epsilon=eps, eta=eta,
            config=cfg.layout, rng=rng, allowed_faces=set(DIRECTION),
        )
        counts.append(len(placed))
        normals = {}
        for p in placed:
            faces[p.face.name] += 1
            normals.setdefault(p.sign_index, []).append(np.asarray(p.normal3, float))
        for ns in normals.values():
            ns = np.array(ns)
            ns /= np.linalg.norm(ns, axis=1, keepdims=True)
            angle = np.degrees(np.arccos(np.clip(ns @ ns.T, -1.0, 1.0))).max()
            bend += [angle] * len(ns)
    return faces, np.array(bend), float(np.mean(counts))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+", help="names of TabletConfig presets in config.py")
    ap.add_argument("-n", "--tablets", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--paleocodes", default="paleocodes.json")
    args = ap.parse_args()

    print(f"{'config':<26}{'wedges':>7}{'front+back':>11}{'top+bot':>8}{'ends':>6}"
          f"{'bend<5':>8}{'>10':>6}{'>20':>6}{'>30':>6}")
    for name in args.configs:
        cfg = getattr(cfgmod, name)
        faces, bend, per_tablet = simulate(cfg, args.tablets, args.seed, args.paleocodes)
        total = sum(faces.values())
        share = lambda *k: sum(faces.get(x, 0) for x in k) / total
        print(f"{name:<26}{per_tablet:7.0f}{share('FRONT', 'BACK'):11.0%}{share('TOP', 'BOTTOM'):8.0%}"
              f"{share('LEFT', 'RIGHT'):6.0%}{np.mean(bend < 5):8.0%}{np.mean(bend > 10):6.0%}"
              f"{np.mean(bend > 20):6.0%}{np.mean(bend > 30):6.0%}")


if __name__ == "__main__":
    main()
