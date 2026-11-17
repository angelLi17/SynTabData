"""Shared loaders for the analysis scripts."""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _read_ply_numpy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read vertices and triangle faces from a PLY file (binary or ASCII)."""
    with open(path, "rb") as f:
        fmt = None
        elements: List[Tuple[str, int, List[str]]] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header found")
            parts = line.decode("ascii", "replace").strip().split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                elements.append((parts[1], int(parts[2]), []))
            elif parts[0] == "property":
                elements[-1][2].append(" ".join(parts[1:]))
            elif parts[0] == "end_header":
                break

        if fmt not in ("binary_little_endian", "ascii"):
            raise ValueError(f"{path}: unsupported PLY format {fmt!r}; install trimesh")

        np_types = {
            "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
            "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
            "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
            "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8",
        }
        verts = faces = None
        for name, count, props in elements:
            is_list = any(p.startswith("list") for p in props)
            if fmt == "ascii":
                rows = [f.readline().split() for _ in range(count)]
                if name == "vertex":
                    verts = np.array([r[:3] for r in rows], dtype=np.float64)
                elif name == "face":
                    faces = np.array([r[1:4] for r in rows], dtype=np.int64)
                continue
            if not is_list:
                dt = np.dtype([(p.split()[1], "<" + np_types[p.split()[0]]) for p in props])
                data = np.frombuffer(f.read(dt.itemsize * count), dtype=dt, count=count)
                if name == "vertex":
                    verts = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
            else:
                if len(props) != 1:
                    raise ValueError(f"{path}: mixed list properties; install trimesh")
                _, cnt_t, idx_t, _ = props[0].split()
                dt = np.dtype([("n", "<" + np_types[cnt_t]), ("v", "<" + np_types[idx_t], (3,))])
                data = np.frombuffer(f.read(dt.itemsize * count), dtype=dt, count=count)
                if count and not np.all(data["n"] == 3):
                    raise ValueError(f"{path}: non-triangle faces; install trimesh")
                if name == "face":
                    faces = data["v"].astype(np.int64)
        if verts is None or faces is None:
            raise ValueError(f"{path}: missing vertex or face element")
        return verts, faces


def load_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vertices [N,3], faces [M,3]) without merging or reordering vertices."""
    try:
        return _read_ply_numpy(path)
    except ValueError:
        import trimesh  # type: ignore
        m = trimesh.load(path, process=False)
        return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def load_labels(path: str, column: int = 1) -> np.ndarray:
    """Read a per-vertex label file: lines of `index label`, `#` lines ignored."""
    data = np.loadtxt(path, dtype=np.int64, comments="#", ndmin=2)
    idx = data[:, 0]
    if not np.array_equal(idx, np.arange(len(idx))):
        order = np.argsort(idx)
        data = data[order]
    return data[:, column]


def unique_edges(faces: np.ndarray) -> np.ndarray:
    """Undirected unique edges [E,2] of a triangle mesh."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def find_tablets(out_dir: str) -> List[Dict[str, Optional[str]]]:
    """List tablets in a SynTab output folder (plys/, labels/, annotations/)."""
    tablets = []
    for ply in sorted(glob.glob(os.path.join(out_dir, "plys", "*.ply"))):
        name = os.path.splitext(os.path.basename(ply))[0]
        lab = os.path.join(out_dir, "labels", name + ".txt")
        ann = os.path.join(out_dir, "annotations", name + ".json")
        tablets.append({
            "name": name,
            "mesh": ply,
            "labels": lab if os.path.exists(lab) else None,
            "annotations": ann if os.path.exists(ann) else None,
        })
    return tablets


def _body_value(body: list, purpose: str) -> Optional[str]:
    for b in body:
        if b.get("purpose") == purpose:
            return b.get("value")
    return None


def load_annotations(path: str) -> Tuple[List[dict], List[dict]]:
    """Return (wedges, signs) from a SynTab WebAnnotation JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    wedges, signs = [], []
    for ann in raw.values():
        body = ann.get("body", [])
        tag = next((b["source"]["label"] for b in body if b.get("purpose") == "tagging"), None)
        if tag == "Wedge":
            wedges.append({
                "sign_index": int(_body_value(body, "Signindex")),
                "wedge_index": int(_body_value(body, "Wedgeindex")),
                "wedge_type": _body_value(body, "Wedgetype"),
                "face": _body_value(body, "Face"),
                "depth": float(_body_value(body, "Depth")),
                "size_class": _body_value(body, "Wedgesize"),
                "wedge_id": None,
            })
        elif tag == "Character":
            ids = _body_value(body, "Wedgeids") or ""
            signs.append({
                "sign_index": int(_body_value(body, "Signindex")),
                "faces": (_body_value(body, "Faces") or "").split(","),
                "wedge_ids": [int(x) for x in ids.split(",") if x],
                "paleocode": _body_value(body, "Paleocode"),
            })

    ids_by_sign = {s["sign_index"]: s["wedge_ids"] for s in signs}
    by_sign: Dict[int, List[dict]] = {}
    for w in wedges:
        by_sign.setdefault(w["sign_index"], []).append(w)
    for sign_index, ws in by_sign.items():
        ids = ids_by_sign.get(sign_index, [])
        ws.sort(key=lambda w: w["wedge_index"])
        if len(ids) == len(ws):
            for w, wid in zip(ws, ids):
                w["wedge_id"] = wid
    return wedges, signs
