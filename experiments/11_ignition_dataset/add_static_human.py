"""Static human-access features per dataset cell from BC WUI layers (scripts/download_bc_wui.py).

Output: data/ignition_dataset/res<r>/static_human.parquet with
  s_hib_frac      share of the cell inside the WUI Human Interface Buffer (around structures)
  s_hib_dist_km   distance from cell centre to the nearest Human Interface Buffer polygon (0 inside)
  s_wui1km_frac   share of the cell inside the WUI 1 km eligible buffer
Static, current products: no time dimension (small temporal-leakage risk; settlements change slowly).
Distances/areas use a sinusoidal projection centred on -125 deg (equal-area; local km).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree, make_valid
from shapely.geometry import Polygon, shape

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "data/static"
LON0 = -125.0


def sinusoidal(geom):
    def f(xy):
        lat = xy[:, 1]
        return np.column_stack([(xy[:, 0] - LON0) * np.cos(np.deg2rad(lat)) * 111.320, lat * 110.574])
    return shapely.transform(geom, f)


def load_layer(name):
    feats = json.loads((STATIC / f"{name}.geojson").read_text())["features"]
    geoms = [sinusoidal(make_valid(shape(f["geometry"]))) for f in feats if f["geometry"]]
    return [g for g in geoms if not g.is_empty]


def coverage(cells_poly, layer):
    tree = STRtree(layer)
    out = np.zeros(len(cells_poly))
    for i, cell in enumerate(cells_poly):
        hits = tree.query(cell, predicate="intersects")
        if len(hits):
            out[i] = shapely.union_all([layer[j].intersection(cell) for j in hits]).area / cell.area
    return np.clip(out, 0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolution", type=int, default=6)
    a = p.parse_args()
    d = ROOT / f"data/ignition_dataset/res{a.resolution}"
    grid = pd.read_parquet(d / "grid.parquet")
    polys = [sinusoidal(Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)])) for c in grid["cell"]]
    centres = [p.centroid for p in polys]
    hib = load_layer("wui_human_interface_buffer")
    w1 = load_layer("wui_1km_buffer")
    tree = STRtree(hib)
    nearest = tree.query_nearest(centres, all_matches=False)
    dist = np.array([centres[i].distance(hib[j]) for i, j in zip(*nearest)])
    out = pd.DataFrame({"h3_cell": grid["cell"], "s_hib_frac": coverage(polys, hib).astype("float32"),
                        "s_hib_dist_km": dist.astype("float32"), "s_wui1km_frac": coverage(polys, w1).astype("float32")})
    out.to_parquet(d / "static_human.parquet", index=False)
    print(out.describe())


if __name__ == "__main__":
    main()
