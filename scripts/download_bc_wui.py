"""Download BC Wildfire Wildland-Urban Interface (WUI) layers as GeoJSON.

Layer 1  PROT_WUI_HMN_INTRFCE_BUFFR_SP: Human Interface Buffer (buffers around structures).
Layer 30 PROT_WUI_ELIGIBLE_BUFFER_SP:   WUI 1 km eligible buffer.
These are current (static) products, used as a human-access / settlement proxy.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import urllib.parse
import urllib.request

SERVICE = "https://delivery.maps.gov.bc.ca/arcgis/rest/services/whse/bcgw_pub_WHSE_LAND_AND_NATURAL_RESOURCE/MapServer"
ROOT = Path(__file__).resolve().parents[1]
LAYERS = {1: "wui_human_interface_buffer", 30: "wui_1km_buffer"}


def query(layer, **params):
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(f"{SERVICE}/{layer}/query", data=data, timeout=300) as r:
        result = json.load(r)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offset-deg", type=float, default=0.0005)
    p.add_argument("--page", type=int, default=50)
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/static")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for layer, name in LAYERS.items():
        snap = query(layer, where="1=1", returnIdsOnly="true", f="json")
        ids, oid = sorted(set(snap["objectIds"])), snap["objectIdFieldName"]
        feats = []
        for s in range(0, len(ids), a.page):
            req = ids[s:s + a.page]
            page = query(layer, objectIds=",".join(map(str, req)), outFields="*", returnGeometry="true",
                         outSR=4326, maxAllowableOffset=a.offset_deg, f="geojson")
            got = [f["properties"][oid] for f in page.get("features", [])]
            if set(got) != set(req):
                raise RuntimeError(f"Incomplete page for layer {layer}")
            feats.extend(page["features"])
        path = a.output_dir / f"{name}.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
        (a.output_dir / f"{name}_manifest.json").write_text(json.dumps(dict(
            source=f"{SERVICE}/{layer}", records=len(feats), offset_deg=a.offset_deg,
            downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat()), indent=2))
        print(f"Saved {len(feats):,} features to {path}")


if __name__ == "__main__":
    main()
