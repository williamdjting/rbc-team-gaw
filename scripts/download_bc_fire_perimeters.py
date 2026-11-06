"""Download BC historical fire perimeters, checking every requested object ID.

Perimeters are final (or latest) burned-area polygons, one per fire. They do not
describe daily growth. Geometry is generalized by `--offset-deg` (default about
50 m), which is far below an H3 resolution-7 cell edge (about 1.2 km).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import urllib.parse
import urllib.request

URL = "https://delivery.maps.gov.bc.ca/arcgis/rest/services/whse/bcgw_pub_WHSE_LAND_AND_NATURAL_RESOURCE/MapServer/6/query"
ROOT = Path(__file__).resolve().parents[1]
FIELDS = "FIRE_NUMBER,VERSION_NUMBER,FIRE_YEAR,FIRE_CAUSE,FIRE_LABEL,FIRE_SIZE_HECTARES,SOURCE,FIRE_DATE,CREATION_METHOD,OBJECTID"


def query(**params):
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(URL, data=data, timeout=300) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--offset-deg", type=float, default=0.0005)
    parser.add_argument("--page", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    where = f"FIRE_YEAR >= {args.start_year} AND FIRE_YEAR <= {args.end_year}"
    snapshot = query(where=where, returnIdsOnly="true", f="json")
    ids = sorted(set(snapshot.get("objectIds") or []))
    if not ids:
        raise RuntimeError("No perimeter records returned")
    oid = snapshot["objectIdFieldName"]
    features = []
    for start in range(0, len(ids), args.page):
        requested = ids[start:start + args.page]
        page = query(objectIds=",".join(map(str, requested)), outFields=FIELDS, returnGeometry="true",
                     outSR=4326, maxAllowableOffset=args.offset_deg, f="geojson")
        returned = page.get("features", [])
        found = [feature["properties"][oid] for feature in returned]
        if len(found) != len(requested) or set(found) != set(requested):
            raise RuntimeError("Incomplete or duplicate response; download not saved")
        for feature in returned:
            value = feature["properties"].get("FIRE_DATE")
            feature["properties"]["FIRE_DATE_RAW_ISO"] = (
                (dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(milliseconds=value)).isoformat()
                if isinstance(value, (int, float)) else value)
        features.extend(returned)
        print(f"Downloaded {len(features):,} / {len(ids):,} perimeters", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bc_fire_perimeters_{args.start_year}_{args.end_year}"
    path = args.output_dir / f"{stem}.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    metadata = dict(source=URL, where=where, records=len(features), object_id_field=oid,
                    max_allowable_offset_deg=args.offset_deg,
                    downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    note="Final/latest burned-area polygons, one per fire; not daily extent.")
    (args.output_dir / f"{stem}_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
