"""Download BC historical incident points, checking every requested object ID."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import urllib.parse
import urllib.request

URL = "https://delivery.maps.gov.bc.ca/arcgis/rest/services/whse/bcgw_pub_WHSE_LAND_AND_NATURAL_RESOURCE/MapServer/22/query"
ROOT = Path(__file__).resolve().parents[1]


def query(**params):
    url = URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("Start year must not exceed end year")
    where = f"FIRE_YEAR >= {args.start_year} AND FIRE_YEAR <= {args.end_year}"
    snapshot = query(where=where, returnIdsOnly="true", f="json")
    ids = sorted(set(snapshot.get("objectIds") or []))
    if not ids:
        raise RuntimeError("No incident records returned")
    oid = snapshot["objectIdFieldName"]
    features = []
    for start in range(0, len(ids), 200):
        requested = ids[start:start + 200]
        page = query(objectIds=",".join(map(str, requested)), outFields="*",
                     returnGeometry="true", outSR=4326, f="geojson")
        returned = page.get("features", [])
        found = [feature["properties"][oid] for feature in returned]
        if len(found) != len(requested) or set(found) != set(requested):
            raise RuntimeError("Incomplete or duplicate response; download not saved")
        features.extend(returned)
        print(f"Downloaded {len(features):,} / {len(ids):,} incidents", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bc_historical_incidents_{args.start_year}_{args.end_year}"
    geojson = args.output_dir / f"{stem}.geojson"
    geojson.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    rows = []
    for feature in features:
        row = dict(feature["properties"])
        geometry = feature.get("geometry")
        coordinates = geometry["coordinates"] if geometry else [None, None]
        row["point_longitude"], row["point_latitude"] = coordinates[:2]
        # Preserve source date values; add ISO UTC fields without guessing local dates.
        for field in ["IGNITION_DATE", "FIRE_OUT_DATE"]:
            value = row.get(field)
            row[field + "_UTC"] = ((dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(milliseconds=value)).isoformat()
                                        if isinstance(value, (int, float)) else value)
        rows.append(row)
    csv_path = args.output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(source=URL, where=where, records=len(features), object_id_field=oid,
                    downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    note="Incident point records, not daily fire extent. Date accuracy and timezone semantics require audit.")
    (args.output_dir / f"{stem}_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {csv_path}\nSaved {geojson}")


if __name__ == "__main__":
    main()
