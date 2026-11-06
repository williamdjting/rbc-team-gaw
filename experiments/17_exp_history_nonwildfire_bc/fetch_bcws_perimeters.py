"""Download BC Wildfire Service final fire perimeters for 2018-2023.

Source: DataBC WFS layer WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_FIRE_POLYS_SP.
These are final (end-of-season) perimeters, so they are used only to label past
detections as wildfire-linked, never as forecast inputs. Writes
data/bcws_perimeters_2018_2023.geojson and a manifest.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data/bcws_perimeters_2018_2023.geojson"
WFS = "https://openmaps.gov.bc.ca/geo/pub/wfs"
LAYER = "WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_FIRE_POLYS_SP"
FILTER = "FIRE_YEAR BETWEEN 2018 AND 2023"
PAGE = 500


def request(**params):
    params = {"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeName": LAYER,
              "CQL_FILTER": FILTER, "outputFormat": "json", "srsName": "EPSG:4326", **params}
    with urllib.request.urlopen(f"{WFS}?{urllib.parse.urlencode(params)}", timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    expected = request(count=1)["numberMatched"]
    features = []
    while len(features) < expected:
        page = request(count=PAGE, startIndex=len(features), sortBy="OBJECTID",
                       propertyName="FIRE_NUMBER,FIRE_YEAR,FIRE_CAUSE,FIRE_SIZE_HECTARES,FIRE_DATE,OBJECTID,SHAPE")["features"]
        assert page, f"empty page at {len(features)}"
        features += page
        print(f"  {len(features):,} / {expected:,}", flush=True)
        time.sleep(0.2)
    assert all(2018 <= f["properties"]["FIRE_YEAR"] <= 2023 for f in features), "2024 must never be loaded"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    years = {}
    for f in features:
        years[f["properties"]["FIRE_YEAR"]] = years.get(f["properties"]["FIRE_YEAR"], 0) + 1
    OUT.with_suffix(".manifest.json").write_text(json.dumps({
        "source": WFS, "layer": LAYER, "filter": FILTER, "features": len(features), "by_year": years,
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    print(f"Wrote {len(features):,} perimeters to {OUT}")


if __name__ == "__main__":
    main()
