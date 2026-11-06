"""Download BC Wildfire Service historical fire incidents for 2018-2023.

Source: DataBC "Fire Incident Locations - Historical" (PROT_HISTORICAL_INCIDENTS_SP),
ArcGIS layer 603. Only fire years 2018-2023 are requested, so 2024 is never loaded.
Writes data/bcws_incidents_2018_2023.csv and a small JSON manifest.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "data/bcws_incidents_2018_2023.csv"
LAYER = "https://delivery.maps.gov.bc.ca/arcgis/rest/services/mpcm/bcgwpub/MapServer/603/query"
FIELDS = ["OBJECTID", "FIRE_LABEL", "FIRE_NUMBER", "FIRE_YEAR", "FIRE_TYPE", "FIRE_CAUSE", "IGNITION_DATE",
          "FIRE_OUT_DATE", "LATITUDE", "LONGITUDE", "FIRE_CENTRE", "CURRENT_SIZE", "RESPONSE_TYPE_DESC"]
WHERE = "FIRE_YEAR BETWEEN 2018 AND 2023"
PAGE = 1000


def query(**params):
    params = {"where": WHERE, "f": "json", **params}
    with urllib.request.urlopen(f"{LAYER}?{urllib.parse.urlencode(params)}", timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    assert "error" not in payload, payload
    return payload


def main():
    expected = query(returnCountOnly="true")["count"]
    records = []
    while len(records) < expected:
        page = query(outFields=",".join(FIELDS), returnGeometry="false", orderByFields="OBJECTID",
                     resultOffset=len(records), resultRecordCount=PAGE)["features"]
        assert page, f"empty page at offset {len(records)}"
        records += [feature["attributes"] for feature in page]
        print(f"  {len(records):,} / {expected:,}", flush=True)
        time.sleep(0.2)
    frame = pd.DataFrame(records, columns=FIELDS)
    for column in ("IGNITION_DATE", "FIRE_OUT_DATE"):
        frame[column] = pd.to_datetime(frame[column], unit="ms", utc=True)
    assert len(frame) == expected and frame.OBJECTID.is_unique
    assert frame.FIRE_YEAR.between(2018, 2023).all(), "2024 must never be loaded"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (OUT.with_suffix(".manifest.json")).write_text(json.dumps({
        "source": LAYER, "where": WHERE, "rows": len(frame), "sha256": digest,
        "downloaded_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "by_year": frame.FIRE_YEAR.value_counts().sort_index().to_dict(),
        "by_type": frame.FIRE_TYPE.value_counts().to_dict(),
        "by_cause": frame.FIRE_CAUSE.value_counts().to_dict()}, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(frame):,} incidents to {OUT}")


if __name__ == "__main__":
    main()
