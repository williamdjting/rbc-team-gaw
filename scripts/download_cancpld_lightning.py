"""Download official 3-hourly CanCPLD CG grids and preserve provenance."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://crd-data-donnees-rdc.ec.gc.ca/CDAS/products/CanCPLD/CLDN/cg/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2018, 2025)))
    args = parser.parse_args()
    out = ROOT / "data/cancpld_lightning/raw"
    out.mkdir(parents=True, exist_ok=True)

    def download(year):
        path = out / f"cg_flashes_3hr_0.1-deg_{year}.nc"
        url = BASE + path.name
        if not path.exists():
            temporary = path.with_suffix(".nc.part")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                expected = response.headers.get("Content-Length")
                while block := response.read(1 << 20):
                    stream.write(block)
            if expected and temporary.stat().st_size != int(expected):
                raise RuntimeError(f"Incomplete download: {year}")
            temporary.replace(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        print(f"Downloaded {year}: {path.stat().st_size:,} bytes", flush=True)
        return {"year": year, "url": url, "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size, "sha256": digest}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download, args.years))
    (out.parent / "download_manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
