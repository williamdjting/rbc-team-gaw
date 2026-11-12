"""Check frozen artifacts without importing ML libraries or reading holdout data."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    protocol = json.loads((here / "frozen_protocol.json").read_text(encoding="utf-8"))
    checks = dict(protocol["source_sha256"])
    checks[(here / protocol["checkpoint"]).relative_to(root).as_posix()] = protocol["checkpoint_sha256"]
    failures = []
    for name, expected in checks.items():
        path = root / name
        if not path.is_file():
            failures.append(f"Missing: {name}")
        elif sha256(path) != expected:
            failures.append(f"Changed since freeze: {name}")
    expected = {(r, m, d) for r in protocol["regions"]
                for m in ("mlp", "rule_pair_yesterday")
                for d in ("res7_exact", "res7_margin1")}
    observed = {(r["region"], r["model"], r["definition"]) for r in protocol["thresholds"]}
    if observed != expected or len(protocol["thresholds"]) != len(expected):
        failures.append("Incomplete or duplicated threshold table")
    if list(protocol["scaler"]) != protocol["inputs"]:
        failures.append("Scaler order differs from model feature order")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(checks)} artifact/source hashes and {len(expected)} thresholds.")
    print("Freeze verified only. No 2024 data loaded or performance computed.")


if __name__ == "__main__":
    main()
