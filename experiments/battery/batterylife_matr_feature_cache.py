"""Build a compact, auditable post-blind MATR early-feature cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from batterylife_blind_prelabel_manifest import (
    load_unlabeled_archive,
    top_level_json_keys,
)


HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT / "data" / "batterylife_zenodo" / "MATR.zip"
LABELS = ROOT / "data" / "batterylife_processed" / "Life labels" / "MATR_labels.json"
OUTPUT = ROOT / "data" / "derived" / "matr_early_features_v2.npz"
EXPECTED_MD5 = "83a1528858b9e1b7b6886757bb561669"


def digest(path, algorithm):
    value = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    actual = digest(ARCHIVE, "md5")
    if actual != EXPECTED_MD5:
        raise RuntimeError(f"MATR archive MD5 mismatch: {actual}")
    horizons = (10, 20, 50)
    allowed = top_level_json_keys(LABELS)
    loaded = load_unlabeled_archive(ARCHIVE, horizons, allowed)
    labels = json.load(open(LABELS))
    names = np.asarray([cell["name"] for cell in loaded[10]])
    if any(np.any(names != np.asarray([cell["name"] for cell in loaded[h]])) for h in horizons):
        raise RuntimeError("MATR horizon identity/order mismatch")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "names": names,
        "life": np.asarray([float(labels[name]) for name in names]),
        "archive_md5": np.asarray(actual),
        "label_sha256": np.asarray(digest(LABELS, "sha256")),
    }
    for horizon in horizons:
        arrays[f"protocol_h{horizon}"] = np.asarray(
            [cell["protocol"] for cell in loaded[horizon]], dtype=float
        )
        arrays[f"curve_h{horizon}"] = np.asarray(
            [cell["curve"] for cell in loaded[horizon]], dtype=float
        )
    np.savez_compressed(OUTPUT, **arrays)
    print(json.dumps({
        "output": str(OUTPUT),
        "size": OUTPUT.stat().st_size,
        "sha256": digest(OUTPUT, "sha256"),
        "n_cells": len(names),
        "horizons": list(horizons),
        "archive_md5": actual,
    }, indent=2))


if __name__ == "__main__":
    main()
