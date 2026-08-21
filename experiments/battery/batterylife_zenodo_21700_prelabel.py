"""Label-blind prelabel probe for one manageable 21700 Zenodo subexperiment.

The 11-GB archive is never downloaded.  ZIP central-directory and member byte
ranges are read over HTTP Range requests.  Only the first 50 cycle-summary rows
are parsed; performance-summary/EOL files are never requested in this phase.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import struct
import urllib.request
import zlib
from pathlib import Path

import numpy as np

from batterylife_asymmetric_cohort_router import LOW_PROTOCOL_THRESHOLD, protocol_dispersion
from batterylife_curve_aware_support import robust_scale
from batterylife_transductive_pool_acquisition import distance_matrix


HERE = Path(__file__).parent
OUT = HERE / "batterylife_zenodo_21700_expt4_prelabel.json"
API = "https://zenodo.org/api/records/10637534/files/Expt%204%20-%20Drive%20Cycle%20Aging%20(Control).zip/content"
EXPT = "Expt 4 - Drive Cycle Aging (Control)"
MAX_H = 50


def get_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as stream:
        return stream.read()


def central_directory(url: str) -> tuple[int, dict[str, tuple[int, int, int, int]]]:
    head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
    size = int(head.headers["Content-Length"])
    tail_n = 4 * 1024 * 1024
    start = size - tail_n
    tail = get_range(url, start, size - 1)
    pos = tail.rfind(b"PK\x06\x06")
    if pos < 0:
        raise RuntimeError("ZIP64 EOCD not found")
    fields = struct.unpack("<4sQ2H2L4Q", tail[pos:pos + 56])
    count, cd_size, cd_offset = fields[7], fields[8], fields[9]
    central = tail[cd_offset - start:cd_offset - start + cd_size]
    entries = {}
    off = 0
    for _ in range(count):
        fixed = struct.unpack("<4s6H3L5H2L", central[off:off + 46])
        fn, extra_len, comment_len = fixed[10:13]
        name = central[off + 46:off + 46 + fn].decode("utf-8")
        extra = central[off + 46 + fn:off + 46 + fn + extra_len]
        comp, uncomp, local = fixed[9], fixed[8], fixed[16]
        i = 0
        while i + 4 <= len(extra):
            hid, length = struct.unpack("<HH", extra[i:i + 4])
            data = extra[i + 4:i + 4 + length]
            if hid == 1:
                j = 0
                if uncomp == 0xFFFFFFFF:
                    uncomp = struct.unpack("<Q", data[j:j + 8])[0]; j += 8
                if comp == 0xFFFFFFFF:
                    comp = struct.unpack("<Q", data[j:j + 8])[0]; j += 8
                if local == 0xFFFFFFFF:
                    local = struct.unpack("<Q", data[j:j + 8])[0]; j += 8
            i += 4 + length
        entries[name] = (local, comp, uncomp, fixed[4])
        off += 46 + fn + extra_len + comment_len
    return size, entries


def member_bytes(url: str, entry: tuple[int, int, int, int], limit: int | None = None) -> bytes:
    local, comp, _, method = entry
    header = get_range(url, local, local + 4096)
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = local + 30 + name_len + extra_len
    raw = get_range(url, start, start + comp - 1)
    if method == 8:
        dec = zlib.decompressobj(-15)
        output = dec.decompress(raw, limit or 0)
        if limit is None:
            output += dec.flush()
        return output
    return raw[:limit] if limit else raw


def first_rows(data: bytes, n_rows: int = MAX_H + 1) -> tuple[list[str], list[list[str]]]:
    lines = data.decode("utf-8", "replace").splitlines()
    header = lines[0].split(",")
    rows = [line.split(",") for line in lines[1:n_rows]]
    return header, rows


def protocol_from_metadata(cell: str, temperature: float) -> list[float]:
    # Expt 4 is drive-cycle aging with a common 0-100% SoC range.
    return [float(temperature), 0.0, 100.0, 4.0, 1.0]


def main() -> None:
    size, entries = central_directory(API)
    cycle_entries = {}
    for cell in "ABCDEFGH":
        suffix = f"Summary per Cycle/expt 4 - cell {cell} - cycle_data.csv"
        matches = [name for name in entries if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one cycle summary for {cell}, got {matches}")
        cycle_entries[cell] = matches[0]
    temperatures = {c: (10.0 if c in "ABC" else 25.0 if c in "DE" else 40.0) for c in "ABCDEFGH"}
    cells = []
    for cell in "ABCDEFGH":
        raw = member_bytes(API, entries[cycle_entries[cell]], limit=8 * 1024 * 1024)
        header, rows = first_rows(raw)
        if len(rows) < MAX_H:
            raise RuntimeError(f"{cell}: fewer than {MAX_H} cycle rows")
        lookup = {name: index for index, name in enumerate(header)}
        raw_capacity = []
        for row in rows[:MAX_H]:
            value = row[lookup["Discharge Capacity [A h]"]].strip()
            raw_capacity.append(float(value) if value else np.nan)
        capacities_array = np.asarray(raw_capacity, dtype=float)
        valid = np.isfinite(capacities_array)
        if valid.sum() < MAX_H:
            if valid.sum() < 2:
                raise RuntimeError(f"{cell}: too many missing early capacity rows")
            capacities_array = np.interp(
                np.arange(MAX_H), np.flatnonzero(valid), capacities_array[valid]
            )
        capacities = capacities_array.tolist()
        cells.append({
            "name": f"21700_expt4_cell_{cell}",
            "cell": cell,
            "temperature_C": temperatures[cell],
            "protocol": protocol_from_metadata(cell, temperatures[cell]),
            "curve_h50_capacity_Ah": capacities,
            "cycle_summary_member": cycle_entries[cell],
        })
    protocol = np.asarray([c["protocol"] for c in cells], float)
    curve = np.asarray([c["curve_h50_capacity_Ah"] for c in cells], float)
    dp = distance_matrix(protocol, robust_scale(protocol), 1e9)
    dc = distance_matrix(curve, robust_scale(curve), 1e9)
    spread = protocol_dispersion(dp)
    # Route eligibility is label blind.  Support identities are deliberately
    # not stored here because they are only needed after label opening.
    settings = []
    for horizon in (10, 20, 50):
        for budget in (1, 3, 5, 10):
            if spread <= 1e-12:
                route = "fallback_zero_protocol_dispersion"
            elif LOW_PROTOCOL_THRESHOLD <= spread < 0.60:
                route = "fallback_medium_protocol_dispersion"
            elif budget == 1:
                route = "fallback_one_label_unidentifiable"
            elif budget >= 5 and spread >= 0.60:
                route = "fallback_large_budget_high_protocol_dispersion"
            else:
                route = "active_w2" if spread < LOW_PROTOCOL_THRESHOLD else "active_w0.5"
            settings.append({"horizon": horizon, "label_budget_k": budget,
                             "protocol_dispersion": spread, "route": route,
                             "episodes": 100})
    output = {
        "phase": "PRELABEL_FROZEN_MANIFEST",
        "dataset": "Zenodo-21700-Expt4",
        "archive_url": API,
        "archive_size_bytes": size,
        "archive_member_count": len(entries),
        "information_barrier": "Only first 50 rows of cycle-summary members were parsed; performance-summary/EOL members were never requested.",
        "cells": cells,
        "protocol_dispersion": spread,
        "settings": settings,
        "active_settings": [s for s in settings if s["route"].startswith("active_")],
        "source_entry_hash": hashlib.sha256("\n".join(sorted(cycle_entries.values())).encode()).hexdigest(),
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"protocol_dispersion": spread,
                      "active_settings": len(output["active_settings"]),
                      "routes": {r: sum(s["route"] == r for s in settings) for r in sorted({s["route"] for s in settings})}}, indent=2))


if __name__ == "__main__":
    main()
