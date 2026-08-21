"""Run the final local reproducibility audit without requiring Docker.

This intentionally verifies artifacts and source integrity; it does not claim
that a container image was built when the host has no container daemon.
"""
from __future__ import annotations

import hashlib
import json
import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BAT = ROOT / "experiments" / "battery"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return {"command": cmd, "returncode": p.returncode, "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:]}


def main() -> int:
    checks = []
    for script in ("batterylife_verify_apace_v2_artifacts.py", "verify_experiment_execution_artifacts.py"):
        result = run([sys.executable, str(BAT / script)])
        checks.append(result)
    compile_result = run([sys.executable, "-m", "compileall", "-q", str(BAT)])
    checks.append(compile_result)

    required = [
        ROOT / "METHOD_FREEZE_V2.md",
        ROOT / "requirements-battery-experiments.lock",
        ROOT / "Dockerfile.battery-experiments",
        ROOT / "MANUSCRIPT_DRAFT_CN.md",
        BAT / "paper_figures" / "manifest.json",
    ]
    files = [{"path": str(p.relative_to(ROOT)), "exists": p.exists(), "sha256": sha256(p) if p.exists() else None} for p in required]
    lock = ROOT / "requirements-battery-experiments.lock"
    dependency_rows = []
    for line in lock.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^#\s]+)", line.strip())
        if not m:
            continue
        name, expected = m.groups()
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        dependency_rows.append({"package": name, "expected": expected, "installed": actual, "match": actual == expected})
    passed = all(x["returncode"] == 0 for x in checks) and all(x["exists"] for x in files)
    report = {
        "passed": passed,
        "container_runtime_present": any(__import__("shutil").which(x) for x in ("docker", "podman", "apptainer")),
        "note": "Container build/run is not asserted when no runtime is present; lockfile and Dockerfile are hashed below.",
        "checks": checks,
        "required_files": files,
        "lockfile_matches_current_environment": all(x["match"] for x in dependency_rows),
        "dependencies": dependency_rows,
    }
    out = BAT / "final_reproducibility_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": passed, "report": str(out.relative_to(ROOT))}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
