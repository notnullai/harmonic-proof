"""Re-run Pontex Machine Health's published benchmark on your own copy of the data.

For each benchmark run this reads the original public dataset files, keeps
only what the run uses, uploads it to the verify service, and prints the
service's answer next to the published one (expected.json), pass or fail.

    python verify.py --server https://<verify service> --key <your key> --data ./datasets
    python verify.py ... --only ims/test2-bearing1,cwru/all
    python verify.py --convert-only out/ --data ./datasets      # write the uploads, send nothing

Needs Python 3.10+, numpy, and scipy (CWRU's .mat files only). The key is
also read from HARMONIC_VERIFY_KEY.

What is converted, and how (this file is the whole conversion; nothing else
touches your data before it is sent):

  * IMS: each snapshot file's whitespace-separated values, read as float32,
    one column (the run's bearing channel), its first 20480 rows; plus the
    file names (their time stamps). The service needs the whole snapshot.
  * XJTU-SY: each N.csv's two columns after the header, read as float32, the
    first 32768 rows; plus the file numbers. A file with null bytes (a
    damaged unpack) stops the run.
  * CWRU: each record's drive-end signal (X<nnn>_DE_time) as stored (float64),
    and the record's RPM value where the file carries one.
  * Real gearbox (OpenEI): column 0 of each table, parsed as float64.

The arrays go in one .npz (numpy's zip of .npy files, deflate-compressed),
uploaded in 16 MiB parts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# --- where each dataset's files are, under --data (as download.py leaves them) ---------

IMS_TESTS = {1: ("ims/test1/1st_test", 8), 2: ("ims/test2/2nd_test", 4), 3: ("ims/test3/4th_test/txt", 4)}
IMS_RUNS = {"test2-bearing1": (2, 0), "test1-bearing3": (1, 4),
            "test1-bearing4": (1, 6), "test3-bearing3": (3, 2)}
IMS_SAMPLES = 20480

XJTU_ROOT = "xjtu-sy/XJTU-SY_Bearing_Datasets"
XJTU_CONDITION = {"1": "35Hz12kN", "2": "37.5Hz11kN", "3": "40Hz10kN"}
XJTU_SAMPLES = 32768
CWRU_RECORDS = (97, 98, 99, 100, 105, 118, 130, 209, 222, 234)
GEARBOX_LOADS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)


# --- conversion ------------------------------------------------------------------------


def convert_ims(data: Path, run: str) -> dict[str, np.ndarray]:
    test, channel = IMS_RUNS[run]
    folder, n_channels = IMS_TESTS[test]
    files = []
    for p in (data / folder).iterdir():
        try:
            files.append((datetime.strptime(p.name, "%Y.%m.%d.%H.%M.%S"), p))
        except ValueError:
            continue
    files.sort()
    samples = np.empty((len(files), IMS_SAMPLES), dtype="<f4")
    for i, (_, p) in enumerate(files):
        values = np.array(p.read_text().split(), dtype=np.float32).reshape(-1, n_channels)
        samples[i] = values[:IMS_SAMPLES, channel]
    stamps = np.array([p.name for _, p in files], dtype="<U19")
    return {"stamps": stamps, "samples": samples}


def convert_xjtu(data: Path, run: str) -> dict[str, np.ndarray]:
    bearing = run.removeprefix("bearing")
    folder = data / XJTU_ROOT / XJTU_CONDITION[bearing[0]] / f"Bearing{bearing}"
    minutes = sorted(int(p.stem) for p in folder.glob("*.csv") if p.stem.isdigit())
    ch = np.empty((2, len(minutes), XJTU_SAMPLES), dtype="<f4")
    for i, m in enumerate(minutes):
        text = (folder / f"{m}.csv").read_text()
        if "\x00" in text:
            sys.exit(f"{folder / f'{m}.csv'} is damaged (null bytes). Unpack the six parts as one "
                     "set with RARLAB unrar; unpacking each part on its own damages the five "
                     "files that span two parts.")
        body = text.split("\n", 1)[1]
        values = np.array(body.replace(",", " ").split(), dtype=np.float32).reshape(-1, 2)
        ch[0, i], ch[1, i] = values[:XJTU_SAMPLES, 0], values[:XJTU_SAMPLES, 1]
    return {"minutes": np.array(minutes, dtype="<i4"), "ch0": ch[0], "ch1": ch[1]}


def convert_cwru(data: Path, run: str) -> dict[str, np.ndarray]:
    from scipy.io import loadmat

    out = {}
    for n in CWRU_RECORDS:
        d = loadmat(data / "cwru" / f"{n}.mat")
        out[f"r{n}"] = np.asarray(d[f"X{n:03d}_DE_time"], dtype="<f8").ravel()
        if f"X{n:03d}RPM" in d:
            out[f"r{n}_rpm"] = np.asarray(d[f"X{n:03d}RPM"], dtype="<f8").ravel()[:1]
    return out


def convert_gearbox(data: Path, run: str) -> dict[str, np.ndarray]:
    out = {}
    for kind, folder, prefix in (("healthy", "Healthy Data", "h"), ("broken", "BrokenTooth Data", "b")):
        for p in GEARBOX_LOADS:
            table = np.loadtxt(data / "openei_gearbox" / folder / f"{prefix}30hz{p}.txt")
            out[f"{kind}_{p}"] = np.ascontiguousarray(table[:, 0], dtype="<f8")
    return out


CONVERT = {"ims": convert_ims, "xjtu-sy": convert_xjtu, "cwru": convert_cwru, "gearbox": convert_gearbox}


def convert(dataset: str, run: str, data: Path, out: Path) -> Path:
    """Write one run's upload to `out` (a .npz) and return it."""
    arrays = CONVERT[dataset](data, run)
    np.savez_compressed(out, **arrays)
    return out


# --- the service ---------------------------------------------------------------------


class Service:
    def __init__(self, server: str, key: str):
        self.server, self.key = server.rstrip("/"), key

    def call(self, method: str, path: str, body: bytes | None = None,
             content_type: str = "application/json") -> dict:
        req = urllib.request.Request(self.server + path, data=body, method=method,
                                     headers={"X-Verify-Key": self.key, "Content-Type": content_type})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                if e.code == 503 and attempt < 5:
                    time.sleep(int(e.headers.get("Retry-After", "60")))
                    continue
                raise SystemExit(f"{method} {path}: HTTP {e.code}: {detail}") from None
        raise SystemExit(f"{method} {path}: the service stayed busy")

    def run(self, dataset: str, run: str, upload: Path, part_bytes: int) -> dict:
        job = self.call("POST", "/verify/jobs", json.dumps({"dataset": dataset, "run": run}).encode())
        jid = job["id"]
        with upload.open("rb") as fh:
            offset = 0
            while part := fh.read(part_bytes):
                self.call("PUT", f"/verify/jobs/{jid}/upload?offset={offset}", part,
                          "application/octet-stream")
                offset += len(part)
        self.call("POST", f"/verify/jobs/{jid}/start", b"{}")
        while True:
            job = self.call("GET", f"/verify/jobs/{jid}")
            if job["status"] in ("done", "failed"):
                return job
            time.sleep(15)


# --- comparing -----------------------------------------------------------------------


def _diff(ours, theirs, path: str = "") -> list[str]:
    if isinstance(ours, dict) and isinstance(theirs, dict):
        out = []
        for k in sorted(set(ours) | set(theirs)):
            out += _diff(ours.get(k), theirs.get(k), f"{path}.{k}" if path else k)
        return out
    return [] if ours == theirs else [f"{path}: got {ours!r}, published {theirs!r}"]


def summary(row: dict) -> str:
    d = row.get("detection")
    det = "not detected" if not d else f"detected at {d['hours']} h ({d['lead_hours']} h lead)"
    c = row["commission"]
    named = "" if not row.get("diagnosis") else \
        f", named right/wrong {row['diagnosis']['right']}/{row['diagnosis']['wrong']}"
    return f"{det}; {c['false_alarms']} false alarms of {c['warm']} while learning{named}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", help="the verify service's address")
    ap.add_argument("--key", default=os.environ.get("HARMONIC_VERIFY_KEY"))
    ap.add_argument("--data", type=Path, default=HERE / "datasets", help="where download.py put the datasets")
    ap.add_argument("--expected", type=Path, default=HERE / "expected.json")
    ap.add_argument("--only", help="comma-separated dataset/run keys, e.g. ims/test2-bearing1,cwru/all")
    ap.add_argument("--convert-only", type=Path, metavar="DIR", help="write the uploads to DIR; send nothing")
    ap.add_argument("--save", type=Path, metavar="FILE", help="also write every answer, as JSON, to FILE")
    args = ap.parse_args(argv)

    expected = json.loads(args.expected.read_text())
    runs = expected["runs"]
    if args.only:
        wanted = set(args.only.split(","))
        runs = [r for r in runs if f"{r['dataset']}/{r['run']}" in wanted]
        if len(runs) != len(wanted):
            raise SystemExit(f"unknown runs: {sorted(wanted - {r['dataset'] + '/' + r['run'] for r in runs})}")
    if args.convert_only:
        args.convert_only.mkdir(parents=True, exist_ok=True)
        for r in runs:
            p = convert(r["dataset"], r["run"], args.data, args.convert_only / f"{r['dataset']}_{r['run']}.npz")
            print(f"{r['dataset']}/{r['run']}: {p} ({p.stat().st_size / 2**20:.1f} MiB)")
        return 0
    if not args.server or not args.key:
        raise SystemExit("--server and --key (or HARMONIC_VERIFY_KEY) are required")

    service = Service(args.server, args.key)
    part = service.call("GET", "/verify/catalogue")["upload"]["part_bytes"]
    print(f"published results: build {expected['build']}, generated {expected['generated']}\n")
    passed = failed = 0
    answers: list[dict] = []
    for r in runs:
        name = f"{r['dataset']}/{r['run']}"
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.time()
            up = convert(r["dataset"], r["run"], args.data, Path(tmp) / "upload.npz")
            up_bytes = up.stat().st_size
            print(f"{name}: converted in {time.time() - t0:.0f} s, uploading "
                  f"{up_bytes / 2**20:.1f} MiB", flush=True)
            job = service.run(r["dataset"], r["run"], up, part)
        answers.append(job | {"upload_bytes": up_bytes})
        if args.save:
            args.save.write_text(json.dumps({"server": args.server, "answers": answers}, indent=1) + "\n")
        if job["status"] != "done":
            print(f"  FAIL  the service could not run it: {job.get('error')}")
            failed += len(r["rows"])
            continue
        res = job["result"]
        if res["brain_hash"] != r["brain_hash"]:
            print(f"  note: the service's networks differ from the published run's "
                  f"(brain {res['brain_hash'][:12]} vs {r['brain_hash'][:12]}); build {res['build']}")
        got = {(x["dataset"], x["run"]): x for x in res["rows"]}
        for want in r["rows"]:
            ours = got.get((want["dataset"], want["run"]))
            diffs = ["missing from the answer"] if ours is None else _diff(ours, want)
            ok = not diffs
            passed += ok
            failed += not ok
            print(f"  {'pass' if ok else 'FAIL'}  {want['dataset']} {want['run']}: "
                  f"{summary(ours) if ours else '-'}")
            if not ok:
                print(f"        published: {summary(want)}")
                for d in diffs[:10]:
                    print(f"        {d}")
        print(f"  answered by build {res['build']}, brain {res['brain_hash'][:16]}, "
              f"in {job.get('seconds', 0):.0f} s", flush=True)
    print(f"\n{passed} of {passed + failed} published rows reproduced exactly")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
