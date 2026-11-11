"""Fetch each benchmark dataset from its original source, and check it.

    python download.py --data ./datasets                 # all that can be fetched directly
    python download.py --data ./datasets --only cwru,gearbox
    python download.py --data ./datasets --check        # only count what is unpacked

Nothing is redistributed here: every file comes from the dataset's own
publisher. Each download is checked against the SHA-256 of the copy our
published results were computed from; a mismatch is reported, not hidden.
The archives that need an external unpacker (IMS: 7-Zip and RAR; XJTU-SY:
RAR) are unpacked by the commands this prints. XJTU-SY is published only
through file-sharing services, so it is fetched by hand (see README.md).

Licences and citations are in README.md. Needs only the Python standard
library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

IMS_URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
IMS_7Z_SHA256 = "c6925060e702708afc47261d349339d7cd97a98166161da6b29c377eb3e66f2c"

CWRU_URL = "https://engineering.case.edu/sites/default/files/{n}.mat"
CWRU_SHA256 = {
    97: "16bf48babcf1c7ac224bc1a81cd9eafdb27e42d5cf559761907e067e8eeadf3c",
    98: "37e6612c05e65c415dcfa2ab27a3fda648a5863160fa898b884a14743044e045",
    99: "4b97e6b5361f45efb6951dc3b1aebcdb3b89cb69d0f96d6f5c297dd9f45eee75",
    100: "88a5990cb541320e91505a1d72139e1993500ffe6e292a451011667f4138ca78",
    105: "f80b0ea04fd06b372a0eaec7c056543ea37e4bb4727a5b173d2a5bacd2aa9cab",
    118: "b00628f8dd8d1d930af77fa465d1e5cdb385fe259489053f91f3680bda7f640e",
    130: "35a095307d0971477049b343a1b5981dde465a58fb7f233ad89b035068c1717d",
    209: "9f723d6d9d2eba714c6dc50f99321dffa73d8b1e4d3605675b2c2251511eff80",
    222: "fcc0ee740fa123b1f4683f555458891719efb2857afd9fdee63e8fde0ed14469",
    234: "ebcbdb96612e8085bbdbf81f0e5e8e8a8d1a693d1ec559c4894f7792771aa9b2",
}

GEARBOX_URL = "https://data.openei.org/files/623/gearboxdata.zip"
GEARBOX_SHA256 = "df2a632de25dffcc182bc3bf450a91f408b6d26db36cea025190ae5911416226"

XJTU_PAGE = "https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets"
XJTU_PARTS_SHA256 = {
    "XJTU-SY_Bearing_Datasets.part01.rar": "c500657353f089a4ab50212ff4ddfc7b982729e92e2b127440c8ad881d80b968",
    "XJTU-SY_Bearing_Datasets.part02.rar": "4dfa6286a8e9c7cec1e925b347642349f36e27ab3f703c90d1d08977b7b4f61f",
    "XJTU-SY_Bearing_Datasets.part03.rar": "6929f531284f79e0209246b4ee23b23dd8d4faac1c0c3ff8fb131cda4a18bd3a",
    "XJTU-SY_Bearing_Datasets.part04.rar": "2245825acd29bed3b95e7a77879a3fbadf20f4ad4f99486384c714174621597d",
    "XJTU-SY_Bearing_Datasets.part05.rar": "e1b0a41a32b865e48ea7981c56f952a960d94335f3496b72eb4a65932f1671bd",
    "XJTU-SY_Bearing_Datasets.part06.rar": "df1854821a9d481104476379f7bc045ea1e101427c6e36145cd7677dc7c4a684",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"  have {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=600) as r, part.open("wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    part.replace(dest)
    return dest


def check_hash(path: Path, want: str) -> bool:
    got = sha256(path)
    ok = got == want
    print(f"  {'ok  ' if ok else 'DIFFERS'} sha256 {path.name}" + ("" if ok else f": {got}, ours {want}"))
    return ok


def ims(data: Path) -> None:
    print("NASA IMS (NASA Prognostics Data Repository)")
    z = fetch(IMS_URL, data / "ims" / "4.Bearings.zip")
    with zipfile.ZipFile(z) as zf:
        inner = [n for n in zf.namelist() if n.endswith("IMS.7z")]
        if inner and not (data / "ims" / inner[0]).exists():
            zf.extract(inner[0], data / "ims")
    seven = next((data / "ims").rglob("IMS.7z"), None)
    if seven is not None:
        check_hash(seven, IMS_7Z_SHA256)
    print("  unpack (7-Zip for the .7z; a RAR reader for the three .rar inside, e.g. Windows' tar,\n"
          "  unrar, or p7zip-rar):\n"
          f"    cd {data / 'ims'} && 7z x \"{seven.relative_to(data / 'ims') if seven else 'IMS.7z'}\"\n"
          "    mkdir -p test1 test2 test3\n"
          "    tar -xf 1st_test.rar -C test1 && tar -xf 2nd_test.rar -C test2 && tar -xf 3rd_test.rar -C test3\n"
          "  (3rd_test.rar unpacks to 4th_test/txt: a naming quirk of the NASA repackaging.)")


def cwru(data: Path) -> None:
    print("CWRU Bearing Data Center (Case School of Engineering)")
    for n, want in CWRU_SHA256.items():
        check_hash(fetch(CWRU_URL.format(n=n), data / "cwru" / f"{n}.mat"), want)


def gearbox(data: Path) -> None:
    print("Gearbox Fault Diagnosis Data (OpenEI submission 623)")
    z = fetch(GEARBOX_URL, data / "openei_gearbox" / "gearboxdata.zip")
    check_hash(z, GEARBOX_SHA256)
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            if name.endswith(".txt") and ("Healthy Data/" in name or "BrokenTooth Data/" in name):
                rel = Path(*Path(name).parts[-2:])          # "<folder>/<file>.txt", nothing else
                dest = data / "openei_gearbox" / rel
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(name))


def xjtu(data: Path) -> None:
    print("XJTU-SY (Xi'an Jiaotong University and Changxing Sumyoung Technology)")
    folder = data / "xjtu-sy"
    have = [p for p in (folder / n for n in XJTU_PARTS_SHA256) if p.exists()]
    if not have:
        print(f"  fetch the six-part archive by hand from one of the links on {XJTU_PAGE}\n"
              f"  into {folder}, then run this again to check it")
        return
    for p in have:
        check_hash(p, XJTU_PARTS_SHA256[p.name])
    print(f"  unpack: cd {folder} && unrar x XJTU-SY_Bearing_Datasets.part01.rar\n"
          "  RARLAB unrar reads the six parts as one set. Do not unpack each part on its\n"
          "  own (Windows tar does): the five files that span two parts come out damaged.")


def check(data: Path) -> int:
    """Count the unpacked snapshot files against what each run uploads."""
    sys.path.insert(0, str(HERE))
    import verify

    expected = json.loads((HERE / "expected.json").read_text())
    bad = 0
    for r in expected["runs"]:
        name = f"{r['dataset']}/{r['run']}"
        want = r.get("snapshots")
        if r["dataset"] == "ims":
            folder = data / verify.IMS_TESTS[verify.IMS_RUNS[r["run"]][0]][0]
            n = sum(1 for p in folder.iterdir()) if folder.is_dir() else 0
        elif r["dataset"] == "xjtu-sy":
            b = r["run"].removeprefix("bearing")
            folder = data / verify.XJTU_ROOT / verify.XJTU_CONDITION[b[0]] / f"Bearing{b}"
            files = sorted(folder.glob("*.csv")) if folder.is_dir() else []
            n = len(files)
            damaged = [p.name for p in files if b"\x00" in p.read_bytes()]
            if damaged:
                bad += 1
                print(f"DAMAGED {name}: {', '.join(damaged)} (unpack again with unrar)")
        else:
            continue
        ok = n == want
        bad += not ok
        print(f"{'ok  ' if ok else 'DIFFERS'} {name}: {n} snapshot files, the published run used {want}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=HERE / "datasets")
    ap.add_argument("--only", default="ims,xjtu-sy,cwru,gearbox")
    ap.add_argument("--check", action="store_true", help="count unpacked files only")
    args = ap.parse_args(argv)
    if args.check:
        return check(args.data)
    for name in args.only.split(","):
        {"ims": ims, "xjtu-sy": xjtu, "cwru": cwru, "gearbox": gearbox}[name](args.data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
