#!/usr/bin/env python3
"""Cermin `_nbsrc/*.py` ke sel `%%writefile` di notebook. Satu arah: _nbsrc menang.

Repo ini punya sumber ganda. Tiap file di `_nbsrc/` adalah salinan satu sel
notebook dan diawali `%%writefile /content/mangatl/<nama>.py`, tapi yang
benar-benar dijalankan Colab adalah SELNYA. Menyunting satu sisi saja membuat
keduanya menyimpang tanpa suara — dan yang menyimpang justru sisi yang jalan.
Skrip ini menutup celah itu supaya tidak bergantung pada ketelitian manual.

Pakai:
    python sync_nbsrc.py --check    # hanya laporkan penyimpangan, exit 1 kalau ada
    python sync_nbsrc.py            # tulis isi _nbsrc/*.py ke sel masing-masing

Sengaja tidak ada arah sebaliknya (sel -> file): dua arah butuh penengah kalau
keduanya berubah, dan itu jauh lebih mudah salah daripada aturan "edit _nbsrc,
lalu sync".
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NBSRC = ROOT / "_nbsrc"

# Baris pertama sel yang menandai "sel ini adalah file modul".
_MAGIC = re.compile(r"^%%writefile\s+\S*?/([A-Za-z_]\w*\.py)\s*$")


def _cell_text(cell: dict) -> str:
    """Sumber sel sebagai satu string. nbformat mengizinkan str maupun list[str]."""
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def _norm(text: str) -> str:
    """CRLF -> LF. Notebook menyimpan LF; file di Windows bisa CRLF.

    Tanpa normalisasi ini --check melaporkan SEMUA sel menyimpang hanya karena
    akhiran barisnya beda, dan laporannya jadi tidak berguna.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _target(cell: dict) -> str | None:
    """Nama file modul yang ditulis sel ini, atau None kalau bukan sel modul."""
    if cell.get("cell_type") != "code":
        return None
    text = _cell_text(cell)
    first = _norm(text).split("\n", 1)[0]
    m = _MAGIC.match(first.strip())
    return m.group(1) if m else None


def _diff_head(name: str, want: str, got: str, lines: int = 12) -> str:
    """Cuplikan diff pertama — supaya kegagalan --check langsung bisa ditindak."""
    d = difflib.unified_diff(
        got.splitlines(), want.splitlines(),
        fromfile=f"notebook:{name}", tofile=f"_nbsrc/{name}", lineterm="", n=1,
    )
    head = list(d)[:lines]
    return "\n".join("      " + ln for ln in head)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="jangan tulis apa pun; exit 1 kalau ada yang menyimpang")
    ap.add_argument("--notebook", type=Path, default=None,
                    help="path .ipynb (default: satu-satunya .ipynb di root repo)")
    args = ap.parse_args(argv)

    nbs = [args.notebook] if args.notebook else sorted(ROOT.glob("*.ipynb"))
    if len(nbs) != 1:
        print(f"[sync] butuh tepat satu notebook, dapat {len(nbs)}: "
              f"{[p.name for p in nbs]}", file=sys.stderr)
        return 2
    nb_path = nbs[0]
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    seen: set[str] = set()
    drift: list[str] = []
    same: list[str] = []
    missing: list[str] = []

    for cell in nb.get("cells", []):
        name = _target(cell)
        if name is None:
            continue
        seen.add(name)
        local = NBSRC / name
        if not local.exists():
            missing.append(name)
            continue
        want = _norm(local.read_text(encoding="utf-8"))
        got = _norm(_cell_text(cell))
        if want == got:
            same.append(name)
            continue
        drift.append(name)
        if not args.check:
            # keepends: nbformat menyimpan source sebagai daftar baris ber-'\n'.
            cell["source"] = want.splitlines(keepends=True)
        else:
            print(f"  [DRIFT] {name}")
            print(_diff_head(name, want, got))

    orphan = sorted(p.name for p in NBSRC.glob("*.py") if p.name not in seen)

    for name in same:
        print(f"  [ OK  ] {name}")
    for name in missing:
        print(f"  [ ??  ] {name} — sel ada di notebook, file _nbsrc tidak ada")
    for name in orphan:
        print(f"  [ ??  ] {name} — file _nbsrc tidak punya sel di notebook")

    if args.check:
        if drift:
            print(f"\n[sync] {len(drift)} sel menyimpang: {', '.join(drift)}")
            print("[sync] jalankan `python sync_nbsrc.py` untuk mencerminkannya")
            return 1
        print(f"\n[sync] {len(same)} sel modul sinkron dengan _nbsrc/")
        return 0

    if not drift:
        print(f"\n[sync] tidak ada yang perlu ditulis ({len(same)} sel sinkron)")
        return 0

    # indent=1 + ensure_ascii=False = format tulisan nbformat/Jupyter sendiri,
    # jadi sel yang TIDAK disentuh tidak ikut berubah bentuknya.
    nb_path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n[sync] {len(drift)} sel dicerminkan: {', '.join(sorted(drift))}")
    print(f"[sync] ditulis ke {nb_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
