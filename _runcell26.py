#!/usr/bin/env python3
"""Jalankan ISI SEL 26 dari 2.ipynb apa adanya, di luar Colab.

Yang diuji bukan salinan kodenya tapi teks sel itu sendiri: diambil dari JSON
notebook, AUDIT_PAGE diisi, lalu exec. Kalau ini lolos, sel itu berjalan di
Colab juga — modul yang diimpornya sama (config/imgio/pipeline/typeset) dan
tidak ada API Colab yang dipakai.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
STAGE = ROOT / ".stage"
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))
_MAGIC = re.compile(r"^%%writefile\b.*\n?")


def _stage() -> None:
    STAGE.mkdir(exist_ok=True)
    for src in sorted((ROOT / "_nbsrc").glob("*.py")):
        body = _MAGIC.sub("", src.read_text(encoding="utf-8"), count=1)
        dest = STAGE / src.name
        if not dest.exists() or dest.read_text(encoding="utf-8") != body:
            dest.write_text(body, encoding="utf-8")


def main() -> int:
    page = sys.argv[1] if len(sys.argv) > 1 else "hasilnew/jp_13.JPG"
    _stage()
    sys.path.insert(0, str(STAGE))

    nb = json.loads((ROOT / "2.ipynb").read_text(encoding="utf-8"))
    cell = None
    for c in nb["cells"]:
        s = "".join(c["source"])
        if s.startswith("# Sel 26 (opsional) — AUDIT KEBERSIHAN BALON"):
            cell = s
            break
    if cell is None:
        print("sel 26 tidak ditemukan")
        return 2

    # Satu-satunya perubahan: mengisi AUDIT_PAGE, persis seperti user di Colab.
    cell = cell.replace('AUDIT_PAGE = ""', f'AUDIT_PAGE = r"{page}"', 1)
    print(f"[exec] sel 26 apa adanya, AUDIT_PAGE={page}\n")
    exec(compile(cell, "2.ipynb#sel26", "exec"), {"__name__": "__main__"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
