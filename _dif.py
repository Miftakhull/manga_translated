#!/usr/bin/env python3
"""Diff penuh modul kerja6/2.ipynb (run BERSIH 16 Agu 12:43) vs _nbsrc/ sekarang.

Argumen = daftar nama modul. Keluaran unified diff mentah, supaya hunk-nya bisa
dibaca, bukan cuma jumlah barisnya.
"""

from __future__ import annotations

import difflib
import json
import pathlib
import sys


def main(argv: list[str]) -> int:
    nb = json.load(open("kerja6/2.ipynb", encoding="utf-8"))
    good: dict[str, str] = {}
    for c in nb["cells"]:
        s = "".join(c["source"])
        first = s.split("\n", 1)[0]
        if "%%writefile" in first:
            good[first.split("/")[-1].strip()] = s.split("\n", 1)[1]
    for name in argv or sorted(good):
        if name not in good:
            print(f"### {name}: TIDAK ADA di kerja6/2.ipynb")
            continue
        a = good[name].splitlines()
        b = (pathlib.Path("_nbsrc") / name).read_text(encoding="utf-8").splitlines()
        print(f"\n{'=' * 72}\n=== {name}   kerja6(BERSIH) -> _nbsrc(SEKARANG)\n{'=' * 72}")
        n = 0
        for ln in difflib.unified_diff(a, b, "kerja6", "nbsrc", n=3, lineterm=""):
            print(ln)
            n += 1
        print(f"--- total {n} baris diff ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
