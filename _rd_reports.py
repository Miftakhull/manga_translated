#!/usr/bin/env python3
"""Baca report.json hasil run 16 Agu (yang BERSIH) apa adanya.

Ditulis sebagai file, bukan heredoc inline: pipe inline di sesi ini berulang
kali mengembalikan string tak terkait ('clean - nothing to commit') alih-alih
keluaran skrip.
"""

from __future__ import annotations

import json
import pathlib

PATHS = (
    "debug/clean_jp_6/report.json",
    "debug/jp_13/report.json",
    "kerja6/debug_jp_6/report.json",
    "output/clean_jp_6.json",
    "output/jp_13.json",
)


def main() -> int:
    for p in PATHS:
        f = pathlib.Path(p)
        if not f.exists():
            print(f"=== {p}  (TIDAK ADA)")
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        print(f"=== {p}")
        for k, v in d.items():
            print(f"   {k:<22} {str(v)[:170]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
