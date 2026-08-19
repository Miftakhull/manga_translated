#!/usr/bin/env python3
"""Inventaris output/ + debug/ dengan mtime, dan ringkasan tiap report.json.

Tujuan: menemukan run mana yang menghasilkan paruh ATAS _cmp_jp_*.png (yang
BERSIH dan diterjemahkan semua), lalu membaca angka-angkanya.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import pathlib

KEYS = ("region_count", "bubble_count", "residue_count", "residue_idx",
        "overflow_count", "protected_count", "sfx_idx", "translated_count",
        "untranslated_count", "untranslated_idx", "route_flat", "route_lama")


def mt(p: str) -> str:
    return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%m-%d %H:%M:%S")


def main() -> int:
    print("########## ISI output/ ##########")
    for p in sorted(glob.glob("output/*"), key=os.path.getmtime):
        print(f"  {mt(p)}  {os.path.getsize(p):>9}  {p}")

    print("\n########## ISI debug/ (folder) ##########")
    for d in sorted(glob.glob("debug/*"), key=os.path.getmtime):
        print(f"  {mt(d)}  {d}")
        if os.path.isdir(d):
            for f in sorted(glob.glob(d + "/*"), key=os.path.getmtime)[:12]:
                print(f"      {mt(f)}  {os.path.getsize(f):>9}  {os.path.basename(f)}")

    print("\n########## RINGKASAN report/json ##########")
    for p in sorted(glob.glob("output/*.json") + glob.glob("debug/*/report.json")
                    + glob.glob("kerja6/*/report.json"), key=os.path.getmtime):
        try:
            d = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  {p}: TIDAK TERBACA {exc}")
            continue
        print(f"\n  === {mt(p)}  {p}")
        for k in KEYS:
            if k in d:
                print(f"       {k:<20} {str(d[k])[:120]}")
        regs = d.get("regions") or []
        tr = [r for r in regs if r.get("translation")]
        print(f"       region berisi translation: {len(tr)}/{len(regs)}")
        for r in regs[:12]:
            t = r.get("translation")
            print(f"         idx={r.get('idx'):<3} prot={str(r.get('protected'))[:5]:<5} "
                  f"src={(r.get('src_text') or '')[:16]:<16} -> {str(t)[:34]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
