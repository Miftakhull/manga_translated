#!/usr/bin/env python3
"""Apakah CLEAN sungguh regresi? Dijawab dengan hash, bukan dugaan.

09_cleaned.png dari tiga run yang ada di disk dibandingkan byte-per-byte:
  kerja6/debug_jp_6/  16 Agu 12:43  <- run BERSIH yang ditunjuk user
  debug/clean_jp_6/   16 Agu 12:34
  debug/jp_6/         17 Agu 06:05  <- kode SEKARANG
Kalau ketiganya identik, tahap clean di kode lokal TIDAK berubah, dan coretan di
hasilnew2 datang dari yang lain (Colab: GPU/LaMa, atau region yang tidak
terdeteksi) — bukan dari edit kode tanggal 16-17.
"""

from __future__ import annotations

import hashlib
import json
import pathlib


def h(p: str) -> str:
    b = pathlib.Path(p).read_bytes()
    return f"{hashlib.sha256(b).hexdigest()[:16]}  {len(b):>8} B"


def main() -> int:
    print("===== 09_cleaned.png (hasil tahap CLEAN) =====")
    for p in ("kerja6/debug_jp_6/09_cleaned.png",
              "debug/clean_jp_6/09_cleaned.png",
              "debug/jp_6/09_cleaned.png",
              "kerja6/debug_jp_6/07_mask_after_sfx_exclusion.png",
              "debug/jp_6/07_mask_after_sfx_exclusion.png",
              "kerja6/debug_jp_6/05_mask.png",
              "debug/jp_6/05_mask.png"):
        if pathlib.Path(p).exists():
            print(f"  {h(p)}  {p}")
        else:
            print(f"  (tidak ada) {p}")

    print("\n===== angka clean di tiap report =====")
    for p in ("kerja6/debug_jp_6/report.json", "debug/clean_jp_6/report.json",
              "debug/jp_6/report.json", "debug/jp_13/report.json"):
        if not pathlib.Path(p).exists():
            continue
        d = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        print(f"  {p}")
        print(f"     region={d.get('region_count')} bubble={d.get('bubble_count')} "
              f"residue={d.get('residue_count')} residue_idx={d.get('residue_idx')} "
              f"route_flat={d.get('route_flat')} route_lama={d.get('route_lama')}")
        # rute per region: flat vs lama menentukan mutu bersihnya
        for r in d.get("regions", []):
            print(f"       idx={r.get('idx')} label={r.get('label')} "
                  f"route={r.get('route')} prot={r.get('protected')} "
                  f"bbox={r.get('bbox')} src={(r.get('src_text') or '')[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
