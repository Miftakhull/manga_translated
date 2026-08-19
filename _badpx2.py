#!/usr/bin/env python3
"""Tiga komponen 'sisa' di jp_13 diperiksa dari mask yang SUDAH tersimpan —
tanpa memuat model, jadi jawabannya datang dalam hitungan detik.

debug/jp_13/05_mask.png = gabungan ink_mask mentah (yang dianggap teks)
debug/jp_13/07_mask_after_sfx_exclusion.png = yang benar-benar dihapus

Untuk tiap piksel yang ditandai kotor oleh audit (deviasi > 16 dari 255 di dalam
ink_mask yang didilatasi 2 px — persis rumus run_full.py):
  * apakah piksel itu ada di mask hapus (07) -> pipeline memang menyuruh hapus,
  * atau cuma kena karena DILATASI 2 px -> yang gelap itu tetangga di luar mask,
    yaitu garis balon / tinta panel, dan audit-nya yang terlalu lebar.
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

DBG = pathlib.Path("debug/jp_13")
COMPS = [
    ("r0 garis 62px", 603, 14, 614, 28),
    ("r0 garis 23px", 562, 95, 574, 101),
    ("r1 titik 14px", 375, 12, 378, 19),
]


def main() -> int:
    inp = cv2.imread(str(DBG / "01_input.png"))
    cln = cv2.imread(str(DBG / "09_cleaned.png"))
    raw = cv2.imread(str(DBG / "05_mask.png"), cv2.IMREAD_GRAYSCALE)
    era = cv2.imread(str(DBG / "07_mask_after_sfx_exclusion.png"),
                     cv2.IMREAD_GRAYSCALE)
    gi = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY).astype(np.int16)
    gc = cv2.cvtColor(cln, cv2.COLOR_BGR2GRAY).astype(np.int16)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grow = cv2.dilate(raw, el, iterations=1)
    print(f"ink mentah={int((raw > 0).sum())} px  "
          f"setelah dilatasi 2px={int((grow > 0).sum())} px  "
          f"mask hapus={int((era > 0).sum())} px")

    for tag, x1, y1, x2, y2 in COMPS:
        print(f"\n{tag}  bbox=[{x1},{y1},{x2},{y2}]")
        rows = []
        for y in range(y1, y2):
            for x in range(x1, x2):
                if grow[y, x] == 0:
                    continue
                if abs(int(gc[y, x]) - 255) <= 16:
                    continue
                rows.append((x, y, int(gi[y, x]), int(gc[y, x]),
                             int(raw[y, x] > 0), int(era[y, x] > 0)))
        n_raw = sum(r[4] for r in rows)
        n_era = sum(r[5] for r in rows)
        print(f"   ditandai kotor : {len(rows)} px")
        print(f"   di ink MENTAH  : {n_raw} px "
              f"({'ada' if n_raw else 'TIDAK ADA'} — 0 berarti murni efek dilatasi)")
        print(f"   di mask HAPUS  : {n_era} px")
        print("      x    y  input cleaned raw hapus")
        for r in rows[:14]:
            print(f"   {r[0]:>4} {r[1]:>4} {r[2]:>6} {r[3]:>7} {r[4]:>3} {r[5]:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
