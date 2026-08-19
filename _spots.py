#!/usr/bin/env python3
"""Lihat tiga komponen sisa yang dilaporkan audit_clean pada jp_13 — apa itu
tinta Jepang yang tertinggal, garis balon, atau derau JPEG.

Tiap titik dipotong dari 01_input (sebelum) dan 09_cleaned (sesudah) pada kotak
yang sama, diperbesar 12x, lalu ditempel berdampingan supaya perbandingannya
tidak bergantung pada ingatan. Tidak menyentuh kode pipeline.
"""

from __future__ import annotations

import pathlib

from PIL import Image

DBG = pathlib.Path("debug/jp_13")
OUT = pathlib.Path("_dbg")
PAD = 14
SCALE = 12

SPOTS = [
    ("a_r0_62", 603, 14, 614, 28),
    ("b_r0_23", 562, 95, 574, 101),
    ("c_r1_14", 375, 12, 378, 19),
]


def main() -> int:
    inp = Image.open(DBG / "01_input.png").convert("RGB")
    cln = Image.open(DBG / "09_cleaned.png").convert("RGB")
    print(f"input={inp.size} cleaned={cln.size}")
    for tag, x1, y1, x2, y2 in SPOTS:
        box = (max(x1 - PAD, 0), max(y1 - PAD, 0),
               min(x2 + PAD, inp.size[0]), min(y2 + PAD, inp.size[1]))
        a = inp.crop(box)
        b = cln.crop(box)
        w, h = a.size
        big = Image.new("RGB", (w * SCALE * 2 + 8, h * SCALE), (255, 0, 0))
        big.paste(a.resize((w * SCALE, h * SCALE), Image.NEAREST), (0, 0))
        big.paste(b.resize((w * SCALE, h * SCALE), Image.NEAREST),
                  (w * SCALE + 8, 0))
        p = OUT / f"s13_{tag}.png"
        big.save(p)
        print(f"  {p}  box={box}  (kiri=INPUT, kanan=CLEANED)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
