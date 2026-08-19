#!/usr/bin/env python3
"""Konteks lebar di sekitar lingkaran merah hasilnew3/6.JPG — supaya kelihatan
coretannya ada di balon yang mana dan bentuknya apa.

Merahnya juga dihapus (diganti tetangga terdekat non-merah) di satu versi,
supaya yang dinilai adalah gambarnya, bukan garis penanda.
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

OUT = pathlib.Path("_dbg")
SRC = "hasilnew3/6.JPG"
CX, CY = 273, 115          # pusat lingkaran (dari _mark.py)


def main() -> int:
    bgr = cv2.imread(SRC)
    h, w = bgr.shape[:2]
    b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
    red = ((r > 110) & (r - g > 50) & (r - b > 50)).astype(np.uint8)
    # Penanda merah dihapus dengan inpaint supaya tidak mengganggu penilaian.
    nored = cv2.inpaint(bgr, cv2.dilate(red, np.ones((3, 3), np.uint8)), 3,
                        cv2.INPAINT_TELEA)

    for tag, img in (("mark", bgr), ("nored", nored)):
        for half, scale in ((70, 6), (130, 4)):
            x0, y0 = max(CX - half, 0), max(CY - half, 0)
            x1, y1 = min(CX + half, w), min(CY + half, h)
            crop = img[y0:y1, x0:x1]
            big = cv2.resize(crop, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_LANCZOS4)
            p = OUT / f"ctx6_{tag}_{half}.png"
            cv2.imwrite(str(p), big)
            print(f"  {p}  asal=[{x0},{y0},{x1},{y1}]  ->  "
                  f"{big.shape[1]}x{big.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
