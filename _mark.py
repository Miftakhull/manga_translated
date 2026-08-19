#!/usr/bin/env python3
"""Temukan LINGKARAN MERAH yang digambar user di hasilnew3/, lalu potong isinya
diperbesar besar. Merahnya dicari secara numerik supaya letaknya tidak ditebak.

Merah = R tinggi, G dan B rendah, dan JELAS lebih merah dari saluran lain
(halaman manga hitam-putih, jadi piksel berwarna hanya yang digambar user).
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

OUT = pathlib.Path("_dbg")


def main() -> int:
    for name in sys.argv[1:] or ["hasilnew3/6.JPG", "hasilnew3/13.JPG"]:
        p = pathlib.Path(name)
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"{p}: tidak terbaca")
            continue
        b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
        red = (r > 110) & (r - g > 50) & (r - b > 50)
        n = int(red.sum())
        print(f"\n{p}  {bgr.shape[1]}x{bgr.shape[0]}  piksel merah={n}")
        if n == 0:
            print("   tidak ada tanda merah")
            continue
        num, lab, stats, _ = cv2.connectedComponentsWithStats(
            red.astype(np.uint8), connectivity=8)
        order = sorted(range(1, num), key=lambda i: -stats[i][4])
        for k, i in enumerate(order[:4]):
            x, y, w, h, area = stats[i]
            if area < 30:
                continue
            print(f"   tanda#{k}: bbox=[{x},{y},{x + w},{y + h}] {w}x{h} area={area}")
            pad = 6
            x0, y0 = max(x - pad, 0), max(y - pad, 0)
            x1, y1 = min(x + w + pad, bgr.shape[1]), min(y + h + pad, bgr.shape[0])
            crop = bgr[y0:y1, x0:x1]
            for scale, tag in ((8, "z8"), (16, "z16")):
                big = cv2.resize(crop, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_NEAREST)
                f = OUT / f"mark_{p.stem}_{k}_{tag}.png"
                cv2.imwrite(str(f), big)
                print(f"      {f} {big.shape[1]}x{big.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
