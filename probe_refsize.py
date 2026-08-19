#!/usr/bin/env python3
"""Ukur tinggi huruf SUNGGUHAN di hasilnew/6.JPG dan hasilnew/13.JPG.

Kenapa perlu: anggaran balon kita memakai lantai min_font_size = 11 px, dan
pada halaman selebar 698 px anggaran itu memberi 2-39 karakter sementara wording
referensi 35-62 karakter. Salah satu dari dua hal benar: referensi memakai huruf
lebih kecil dari 11 px, atau referensi memang melewati batas. Ini mengukurnya,
bukan menebak — cap height dari komponen terhubung di dalam balon.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def measure(path: Path) -> None:
    img = cv2.imread(str(path))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Teks Inggris referensi = hitam pekat di atas balon putih. Cari komponen
    # kecil-kecil yang berkumpul: itu hurufnya.
    dark = (g < 100).astype(np.uint8)
    n, lab, st, cent = cv2.connectedComponentsWithStats(dark, 8)
    hs = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        # Huruf: tinggi 3..20 px, lebar 2..20 px, cukup padat, bukan garis panel.
        if 3 <= h <= 20 and 2 <= w <= 20 and a >= 4 and a >= 0.20 * w * h:
            hs.append(h)
    hs = np.array(sorted(hs))
    if hs.size == 0:
        print(f"{path.name}: tidak ada komponen huruf")
        return
    # Modus tinggi = cap height huruf besar (referensi ALL CAPS).
    vals, cnts = np.unique(hs, return_counts=True)
    print(f"{path.name} {img.shape[1]}x{img.shape[0]} komponen={hs.size}")
    print("   sebaran tinggi:", dict(zip(vals.tolist(), cnts.tolist())))
    print(f"   modus={int(vals[cnts.argmax()])} median={int(np.median(hs))} "
          f"p25={int(np.percentile(hs,25))} p75={int(np.percentile(hs,75))}")


for p in sys.argv[1:] or ["hasilnew/6.JPG", "hasilnew/13.JPG"]:
    measure(ROOT / p)
