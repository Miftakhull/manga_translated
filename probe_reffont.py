#!/usr/bin/env python3
"""Apakah font referensi CONTOH/6.JPG lebih RAPAT dari Anime Ace kita?

Metode presisi: ambil baris referensi yang teksnya bisa dibaca mata, ukur lebar
piksel baris itu, lalu bandingkan dengan lebar string yang SAMA di Anime Ace
pada ukuran yang tinggi kapitalnya cocok. Tidak menghitung komponen, tidak
menebak jumlah huruf. Offline, nol token.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import ImageFont
from pathlib import Path

ROOT = Path(__file__).resolve().parent
img = cv2.imread(str(ROOT / "CONTOH" / "6.JPG"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fp = next(p for p in sorted((ROOT / "fonts").rglob("*.ttf")) if "ace" in p.name.lower())

# Baris yang sudah diverifikasi mata dari zoom: (teks, x0,y0,x1,y1 di 6.JPG)
# Kotaknya sengaja sedikit longgar; lebar tinta diukur ulang di dalamnya.
LINES = [
    ("EMBARASSING",     437, 268, 535, 285),
    ("TOO EXCITED AND", 984, 258, 1108, 274),
    ("I'M PRAISING",    187, 310, 284, 328),
    ("NEVER SEEN",      664, 208, 761, 226),
    ("DESCRIBED",       460, 300, 519, 318),
    ("THEMSELVES!",     997, 288, 1095, 305),
]

cap_of = {s: (lambda b: b[3] - b[1])(ImageFont.truetype(str(fp), s).getbbox("HAMBURG"))
          for s in range(6, 46)}

print(f"font kita: {fp.name}")
print(f"\n{'teks':>17} {'ink_w':>6} {'capH':>5} {'ace_size':>8} {'ace_w':>6} "
      f"{'rasio':>6}")
ratios = []
for txt, x0, y0, x1, y1 in LINES:
    roi = gray[y0:y1, x0:x1]
    dark = roi < 110
    cols = np.where(dark.any(axis=0))[0]
    rows = np.where(dark.any(axis=1))[0]
    if cols.size == 0:
        print(f"{txt:>17}  -- tidak ada tinta di kotak itu, kotak salah")
        continue
    ink_w = int(cols[-1] - cols[0] + 1)
    ink_h = int(rows[-1] - rows[0] + 1)          # tinggi kapital baris ini
    size = min(cap_of, key=lambda s: abs(cap_of[s] - ink_h))
    f = ImageFont.truetype(str(fp), size)
    ace_w = f.getlength(txt)
    ratios.append(ink_w / ace_w)
    print(f"{txt:>17} {ink_w:>6} {ink_h:>5} {size:>8} {ace_w:>6.1f} "
          f"{ink_w/ace_w:>6.3f}")

if ratios:
    med = float(np.median(ratios))
    print(f"\nrasio lebar referensi / Anime Ace pada tinggi kapital sama: "
          f"median={med:.3f}  min={min(ratios):.3f} max={max(ratios):.3f}")
    print(f"artinya: pada tinggi huruf yang sama, referensi memuat "
          f"~{1/med:.2f}x lebih banyak karakter per baris.")

# Konsekuensi praktis di halaman kita (698 px): kolom lobus kiri-bawah balon
# ganda. Referensi memuat 'IT'S ONLY' di 66 px ref = 34 px skala kita.
sc = 698 / img.shape[1]
print(f"\nskala ref->kita = {sc:.4f}")
for txt in ("WONDER", "IT'S ONLY", "NO WONDER"):
    print(f"\n  {txt!r}:")
    for s in (6, 7, 8, 9):
        f = ImageFont.truetype(str(fp), s)
        w_ace = f.getlength(txt)
        print(f"    Anime Ace size {s}: {w_ace:>5.1f} px"
              + (f"   (kalau dirapatkan x{np.median(ratios):.2f} -> "
                 f"{w_ace*np.median(ratios):>5.1f} px)" if ratios else ""))
