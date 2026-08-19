#!/usr/bin/env python3
"""Kerapatan huruf referensi CONTOH/6.JPG vs Anime Ace kita, per baris.

Pertanyaan: referensi memuat baris seperti 'IT'S ONLY' di lobus yang sama
tanpa penggalan, di font yang LEBIH BESAR dari kita. Apakah itu karena
fontnya lebih rapat (advance per huruf lebih kecil pada tinggi kapital yang
sama), atau karena wording-nya lebih pendek? Offline, nol token.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import ImageFont
from pathlib import Path

ROOT = Path(__file__).resolve().parent
img = cv2.imread(str(ROOT / "CONTOH" / "6.JPG"))
H, W = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dark = (gray < 100).astype(np.uint8)
n, lab, stats, _c = cv2.connectedComponentsWithStats(dark, connectivity=8)
g = [tuple(stats[i][:5]) for i in range(1, n)
     if 3 <= stats[i][3] <= 20 and 1 <= stats[i][2] <= 22
     and 6 <= stats[i][4] <= stats[i][2] * stats[i][3] * 0.95]
g.sort(key=lambda p: (p[1], p[0]))

lines: list[list[tuple]] = []
for p in g:
    x, y, w, h, _a = p
    for ln in lines:
        ly0 = min(q[1] for q in ln); ly1 = max(q[1] + q[3] for q in ln)
        lx0 = min(q[0] for q in ln); lx1 = max(q[0] + q[2] for q in ln)
        if (min(y + h, ly1) - max(y, ly0)) > 0.45 * min(h, ly1 - ly0) \
           and (lx0 - 26) <= x <= (lx1 + 26):
            ln.append(p)
            break
    else:
        lines.append([p])
lines = [ln for ln in lines if len(ln) >= 3]

fp = next(p for p in sorted((ROOT / "fonts").rglob("*.ttf")) if "ace" in p.name.lower())
cap = {s: (lambda b: b[3] - b[1])(ImageFont.truetype(str(fp), s).getbbox("HAMBURG"))
       for s in range(5, 44)}

print(f"referensi {W}x{H}  baris terukur={len(lines)}   font kita={fp.name}")
print(f"\n{'baris':>5} {'x0':>5} {'x1':>5} {'lebar':>6} {'glyph':>5} {'capH':>5} "
      f"{'px/glyph':>8} | {'ace_size':>8} {'ace px/glyph':>12} {'rasio':>6}")
tot = []
for i, ln in enumerate(sorted(lines, key=lambda l: (min(q[1] for q in l)))):
    x0 = min(q[0] for q in ln); x1 = max(q[0] + q[2] for q in ln)
    caps = sorted(q[3] for q in ln)
    ch = float(np.median(caps[-max(2, len(caps) // 3):]))
    ng = len(ln)                      # komponen gelap ~ satu per huruf
    ppg = (x1 - x0) / max(ng, 1)
    size = min(cap, key=lambda s: abs(cap[s] - ch))
    f = ImageFont.truetype(str(fp), size)
    # advance rata-rata Anime Ace untuk huruf kapital pada ukuran itu
    ace = f.getlength("ABCDEFGHIJKLMNOPQRSTUVWXYZ") / 26.0
    tot.append(ppg / ace)
    print(f"{i:>5} {x0:>5} {x1:>5} {x1-x0:>6} {ng:>5} {ch:>5.1f} {ppg:>8.2f} | "
          f"{size:>8} {ace:>12.2f} {ppg/ace:>6.2f}")

print(f"\nrasio kerapatan referensi / Anime Ace: median={np.median(tot):.3f} "
      f"(mean={np.mean(tot):.3f})  -> <1.0 artinya referensi LEBIH RAPAT")

# Uji langsung: apakah 'WONDER' muat di kolom 43 px (bbox r3 kita) pada tiap ukuran?
print("\n'WONDER' di Anime Ace vs lebar kolom r3 (43 px) dan blok referensi (34 px):")
for s in range(6, 13):
    f = ImageFont.truetype(str(fp), s)
    print(f"  size {s:>2}: lebar WONDER = {f.getlength('WONDER'):>6.1f} px   "
          f"muat 43px={'ya' if f.getlength('WONDER') <= 43 else 'TIDAK':<5} "
          f"muat 34px={'ya' if f.getlength('WONDER') <= 34 else 'TIDAK'}")
