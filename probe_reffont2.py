#!/usr/bin/env python3
"""Kerapatan font referensi CONTOH/6.JPG vs Anime Ace — pengukuran yang benar.

probe_reffont.py memakai kotak tangan yang lebih PENDEK dari hurufnya, jadi
tinggi kapital ikut terpotong tinggi kotak (ink_h == tinggi crop) dan rasionya
tidak sah. Di sini baris ditemukan dari komponen glyph (tidak ada kotak tangan),
tinggi kapital = median tinggi komponen kapital baris itu, dan lebar baris =
jarak tepi tinta kiri-kanan komponen baris itu. Offline, nol token.
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

dark = (gray < 100).astype(np.uint8)
n, _lab, st, _c = cv2.connectedComponentsWithStats(dark, connectivity=8)
g = [tuple(int(v) for v in st[i][:5]) for i in range(1, n)
     if 3 <= st[i][3] <= 20 and 1 <= st[i][2] <= 22
     and 6 <= st[i][4] <= st[i][2] * st[i][3] * 0.95]
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

def box(ln):
    return (min(q[0] for q in ln), min(q[1] for q in ln),
            max(q[0] + q[2] for q in ln), max(q[1] + q[3] for q in ln))

# Teks yang sudah dibaca mata dari zoom, dengan titik acuan kasar (cx, cy).
# Baris dicocokkan ke klaster komponen TERDEKAT, jadi kotaknya tidak perlu tepat.
WANT = [
    ("EMBARASSING",     486, 277),
    ("TO BE",           486, 292),
    ("DESCRIBED",       490, 307),
    ("THAT WAY...",     492, 322),
    ("TOO EXCITED AND", 1046, 266),
    ("MOSTLY CAME BY",  1046, 280),
    ("NEVER SEEN",      712, 217),
    ("ANYONE AS",       712, 235),
    ("I'M PRAISING",    235, 319),
    ("BOTTOM OF",       232, 350),
]

cap_of = {s: (lambda b: b[3] - b[1])(ImageFont.truetype(str(fp), s).getbbox("HAMBURG"))
          for s in range(6, 46)}

print(f"font kita: {fp.name}   baris terdeteksi: {len(lines)}")
print(f"\n{'teks':>17} {'bbox baris referensi':>22} {'w':>4} {'capH':>5} "
      f"{'size':>4} {'ace_w':>6} {'rasio':>6}")
ratios = []
for txt, cx, cy in WANT:
    best, bd = None, 1e9
    for ln in lines:
        x0, y0, x1, y1 = box(ln)
        d = abs((x0 + x1) / 2 - cx) + abs((y0 + y1) / 2 - cy) * 2.0
        if d < bd:
            best, bd = ln, d
    x0, y0, x1, y1 = box(best)
    hs = sorted(q[3] for q in best)
    capH = float(np.median(hs[-max(2, len(hs) // 3):]))   # kapital, bukan koma
    size = min(cap_of, key=lambda s: abs(cap_of[s] - capH))
    ace_w = ImageFont.truetype(str(fp), size).getlength(txt)
    ratios.append((x1 - x0) / ace_w)
    print(f"{txt:>17} {f'[{x0},{y0},{x1},{y1}]':>22} {x1-x0:>4} {capH:>5.1f} "
          f"{size:>4} {ace_w:>6.1f} {(x1-x0)/ace_w:>6.3f}")

med = float(np.median(ratios))
print(f"\nrasio lebar referensi / Anime Ace pada tinggi kapital sama: "
      f"median={med:.3f}  min={min(ratios):.3f}  max={max(ratios):.3f}")
print(f"-> pada tinggi huruf sama, referensi memuat ~{1/med:.2f}x lebih banyak "
      f"karakter per baris." if med < 1 else "-> Anime Ace justru lebih rapat.")

# Tinggi kapital referensi diskalakan ke halaman kita (698 px).
sc = 698 / img.shape[1]
caps = []
for ln in lines:
    if len(ln) < 3:
        continue
    hs = sorted(q[3] for q in ln)
    caps.append(float(np.median(hs[-max(2, len(hs) // 3):])))
caps = np.array(caps)
print(f"\ntinggi kapital referensi: median={np.median(caps):.1f} px "
      f"p10={np.percentile(caps,10):.1f} p90={np.percentile(caps,90):.1f}")
print(f"diskalakan ke 698 px: median={np.median(caps)*sc:.1f} px "
      f"p10={np.percentile(caps,10)*sc:.1f} p90={np.percentile(caps,90)*sc:.1f}")
print(f"ukuran Anime Ace dengan capH itu: "
      f"{min(cap_of, key=lambda s: abs(cap_of[s]-np.median(caps)*sc))}")
