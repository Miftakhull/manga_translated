#!/usr/bin/env python3
"""Ukur typeset referensi CONTOH/6.JPG secara numerik, lalu bandingkan ke output kita.

Pertanyaan yang dijawab, offline dan nol token:
  1. Berapa ukuran font yang dipakai referensi di tiap balon (px), dan seberapa
     seragam antar balon?
  2. Seberapa LEBAR blok teks tiap lobus balon ganda di referensi, setelah
     diskalakan ke halaman kita (698 px)? Ini yang menentukan apakah 'WONDER'
     seharusnya muat satu baris.
  3. Apakah referensi menggabung dua lobus jadi satu blok, atau memisah?
Metode: komponen gelap seukuran glyph -> klaster baris -> klaster blok.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import ImageFont
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REF = ROOT / "CONTOH" / "6.JPG"
OUR_W = 698  # lebar halaman kita (hasilnew/jp_6.JPG)

img = cv2.imread(str(REF))
H, W = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- glyph candidates -------------------------------------------------------
# Teks referensi hitam pekat di atas interior balon putih. Ambang keras plus
# batas ukuran komponen menyingkirkan garis balon (panjang) dan raster art.
dark = (gray < 100).astype(np.uint8)
n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, connectivity=8)
glyphs = []
for i in range(1, n):
    x, y, w, h, a = stats[i]
    if not (3 <= h <= 20 and 1 <= w <= 22):
        continue
    if a < 6 or a > w * h * 0.95:
        continue
    glyphs.append((x, y, w, h, a))
print(f"referensi {W}x{H}  kandidat glyph={len(glyphs)}  skala->kita={OUR_W/W:.4f}")

# --- cluster into lines (y-overlap + x proximity) ---------------------------
glyphs.sort(key=lambda g: (g[1], g[0]))
lines: list[list[tuple]] = []
for g in glyphs:
    x, y, w, h, _a = g
    placed = False
    for ln in lines:
        ly0 = min(p[1] for p in ln)
        ly1 = max(p[1] + p[3] for p in ln)
        lx0 = min(p[0] for p in ln)
        lx1 = max(p[0] + p[2] for p in ln)
        ov = min(y + h, ly1) - max(y, ly0)
        if ov > 0.45 * min(h, ly1 - ly0) and (lx0 - 26) <= x <= (lx1 + 26):
            ln.append(g)
            placed = True
            break
    if not placed:
        lines.append([g])
lines = [ln for ln in lines if len(ln) >= 2]

def lbox(ln):
    x0 = min(p[0] for p in ln); y0 = min(p[1] for p in ln)
    x1 = max(p[0] + p[2] for p in ln); y1 = max(p[1] + p[3] for p in ln)
    return x0, y0, x1, y1

# --- cluster lines into blocks (x-overlap + vertical proximity) -------------
lines.sort(key=lambda ln: lbox(ln)[1])
blocks: list[list[list[tuple]]] = []
for ln in lines:
    x0, y0, x1, y1 = lbox(ln)
    best = None
    for b in blocks:
        bx0, by0, bx1, by1 = lbox([g for l2 in b for g in l2])
        xov = min(x1, bx1) - max(x0, bx0)
        gap = y0 - by1
        if xov > 0.30 * min(x1 - x0, bx1 - bx0) and -4 <= gap <= 22:
            best = b
            break
    (best if best is not None else blocks.append([]) or blocks[-1]).append(ln)

# --- font size mapping: cap height (px) -> Anime Ace point size -------------
fpath = None
for cand in sorted((ROOT / "fonts").rglob("*.ttf")):
    if "ace" in cand.name.lower():
        fpath = cand
        break
cap_of = {}
if fpath:
    for s in range(5, 40):
        f = ImageFont.truetype(str(fpath), s)
        bb = f.getbbox("HAMBURG")
        cap_of[s] = bb[3] - bb[1]

def size_from_cap(cap: float) -> int:
    if not cap_of:
        return 0
    return min(cap_of, key=lambda s: abs(cap_of[s] - cap))

print(f"\nfont referensi utk pemetaan: {fpath.name if fpath else 'TIDAK ADA'}")
print(f"\n{'#':>2} {'bbox referensi':>24} {'lines':>5} {'cap':>4} {'size':>4} "
      f"{'maxline_w':>9} | {'-> koord kita':>20} {'w':>4} {'size':>4} {'maxline_w':>9}")
sc = OUR_W / W
rows = []
for i, b in enumerate(sorted(blocks, key=lambda b: -lbox([g for l2 in b for g in l2])[0])):
    allg = [g for l2 in b for g in l2]
    x0, y0, x1, y1 = lbox(allg)
    caps = sorted(p[3] for p in allg)
    cap = float(np.median(caps[-max(3, len(caps) // 3):]))  # huruf kapital
    widths = [lbox(l2)[2] - lbox(l2)[0] for l2 in b]
    mw = max(widths)
    size = size_from_cap(cap)
    rows.append((x0, y0, x1, y1, len(b), cap, size, mw))
    print(f"{i:>2} {f'[{x0},{y0},{x1},{y1}]':>24} {len(b):>5} {cap:>4.1f} {size:>4} "
          f"{mw:>9} | {f'[{x0*sc:.0f},{y0*sc:.0f},{x1*sc:.0f},{y1*sc:.0f}]':>20} "
          f"{(x1-x0)*sc:>4.0f} {size*sc:>4.1f} {mw*sc:>9.1f}")

if rows:
    sizes = [r[6] for r in rows]
    print(f"\nsize referensi: min={min(sizes)} max={max(sizes)} "
          f"median={int(np.median(sizes))} rasio max/min={max(sizes)/max(min(sizes),1):.2f}")
    print(f"size referensi diskalakan ke {OUR_W}px: "
          f"{[round(s*sc,1) for s in sorted(set(sizes))]}")
