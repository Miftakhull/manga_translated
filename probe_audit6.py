#!/usr/bin/env python3
"""Triase 9 komponen yang dilaporkan audit_clean() di jp_6 — OFFLINE, tanpa faucet.

audit_clean() memakai dev_thr=16, lebih ketat daripada SETTINGS.residue_deviation=20
yang dipakai gate pipeline (yang melaporkan residue_count=0). Selisih itulah yang
harus dinilai per komponen: benar-benar sisa tinta Jepang, atau memang bagian
gambar (garis balon, ekor balon, rambut, garis panel) yang kebetulan masuk mask?

Yang dicetak: potongan 01_input dan 09_cleaned di sekitar tiap bbox, diperbesar,
disusun berdampingan supaya bisa dilihat langsung. Tidak ada tulisan ke pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DBG = ROOT / "debug" / "jp_6"

COMPS = [
    ("r2 garis 41", (409, 96, 414, 115)),
    ("r3 garis 22", (320, 141, 324, 153)),
    ("r6 titik 32", (156, 146, 167, 156)),
    ("r6 titik 17", (114, 149, 119, 156)),
    ("r2 titik 18", (397, 190, 403, 196)),
    ("r2 garis 4", (345, 104, 346, 108)),
]

PAD, Z = 14, 6

src = cv2.imread(str(DBG / "01_input.png"))
cln = cv2.imread(str(DBG / "09_cleaned.png"))
assert src is not None and cln is not None
H, W = cln.shape[:2]
print(f"[img] {W}x{H}")

tiles = []
for name, (x1, y1, x2, y2) in COMPS:
    ax, ay = max(x1 - PAD, 0), max(y1 - PAD, 0)
    bx, by = min(x2 + PAD, W), min(y2 + PAD, H)
    pair = []
    for img in (src, cln):
        crop = cv2.resize(img[ay:by, ax:bx], None, fx=Z, fy=Z,
                          interpolation=cv2.INTER_NEAREST)
        # kotak merah = tepat komponen yang dilaporkan
        cv2.rectangle(crop, ((x1 - ax) * Z, (y1 - ay) * Z),
                      ((x2 - ax) * Z - 1, (y2 - ay) * Z - 1), (0, 0, 255), 1)
        pair.append(crop)
    h = max(p.shape[0] for p in pair)
    row = np.full((h + 18, sum(p.shape[1] for p in pair) + 8, 3), 240, np.uint8)
    xo = 0
    for p in pair:
        row[18:18 + p.shape[0], xo:xo + p.shape[1]] = p
        xo += p.shape[1] + 8
    cv2.putText(row, f"{name}  (kiri=input, kanan=cleaned)", (2, 13),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
    tiles.append(row)

    # statistik piksel: seberapa gelap sisanya dibanding latar balon
    g = cv2.cvtColor(cln[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    print(f"{name:14s} bbox={(x1, y1, x2, y2)} min={g.min()} "
          f"mean={g.mean():.1f} <240px={int((g < 240).sum())}")

wid = max(t.shape[1] for t in tiles)
out = np.full((sum(t.shape[0] + 6 for t in tiles), wid, 3), 255, np.uint8)
yo = 0
for t in tiles:
    out[yo:yo + t.shape[0], 0:t.shape[1]] = t
    yo += t.shape[0] + 6
dst = ROOT / "_cmp" / "audit6.png"
dst.parent.mkdir(exist_ok=True)
cv2.imwrite(str(dst), out)
print(f"[tulis] {dst}")
