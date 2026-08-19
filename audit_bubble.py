#!/usr/bin/env python3
"""Audit KEBERSIHAN balon pada halaman yang sudah dihapus, offline.

Bedanya dengan audit di run_full.py — yang memberi false positive: di sini area
yang diperiksa bukan `ink_mask` (yang tepinya menyeberang ke garis balon dan
rambut), tapi INTERIOR balon yang dihitung ulang dari gambar ASLI lalu dikikis
lagi supaya garis balon jelas di luar. Piksel yang SUDAH gelap di gambar asli
DAN masih gelap di hasil hanya dilaporkan kalau ia berada di area bekas teks —
itu definisi "sisa tinta", bukan "art yang kebetulan gelap".

    python audit_bubble.py hasilnew/jp_6.JPG clean_jp_6.png
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
NBSRC, STAGE = ROOT / "_nbsrc", ROOT / ".stage"
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))
STAGE.mkdir(exist_ok=True)
for s in sorted(NBSRC.glob("*.py")):
    body = _MAGIC.sub("", s.read_text(encoding="utf-8"), count=1)
    d = STAGE / s.name
    if not d.exists() or d.read_text(encoding="utf-8") != body:
        d.write_text(body, encoding="utf-8")
sys.path.insert(0, str(STAGE))

import detect, imgio, textmask                          # noqa: E402,E401

src_path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "hasilnew/jp_6.JPG")
out_path = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "clean_jp_6.png")

img = imgio.load_any(src_path)
out = cv2.cvtColor(cv2.imread(str(out_path)), cv2.COLOR_BGR2RGB)
assert out.shape == img.shape, f"{out.shape} != {img.shape}"

regions, bubbles = detect.detect(img)
soft = textmask.ctd_soft_mask(img)
for r in regions:
    textmask.build_region_mask(img, r, soft)

g_src = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.int16)
g_out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.int16)
h, w = g_src.shape

vis = cv2.cvtColor(out, cv2.COLOR_RGB2BGR).copy()
total = 0
for r in regions:
    if r.bubble_bbox is None:
        continue
    bx1, by1, bx2, by2 = r.bubble_bbox
    # Interior segar dari gambar asli, dikikis 3x stroke: garis balon dan pita
    # anti-aliasing-nya pasti di luar, jadi yang gelap di dalam bukan garis.
    stroke = textmask._stroke_px(r.est_font_size or 20)
    inter = textmask._interior_from_crop(
        img[by1:by2, bx1:bx2], stroke * 3, textmask._ink_center(r, bx1, by1))
    m = inter > 0
    if not m.any():
        continue
    so, oo = g_src[by1:by2, bx1:bx2], g_out[by1:by2, bx1:bx2]
    bg = float(np.median(oo[m]))
    # Sisa tinta = gelap di HASIL, di dalam interior, dan memang tempat tinta
    # asli berada (gelap juga di ASLI). Piksel yang cuma gelap di hasil berarti
    # cacat isian — dilaporkan terpisah supaya dua sebab tidak tercampur.
    dark_out = (np.abs(oo - bg) > 16) & m
    dark_src = (np.abs(so - bg) > 16) & m
    resid = dark_out & dark_src
    n, lab, st, _ = cv2.connectedComponentsWithStats(resid.astype(np.uint8), 8)
    comps = []
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        if area < 2:
            continue
        long_s, short_s = max(cw, ch), max(min(cw, ch), 1)
        comps.append((area, (int(x + bx1), int(y + by1), int(cw), int(ch)),
                      "garis" if (long_s >= 10 or long_s / short_s >= 3) else "titik"))
    comps.sort(reverse=True)
    total += len(comps)
    print(f"r{r.idx} bub={r.bubble_bbox} bg={bg:.0f} interior={int(m.sum())} "
          f"sisa_px={int(resid.sum())} komponen={len(comps)}")
    for a, bb, kind in comps[:6]:
        print(f"    {kind:6} area={a:4} bbox={bb}")
    for a, bb, kind in comps:
        x, y, cw, ch = bb
        cv2.rectangle(vis, (x - 1, y - 1), (x + cw + 1, y + ch + 1),
                      (0, 0, 255) if kind == "garis" else (0, 165, 255), 1)

print(f"\nTOTAL komponen sisa: {total}")
cv2.imwrite("_audit_" + out_path.stem + ".png", vis)
print("-> _audit_" + out_path.stem + ".png (merah=garis, oranye=titik)")
