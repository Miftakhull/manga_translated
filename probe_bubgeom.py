#!/usr/bin/env python3
"""Lacak geometri balon di sepanjang process_page, OFFLINE (nol token).

Yang dicari: pada tahap mana `bubble_bbox`/`bubble_mask` sebuah region berubah,
dan seberapa besar ink_mask relatif interior balonnya. Angka terakhir itu yang
menentukan bentuk hapusan: ink_mask yang kelewat besar = hapusan berbentuk blob.
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

import detect, imgio, textmask, typeset            # noqa: E402,E401

path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "hasilnew/jp_6.JPG")
img = imgio.load_any(path)
print(f"[img] {path.name} {img.shape[1]}x{img.shape[0]}")

regions, bubbles = detect.detect(img)
print(f"[detect] region={len(regions)} bubble={len(bubbles)}")


def snap(tag: str) -> None:
    print(f"\n--- {tag} ---")
    for r in regions:
        bm = r.bubble_mask
        bmi = "-" if bm is None else f"{bm.shape[1]}x{bm.shape[0]} on={int((bm>0).sum())}"
        im = r.ink_mask
        imi = "-" if im is None else f"{im.shape[1]}x{im.shape[0]} on={int((im>0).sum())}"
        flat = "" if bm is None else (" FULLRECT" if bm.min() == 255 else "")
        print(f"  r{r.idx} bbox={r.bbox} bub={r.bubble_bbox} shared={r.shared_bubble_bbox}"
              f" est={r.est_font_size:.1f} bmask={bmi}{flat} ink={imi}")


snap("setelah detect")
soft = textmask.ctd_soft_mask(img)
print(f"[ctd] soft={'None' if soft is None else soft.shape}")
for r in regions:
    textmask.build_region_mask(img, r, soft)
snap("setelah build_region_mask")
print("[part] shared ->", textmask.partition_shared_interiors(img, regions))
snap("setelah partition_shared_interiors")
print("[disjoin] ->", textmask.disjoin_overlapping_interiors(img, regions))
snap("setelah disjoin_overlapping_interiors")
print("[outline] freed px ->", textmask.protect_bubble_outline(img, regions))
snap("setelah protect_bubble_outline")

# Rasio ink vs interior: > ~0.5 berarti mask hapus praktis seluruh balon dan
# bentuknya tidak lagi mengikuti glyph — itu yang tampak sebagai blob.
print("\n--- ink vs interior ---")
for r in regions:
    box, bm = textmask._eff_box_mask(r)
    inter = int((bm > 0).sum())
    ink = 0 if r.ink_mask is None else int((r.ink_mask > 0).sum())
    print(f"  r{r.idx} interior={inter} ink={ink} rasio={ink/max(inter,1):.3f}")

# Dump visual: interior (biru) vs ink (merah) di atas halaman.
vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
ov = vis.copy()
for r in regions:
    box, bm = textmask._eff_box_mask(r)
    bx1, by1, bx2, by2 = box
    sub = ov[by1:by2, bx1:bx2]
    m = (bm[: sub.shape[0], : sub.shape[1]] > 0)
    sub[m] = (0.5 * sub[m] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    if r.ink_mask is not None:
        x1, y1, x2, y2 = r.bbox
        s2 = ov[y1:y2, x1:x2]
        m2 = (r.ink_mask[: s2.shape[0], : s2.shape[1]] > 0)
        s2[m2] = (0.35 * s2[m2] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
cv2.imwrite(f"_geom_{path.stem}.png", np.vstack([vis, np.zeros((6, vis.shape[1], 3), np.uint8), ov]))
print(f"\n-> _geom_{path.stem}.png (atas asli, bawah biru=interior merah=ink)")
