#!/usr/bin/env python3
"""Baris MANA milik r3 yang pecah setelah reclaim produksi, dan kenapa.

probe_reclaim2.py menunjukkan hasil bersihnya buruk: r3 7->6 dan hyphen tetap 1,
padahal simulasi tambah-saja memberi 7->7 hyphen 1->0. Bedanya, produksi
MEMINDAH: r2 (idx lebih kecil, dapat giliran lebih dulu) mengambil 413 px dari
interior r3 di band bawah (y=194..202 bebasnya 45->21 px).

Yang diuji di sini satu dugaan yang bisa salah: penjaga `~ink` melindungi KOTAK
tinta pelepas, tapi tata letak butuh RUN bebas yang menyambung di band baris itu
— begitu tetangga mengiris satu sisi, run-nya pecah jadi dua dan _band_run()
melaporkan lebar yang lebih kecil daripada kotak tintanya sendiri.

Dicetak per baris fase-1 r3 pada size 7: band y-nya, run bebas terlebar sebelum
vs sesudah, dan lebar yang baris itu butuhkan.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NBSRC, STAGE = ROOT / "_nbsrc", ROOT / ".stage"
CACHE = ROOT / ".probe_cache6.pkl"
PAGE = ROOT / "hasilnew" / "jp_6.JPG"
REPORT = ROOT / "debug" / "jp_6" / "report.json"
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))
STAGE.mkdir(exist_ok=True)
for _s in sorted(NBSRC.glob("*.py")):
    _b = _MAGIC.sub("", _s.read_text(encoding="utf-8"), count=1)
    _d = STAGE / _s.name
    if not _d.exists() or _d.read_text(encoding="utf-8") != _b:
        _d.write_text(_b, encoding="utf-8")
sys.path.insert(0, str(STAGE))

import numpy as np                                          # noqa: E402
import imgio, typeset                                       # noqa: E402,E401

FOCUS = int(os.environ.get("FOCUS", 3))

fp = typeset.setup_fonts(verbose=False)
img = imgio.load_any(PAGE)
H, W = img.shape[:2]
typeset.set_page_width(W)
with CACHE.open("rb") as f:
    regions = pickle.load(f)
texts = {r["idx"]: r["translation"] for r in
         json.loads(REPORT.read_text(encoding="utf-8"))["regions"]}
for r in regions:
    r.translation = texts.get(r.idx) or ""

tgt = next(r for r in regions if r.idx == FOCUS)
b0, m0 = typeset._region_box_mask(tgt)
m0 = m0.copy()
t = tgt.translation.upper()
cap0 = typeset.region_font_cap(m0)
s0, l0, y0, _o0 = typeset.fit(t, m0, cap0, fp)

typeset.reclaim_unused_interiors(img, regions)
b1, m1 = typeset._region_box_mask(tgt)
m1 = m1.copy()
cap1 = typeset.region_font_cap(m1)
s1, l1, y1, _o1 = typeset.fit(t, m1, cap1, fp)

print(f"r{FOCUS}  kotak {b0} {m0.shape} -> {b1} {m1.shape}")
print(f"  plafon {cap0} -> {cap1}   fit size {s0} -> {s1}")
print(f"  fase1: {' | '.join(l0)}")
print(f"  fase2: {' | '.join(l1)}")

font = typeset._font(fp, s0)
cmap = typeset._cmap(fp)
lh = typeset._line_height(font)
it, ib = typeset._ink_band(fp, s0)


def runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Semua run piksel interior (x0, panjang) di satu baris."""
    out, run, start = [], 0, 0
    for x, v in enumerate(row > 0):
        if v:
            if run == 0:
                start = x
            run += 1
        elif run:
            out.append((start, run))
            run = 0
    if run:
        out.append((start, run))
    return out


def band_best(mask: np.ndarray, oy: int, ya: int, yb: int) -> int:
    """Run yang menyambung sepanjang band ya..yb (irisan antar baris)."""
    ra, rb = ya - oy, yb - oy
    if ra < 0 or rb > mask.shape[0]:
        return 0
    band = mask[ra:rb]
    if band.size == 0:
        return 0
    col = np.all(band > 0, axis=0).astype(np.uint8)
    return max((n for _s, n in runs(col)), default=0)


print(f"\nband tiap baris fase-1 (size {s0}, tinggi baris {lh}, "
      f"tinta {it}..{ib})")
print(f"{'baris':>12} {'y':>10} {'butuh':>7} {'run0':>6} {'run1':>6}  vonis")
for k, ln in enumerate(l0):
    need = typeset._line_width(ln, font, cmap, s0)
    ya = b0[1] + y0 + k * lh + it
    yb = b0[1] + y0 + k * lh + ib + 1
    r0 = band_best(m0, b0[1], ya, yb)
    r1 = band_best(m1, b1[1], ya, yb)
    print(f"{ln!r:>12} {f'{ya}..{yb}':>10} {need:>7.1f} {r0:>6} {r1:>6}  "
          f"{'PECAH' if r1 < need <= r0 else ''}")

# Dan yang paling menentukan: apakah tata letak fase-1 masih sah di mask baru?
ok0, lay0, _a0 = typeset.layout(t, m0, s0, fp, hyphenate=False)
ok1, lay1, _a1 = typeset.layout(t, m1, s0, fp, hyphenate=False)
okh1, layh1, _ah1 = typeset.layout(t, m1, s0, fp, hyphenate=True)
print(f"\nlayout utuh @size {s0}: mask lama {'ya' if ok0 else '-'} "
      f"({' | '.join(lay0)})")
print(f"layout utuh @size {s0}: mask baru {'ya' if ok1 else '-'} "
      f"({' | '.join(lay1)})")
print(f"layout hyph @size {s0}: mask baru {'ya' if okh1 else '-'} "
      f"({' | '.join(layh1)})")
