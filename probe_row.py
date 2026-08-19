#!/usr/bin/env python3
"""Apakah kolom yang diambil dari r3 benar-benar DIPAKAI tinta r2?

Bagian 1 tetap: sifat mask tiap region (punya balon sungguhan atau persegi
255 penuh) — kalau r3 cuma persegi, irisannya semu dan bukan geometri balon.

Bagian 2 yang menentukan: untuk r2, fit() dijalankan pada mask
SESUDAH disjoin, lalu kotak tinta tiap barisnya dihitung analitik dari
line_axis() + _line_width() (bukan dirender, supaya murah dan pasti sama dengan
yang dipakai render_region). Hasilnya dibandingkan baris-per-baris dengan
kolom yang HILANG dari r3 di ketinggian yang sama.

Kalau di baris tertentu r2 tidak menaruh tinta sama sekali tapi r3 kehilangan
kolom di situ, maka lebar itu disandera tanpa dipakai — dan itulah yang
'lebar tersedia per BARIS' harus kembalikan.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

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

import numpy as np                                     # noqa: E402
import detect, imgio, textmask, typeset                # noqa: E402,E401

REPORT = json.load(open(ROOT / "debug/jp_6/report.json", encoding="utf-8"))
OURS = {r["idx"]: r["translation"] for r in REPORT["regions"]}

fp = typeset.setup_fonts(verbose=False)
img = imgio.load_any(ROOT / "hasilnew/jp_6.JPG")
typeset.set_page_width(img.shape[1])
regions, _ = detect.detect(img)
soft = textmask.ctd_soft_mask(img)
for r in regions:
    textmask.build_region_mask(img, r, soft)
textmask.partition_shared_interiors(img, regions)
pre = {r.idx: (lambda t: (t[0], t[1].copy()))(typeset._region_box_mask(r))
       for r in regions}
textmask.disjoin_overlapping_interiors(img, regions)
byidx = {r.idx: r for r in regions}

print("=== sifat mask tiap region ===")
print(f"{'r':>2} {'kelas':>12} {'balon?':>6} {'mask':>9} {'isi%':>6} {'kotak':>22}")
for r in regions:
    box, m = typeset._region_box_mask(r)
    full = "255pnh" if r.bubble_mask is None else "balon"
    print(f"{r.idx:>2} {r.det_class:>12} {full:>6} "
          f"{m.shape[1]}x{m.shape[0]:<6} {100 * (m > 0).mean():>5.1f}% {str(box):>22}")

# ---- kotak tinta r2 per baris, analitik -------------------------------------
print("\n=== tinta r2 per baris (analitik, sama dengan render_region) ===")
r2 = byidx[2]
box2, m2 = typeset._region_box_mask(r2)
size2, lines2, sy2, over2 = typeset.fit(OURS[2].upper(), m2,
                                        typeset.region_font_cap(m2), fp)
font2 = typeset._font(fp, size2)
lh2 = typeset._line_height(font2)
it2, ib2 = typeset._ink_band(fp, size2)
ax2 = typeset.line_axis(m2, lines2, sy2, size2, fp)
cmap = typeset._cmap(fp)
bands: list[tuple[int, int, int, int]] = []   # (y0abs, y1abs, x0abs, x1abs)
print(f"size={size2} lh={lh2} ink_band=({it2},{ib2}) axis_lokal={ax2} kotak={box2}")
for k, ln in enumerate(lines2):
    w = typeset._line_width(ln, font2, cmap, size2)
    y0 = box2[1] + sy2 + k * lh2 + it2
    y1 = box2[1] + sy2 + k * lh2 + ib2
    x0 = box2[0] + int(ax2 - w / 2)
    x1 = box2[0] + int(ax2 + w / 2)
    bands.append((y0, y1, x0, x1))
    print(f"  {k}: y {y0:>4}..{y1:<4} x {x0:>4}..{x1:<4} lebar={w:>5.1f}  {ln}")

# ---- baris r3: hilang berapa, dan apakah r2 memakainya ----------------------
print("\n=== r3: kolom hilang vs tinta r2 di ketinggian yang sama ===")
r3 = byidx[3]
(pb3, pm3), (qb3, qm3) = pre[3], typeset._region_box_mask(r3)
print(f"r3 {pb3} {pm3.shape[1]}x{pm3.shape[0]} -> {qb3} {qm3.shape[1]}x{qm3.shape[0]}")
print(f"{'y':>5} {'sblm':>5} {'ssdh':>5} {'hilang':>7} {'tinta r2 di y':>28}")
for ya in range(pb3[1], pb3[3], 4):
    def run(box, m):
        yl = ya - box[1]
        if yl < 0 or yl + 4 >= m.shape[0]:
            return None
        rr = typeset._band_run(m, yl, yl + 4)
        return None if rr is None else (rr[1] - rr[0] + 1)
    a, b = run(pb3, pm3), run(qb3, qm3)
    hit = [f"x{x0}..{x1}" for (y0, y1, x0, x1) in bands if y0 <= ya + 4 and y1 >= ya]
    lost = "" if a is None or b is None else str(a - b)
    print(f"{ya:>5} {str(a):>5} {str(b):>5} {lost:>7} {(', '.join(hit) or '-'):>28}")
