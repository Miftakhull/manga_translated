#!/usr/bin/env python3
"""Kenapa klaim r0 di halaman uji balon-bertetangga DITOLAK produksi.

probe_adjfind.py mengukur SELURUH sanderaan (9522 px) mengubah 2 tanda hubung
jadi 1. Produksi menolak. Tiga tersangka, dipisahkan di sini satu-satu:

  (a) `~ink`      — kotak tinta r1 memakan bagian sanderaan, jadi yang benar-benar
                    boleh diklaim jauh lebih kecil daripada 9522 px.
  (b) syarat terima — fit() pada klaim yang lebih kecil itu tidak memperbaiki apa pun.
  (c) syarat pelepas — r1 rugi lebih dari satu poin, jadi klaim dibatalkan.

Dicetak: luas tiap tahap penyaringan, lalu fit() r0 pada masing-masing, lalu
fit() r1 sebelum/sesudah kehilangan klaim itu.
"""

from __future__ import annotations

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
for _s in sorted(NBSRC.glob("*.py")):
    _b = _MAGIC.sub("", _s.read_text(encoding="utf-8"), count=1)
    _d = STAGE / _s.name
    if not _d.exists() or _d.read_text(encoding="utf-8") != _b:
        _d.write_text(_b, encoding="utf-8")
sys.path.insert(0, str(STAGE))

import numpy as np                                          # noqa: E402
import selftest, textmask, typeset                          # noqa: E402,E401

TA = os.environ.get("TA", "MISUNDERSTANDING AGAIN, PREZ?")
TB = os.environ.get("TB", "YES, THE RECORDS ARE HERE.")

fp = typeset.setup_fonts(verbose=False)
clean, img, inner, regions = selftest.make_adjacent_bubbles_page()
for r in regions:
    textmask.build_region_mask(img, r, None)
textmask.disjoin_overlapping_interiors(img, regions)
typeset.set_page_width(img.shape[1])
H, W = img.shape[:2]
regions[0].translation, regions[1].translation = TA, TB

maps = [typeset._paste_mask(*typeset._region_box_mask(r), H, W) > 0 for r in regions]
fills = [typeset._paste_mask(r.fill_bbox, r.fill_mask, H, W) > 0
         if r.fill_mask is not None else np.zeros((H, W), bool) for r in regions]


def lay(mp, text):
    ys, xs = np.nonzero(mp)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    m = np.where(mp[box[1]:box[3], box[0]:box[2]], 255, 0).astype(np.uint8)
    s, ls, sy, ov = typeset.fit(text.upper(), m, typeset.region_font_cap(m), fp)
    return box, m, s, ls, sy, ov


def bands(st) -> np.ndarray:
    box, m, s, ls, sy, _ov = st
    out = np.zeros((H, W), bool)
    for y0, y1, x0, x1 in typeset._line_bands(box, m, s, ls, sy, fp):
        out[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    return out


st0 = [lay(maps[i], regions[i].translation) for i in (0, 1)]
ink = bands(st0[0]) | bands(st0[1])

lost = fills[0] & ~maps[0]
step_other = lost & maps[1]
step_ink = step_other & ~ink
print(f"fill r0 di luar interior r0      : {int(lost.sum())} px")
print(f"  yang diambil r1 (& maps[1])    : {int(step_other.sum())} px")
print(f"  setelah dikurangi kotak tinta  : {int(step_ink.sum())} px "
      f"(dimakan tinta r1: {int((step_other & ink).sum())} px)")

for name, cand in (("SELURUH sanderaan", step_other), ("yang boleh (produksi)", step_ink)):
    if not cand.any():
        print(f"\n{name}: kosong")
        continue
    a = lay(maps[0] | cand, regions[0].translation)
    b = lay(maps[1] & ~cand, regions[1].translation)
    ha = sum(1 for x in a[3] if x.endswith("-"))
    hb = sum(1 for x in b[3] if x.endswith("-"))
    h0 = sum(1 for x in st0[0][3] if x.endswith("-"))
    h1 = sum(1 for x in st0[1][3] if x.endswith("-"))
    print(f"\n{name} ({int(cand.sum())} px)")
    print(f"  r0 {st0[0][2]}/{h0} -> {a[2]}/{ha}  {' | '.join(a[3])}")
    print(f"  r1 {st0[1][2]}/{h1} -> {b[2]}/{hb}  {' | '.join(b[3])}")
    print(f"  vonis: pengklaim {'MEMBAIK' if ha < h0 else 'tidak membaik'}; "
          f"pelepas {'DIRUGIKAN' if (hb > h1 or b[2] < st0[1][2] - 1) else 'aman'}")
