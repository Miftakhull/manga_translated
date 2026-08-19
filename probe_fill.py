#!/usr/bin/env python3
"""Margin sisi & keterisian baris yang BENAR-BENAR dihasilkan, per pad_ratio.

pad_ratio bukan besaran yang bisa dibandingkan langsung ke referensi: referensi
tidak punya pad_ratio, ia punya HASIL — margin sisi 0.165 x sisi terpendek dan
lebar tinta 70% lebar interior (probe_refnative.py). Margin kita muncul dari dua
sumber sekaligus: pad eksplisit DAN sisa akibat baris pecah di batas kata. Jadi
yang dibandingkan harus hasilnya, bukan parameternya.

Untuk tiap kandidat pad_ratio: tata teks pada ukuran model proporsional, lalu
ukur lebar baris terlebar terhadap lebar interior di baris itu.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / ".stage"
PRE = ROOT / ".probe_pre.pkl"
TEXTS = ROOT / "probe_font_texts.json"
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))

STAGE.mkdir(exist_ok=True)
for _s in sorted((ROOT / "_nbsrc").glob("*.py")):
    _b = _MAGIC.sub("", _s.read_text(encoding="utf-8"), count=1)
    _d = STAGE / _s.name
    if not _d.exists() or _d.read_text(encoding="utf-8") != _b:
        _d.write_text(_b, encoding="utf-8")
sys.path.insert(0, str(STAGE))

import imgio     # noqa: E402
import textmask  # noqa: E402
import typeset   # noqa: E402
from config import SETTINGS  # noqa: E402

REF_SIDE = 0.165   # margin_sisi / min(sisi) di referensi
REF_FILL = 0.70    # lebar tinta / lebar interior
SIZE_RATIO = 0.117 / 0.844   # model B: cap/min dibagi cap/size


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with PRE.open("rb") as f:
        regions = pickle.load(f)
    textmask.disjoin_overlapping_interiors(img, regions)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    SETTINGS.line_spacing = float(os.environ.get("LS", 0.95))
    masks = {r.idx: typeset._region_box_mask(r)[1] for r in regions}

    print(f"line_spacing={SETTINGS.line_spacing}   "
          f"target referensi: sisi/min={REF_SIDE:.3f} isi={REF_FILL*100:.0f}%")
    print(f"  {'pad':>5} {'nol':>4} {'sisi/min':>9} {'isi':>6} {'ukuran':>18}")
    for pr in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12):
        SETTINGS.pad_ratio = pr
        sides, fills, sizes, zeros = [], [], [], 0
        for r in regions:
            t = str(texts.get(str(r.idx), "")).upper()
            m = masks[r.idx]
            mh, mw = m.shape[:2]
            mn = min(mh, mw)
            cap = int(round(mn * SIZE_RATIO))
            size, lines, sy, over = typeset.fit(t, m, cap, fp)
            if over or not lines:
                zeros += 1
                continue
            font = typeset._font(fp, size)
            lh = typeset._line_height(font)
            ink_top, ink_bot = typeset._ink_band(fp, size)
            widest, avail_there = 0.0, 1.0
            for k, ln in enumerate(lines):
                lw = typeset._measure(ln, font)
                y0 = sy + k * lh + ink_top
                y1 = sy + k * lh + ink_bot
                band = m[max(y0, 0):max(y1, 1)] > 0
                cols = int(band.any(0).sum()) if band.size else 1
                if lw > widest:
                    widest, avail_there = lw, max(cols, 1)
            sides.append((avail_there - widest) / 2 / mn)
            fills.append(widest / avail_there)
            sizes.append(size)
        print(f"  {pr:>5.3f} {zeros:>4} {np.median(sides):>9.3f} "
              f"{np.median(fills)*100:>5.0f}% "
              f"{f'{min(sizes)}..{max(sizes)}':>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
