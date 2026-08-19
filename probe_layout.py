#!/usr/bin/env python3
"""Kenapa satu region tidak bisa memakai font sebesar referensi.

Untuk tiap region: dimensi mask, pad, lebar tersedia per baris di sekitar
centroid, dan hasil layout() pada rentang ukuran — utuh maupun ber-hyphen.
Dipakai untuk memisahkan tiga tersangka: pad_ratio, _ROW_COVER, dan erosi
interior; bukan menebak mana yang menjepit ukuran.

Argumen opsional: daftar idx region yang mau didetailkan (default 9).
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
CACHE = ROOT / ".probe_pre.pkl"
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


def main() -> int:
    # --ls / --pad menimpa setelan supaya kalibrasi bisa diuji tanpa mengedit
    # config.py dulu.
    argv = list(sys.argv[1:])
    for flag, attr in (("--ls", "line_spacing"), ("--pad", "pad_ratio")):
        if flag in argv:
            i = argv.index(flag)
            setattr(SETTINGS, attr, float(argv[i + 1]))
            del argv[i:i + 2]
    want = {int(a) for a in argv} or {9}
    img = imgio.load_any(ROOT / "jepang_002.webp")
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with CACHE.open("rb") as f:
        regions = pickle.load(f)
    textmask.disjoin_overlapping_interiors(img, regions)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))

    print(f"pad_ratio={SETTINGS.pad_ratio} line_spacing={SETTINGS.line_spacing} "
          f"_ROW_COVER={typeset._ROW_COVER} min={SETTINGS.min_font_size}")
    for r in regions:
        text = str(texts.get(str(r.idx), "")).upper()
        if not text:
            continue
        _box, mask = typeset._region_box_mask(r)
        mh, mw = mask.shape[:2]
        pad = int(min(mh, mw) * SETTINGS.pad_ratio)
        area = int((mask > 0).sum())
        cx, cy = typeset._centroid(mask)
        head = (f"r{r.idx:<2} mask={mw}x{mh} isi={area/(mh*mw)*100:.0f}% pad={pad} "
                f"centroid=({cx},{cy})")
        if r.idx not in want:
            print(head)
            continue

        print("\n" + head)
        # Lebar bebas per baris pada tinggi glyph ukuran 14 — memperlihatkan
        # berapa banyak yang hilang ke pad dibanding lebar mask.
        ink_top, ink_bot = typeset._ink_band(fp, 14)
        for y in range(pad, mh - pad, max((mh - 2 * pad) // 12, 1)):
            lo, hi = 0.0, float(mw)
            for _ in range(9):
                mid = (lo + hi) / 2
                if typeset._row_free(mask, y + ink_top, y + ink_bot,
                                     cx - mid / 2, cx + mid / 2):
                    lo = mid
                else:
                    hi = mid
            print(f"    y={y:>4} bebas={lo:>6.1f} setelah_pad={max(lo-pad*2,0):>6.1f}"
                  f"  ({max(lo-pad*2,0)/mw*100:>3.0f}% dari lebar mask)")

        print("    ukuran  utuh                     ber-hyphen")
        for size in range(SETTINGS.min_font_size, 27):
            ok_p, lp, _ = typeset.layout(text, mask, size, fp, hyphenate=False)
            ok_h, lh_, _ = typeset.layout(text, mask, size, fp, hyphenate=True)
            print(f"    {size:>6}  {'OK ' if ok_p else 'gagal'} {len(lp):>2} baris"
                  f"          {'OK ' if ok_h else 'gagal'} {len(lh_):>2} baris")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
