#!/usr/bin/env python3
"""Kata mana yang menolak muat di balon sempit (region dengan kelayakan 0).

Untuk tiap region gagal: lebar tersedia di baris terlebar, lalu tiap kata diukur
utuh dan pada setiap titik penggalan pyphen — sehingga terlihat apakah yang
menjepit itu lebar kata, batas _MIN_HEAD/_MIN_TAIL, atau titik penggalan yang
terlalu kasar.
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


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with PRE.open("rb") as f:
        regions = pickle.load(f)
    textmask.disjoin_overlapping_interiors(img, regions)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    SETTINGS.line_spacing = float(os.environ.get("LS", 0.95))
    SETTINGS.pad_ratio = float(os.environ.get("PAD", 0.06))
    want = {int(a) for a in sys.argv[1:]} or {6, 10, 12}
    size = SETTINGS.min_font_size
    font = typeset._font(fp, size)
    print(f"ukuran minimum={size} pad_ratio={SETTINGS.pad_ratio} "
          f"line_spacing={SETTINGS.line_spacing}  "
          f"_MIN_HEAD={typeset._MIN_HEAD} _MIN_TAIL={typeset._MIN_TAIL}")

    for r in regions:
        if r.idx not in want:
            continue
        m = typeset._region_box_mask(r)[1]
        mh, mw = m.shape[:2]
        pad = int(min(mh, mw) * SETTINGS.pad_ratio)
        cx, _cy = typeset._centroid(m)
        ink_top, ink_bot = typeset._ink_band(fp, size)
        widths = []
        for y in range(pad, mh - pad - (ink_bot - ink_top)):
            lo, hi = 0.0, float(mw)
            for _ in range(9):
                mid = (lo + hi) / 2
                if typeset._row_free(m, y + ink_top, y + ink_bot,
                                     cx - mid / 2, cx + mid / 2):
                    lo = mid
                else:
                    hi = mid
            widths.append(max(lo - pad * 2, 0.0))
        wmax = max(widths) if widths else 0.0
        text = str(texts.get(str(r.idx), "")).upper()
        print(f"\nr{r.idx}  mask={mw}x{mh} pad={pad} lebar_terbaik={wmax:.1f} "
              f"(baris butuh >= {size*0.9:.1f} supaya tidak ditolak)")
        print(f"    teks: {text!r}")
        for wd in text.split():
            full = font.getlength(wd)
            pts = sorted(typeset._break_points(wd))
            frag = []
            for n in pts:
                head = f"{wd[:n]}-"
                letters_head = sum(c.isalpha() for c in wd[:n])
                letters_tail = sum(c.isalpha() for c in wd[n:])
                ok = (letters_head >= typeset._MIN_HEAD
                      and letters_tail >= typeset._MIN_TAIL)
                frag.append(f"{head}={font.getlength(head):.0f}"
                            f"{'' if ok else '(ditolak MIN_HEAD/TAIL)'}")
            verdict = "MUAT" if full <= wmax else "terlalu lebar"
            print(f"    {wd:<12} utuh={full:>5.0f} {verdict:<14} "
                  f"penggalan: {', '.join(frag) if frag else '(tidak ada titik)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
