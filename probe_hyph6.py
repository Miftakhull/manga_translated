#!/usr/bin/env python3
"""Kenapa r3 jp_6 dirender 'NO WON-DER' padahal balonnya masih longgar.

Dua kemungkinan yang dipisahkan di sini:
(a) 'WONDER' memang tidak muat satu baris di lebar interior r3 pada ukuran itu,
(b) jalur DARURAT fit() (dipakai ketika tidak ada yang muat di >= min_font())
    memanggil _search dengan hyphenate=True saja, jadi versi UTUH tidak pernah
    diuji di ukuran bawah — hyphen muncul bukan karena perlu, tapi karena tidak
    ada pesaing.
Offline, nol token.
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
for s in sorted(NBSRC.glob("*.py")):
    body = _MAGIC.sub("", s.read_text(encoding="utf-8"), count=1)
    d = STAGE / s.name
    if not d.exists() or d.read_text(encoding="utf-8") != body:
        d.write_text(body, encoding="utf-8")
sys.path.insert(0, str(STAGE))

import detect, imgio, textmask, typeset            # noqa: E402,E401

TXT = {
    3: "NO WONDER ♥!",
    0: "SO IN THE END,",
    4: "U-UM... BEING DESCRIBED LIKE THAT IS EMBARRASSING...",
    2: "WELL, OF COURSE—NO ONE'S EVER SEEN SUCH A DIRTY, SEXY PERSON BEFORE.",
    1: "EVERYONE GOT SO EXCITED THEY MOSTLY CAME BY THEMSELVES ♥!",
}

fp = typeset.setup_fonts(verbose=False)
img = imgio.load_any(ROOT / "hasilnew/jp_6.JPG")
typeset.set_page_width(img.shape[1])
regions, _ = detect.detect(img)
soft = textmask.ctd_soft_mask(img)
for r in regions:
    textmask.build_region_mask(img, r, soft)
textmask.partition_shared_interiors(img, regions)
textmask.disjoin_overlapping_interiors(img, regions)

lo = typeset.min_font()
print(f"min_font()={lo} darurat={typeset.emergency_floor()}")
for r in regions:
    if r.idx not in TXT:
        continue
    t = TXT[r.idx].upper()
    m = typeset._region_box_mask(r)[1]
    cap = typeset.region_font_cap(m)
    fs, fl, _y, ov = typeset.fit(t, m, cap, fp)
    print(f"\n=== r{r.idx} interior {m.shape[1]}x{m.shape[0]} cap={cap}  {t[:44]!r}")
    print(f"  fit() -> size={fs} luber={ov} baris={fl}")
    print(f"  {'size':>5} {'utuh':>5} {'hyph':>5}  baris utuh")
    for s in range(cap, typeset.emergency_floor() - 1, -1):
        okp, lp, _ = typeset.layout(t, m, s, fp, hyphenate=False)
        okh, _lh, _ = typeset.layout(t, m, s, fp, hyphenate=True)
        print(f"  {s:>5} {'ya' if okp else '-':>5} {'ya' if okh else '-':>5}  "
              f"{' | '.join(lp) if okp else ''}")
