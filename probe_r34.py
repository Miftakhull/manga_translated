#!/usr/bin/env python3
"""Kenapa r3 dan r4 jp_6 masih menolak wording referensi setelah lantai berskala.

_max_feasible() sengaja mengukur TANPA penggalan kata, sementara fit() yang
benar-benar merender BOLEH menggal. Probe ini memisahkan dua kemungkinan:
(a) wording-nya memang tidak muat pada ukuran apa pun, atau
(b) ia cuma tidak muat UTUH, dan fit() sebenarnya sanggup merendernya.
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

REF = {
    3: "CAN'T HELP IT ♥",
    4: "UH... I-IT'S EMBARRASSING...",
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
    if r.idx not in REF:
        continue
    t = REF[r.idx].upper()
    m = typeset._region_box_mask(r)[1]
    print(f"\n=== r{r.idx} interior {m.shape[1]}x{m.shape[0]}  {t!r}")
    print(f"{'size':>5} {'utuh':>5} {'hyph':>5} {'baris utuh':>28}")
    for s in range(12, 3, -1):
        okp, lp, _ = typeset.layout(t, m, s, fp, hyphenate=False)
        okh, lh, _ = typeset.layout(t, m, s, fp, hyphenate=True)
        print(f"{s:>5} {'ya' if okp else '-':>5} {'ya' if okh else '-':>5} "
              f"{(' | '.join(lp) if okp else '')[:28]:>28}")
    fs, fl, _fy, ov = typeset.fit(t, m, typeset.region_font_cap(m), fp)
    print(f"  fit() -> size={fs} luber={ov} baris={fl}")
    print(f"  _max_feasible (tanpa penggalan) = {typeset._max_feasible(t, m, fp)}")
    # Kata terpanjang wording ini vs kata terpanjang yang muat.
    longest = max(t.split(), key=len)
    print(f"  kata terpanjang wording = {longest!r} ({len(longest)} huruf), "
          f"max_word_len@{lo} = {typeset.max_word_len(m, lo, fp)}")
