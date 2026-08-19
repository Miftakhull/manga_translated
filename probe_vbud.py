#!/usr/bin/env python3
"""Kenapa 'validasi anggaran' gagal: apa yang berubah pada teks mustahil.

Offline, tanpa detector (pakai halaman sintetis selftest), jadi cepat.
Yang dicetak: word_hard, lalu hasil fit() untuk "A"*N pada condense 0.85 dan
1.00 — supaya jelas apakah kegagalan datang dari condense atau dari hal lain.
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

import detect, selftest, textmask, typeset            # noqa: E402,E401
from config import SETTINGS                           # noqa: E402

fp = typeset.setup_fonts(verbose=False)
_clean, _img, _inner, dbl = selftest.make_double_bubble_page()
detect._partition_shared_bubbles(dbl)
soft = textmask.ctd_soft_mask(_img)
for r in dbl:
    textmask.build_region_mask(_img, r, soft)
textmask.partition_shared_interiors(_img, dbl)
r = dbl[0]
mask = typeset._region_box_mask(r)[1]
typeset.set_page_width(_img.shape[1])

print(f"halaman {_img.shape[1]}x{_img.shape[0]}  mask {mask.shape[1]}x{mask.shape[0]}  "
      f"min_font()={typeset.min_font()}  darurat={typeset.emergency_floor()}")

for cnd in (0.85, 1.00):
    SETTINGS.condense = cnd
    cap = typeset.region_font_cap(mask)
    wh = typeset.max_word_len(mask, typeset.min_font(), fp)
    fl = typeset.emergency_floor()
    adv = typeset._line_width("A", typeset._font(fp, fl), typeset._cmap(fp), fl)
    nimp = int(mask.shape[1] / max(adv, 1e-6)) + 8
    print(f"\ncondense={cnd}  cap={cap}  word_hard={wh}  "
          f"adv('A'@{fl})={adv:.2f}  N_geometri={nimp}")
    for n in (wh + 40, nimp, 90, 140, 200, 300):
        txt = "A" * n
        size, lines, _y, over = typeset.fit(txt, mask, cap, fp)
        ok, _s = typeset.renders_ok(txt, mask, fp)
        print(f"  N={n:>4}  size={size:>3}  baris={len(lines):>3}  "
              f"luber={int(over)}  renders_ok={ok}")
