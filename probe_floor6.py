#!/usr/bin/env python3
"""Anggaran balon jp_6 pada lantai font yang berbeda vs panjang wording referensi.

Pertanyaan yang dijawab: kalau lantai ukuran font diturunkan (halaman ini cuma
698 px lebar, sementara lantai 11 px dikalibrasi di halaman 1134 px), apakah
anggaran yang dikirim ke model jadi sepanjang wording referensi hasilnew/6.JPG?
Offline, nol token.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

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
from config import SETTINGS                        # noqa: E402

REF = [
    "WELL",
    "SO IN THE END, THE WHOLE CLASS GOT SO EXCITED THEY CAME BY THEMSELVES ♥",
    "WELL, NO WONDER-WE'VE NEVER SEEN ANYONE SO SLUTTY",
    "CAN'T HELP IT ♥",
    "UH... I-IT'S EMBARRASSING...",
    "PLEASE DON'T TEASE ME SO MUCH...",
    "EH~? BUT I'M PRAISING YOU FROM THE HEART!",
    "AH!",
]

fp = typeset.setup_fonts(verbose=False)
img = imgio.load_any(ROOT / "hasilnew/jp_6.JPG")
regions, _ = detect.detect(img)
soft_mask = textmask.ctd_soft_mask(img)
for r in regions:
    textmask.build_region_mask(img, r, soft_mask)
textmask.partition_shared_interiors(img, regions)
textmask.disjoin_overlapping_interiors(img, regions)

masks = {r.idx: typeset._region_box_mask(r)[1] for r in regions}

print(f"halaman {img.shape[1]}x{img.shape[0]}")
for floor in (11, 9, 8, 7, 6):
    SETTINGS.min_font_size = floor
    print(f"\n=== lantai font {floor} ===")
    print(f"{'r':>2} {'interior':>9} {'cap':>4} {'soft':>5} {'word':>5} "
          f"{'hard':>5} {'ref_len':>7} {'feas':>5} {'muat':>5}")
    for r in regions:
        m = masks[r.idx]
        cap = typeset.region_font_cap(m)
        soft = typeset.char_budget(m, cap, fp)
        word = typeset.max_word_len(m, cap, fp)
        hard = typeset.char_budget(m, floor, fp)
        ref = REF[r.idx].upper() if r.idx < len(REF) else ""
        feas = typeset._max_feasible(ref, m, fp) if ref else 0
        print(f"{r.idx:>2} {str(m.shape[1])+'x'+str(m.shape[0]):>9} {cap:>4} "
              f"{soft:>5} {word:>5} {hard:>5} {len(ref):>7} {feas:>5} "
              f"{'ya' if feas >= floor else 'TIDAK':>5}")
