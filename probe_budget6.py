#!/usr/bin/env python3
"""Kalibrasi anggaran: berapa karakter yang muat di tiap balon jp_6 pada
berbagai ukuran font, dibanding panjang wording referensi hasilnew/6.JPG.

Tujuannya satu angka: sekecil apa ukuran font harus diizinkan supaya anggaran
yang dikirim ke model sepanjang wording referensi. Offline, nol token.
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

# Wording referensi hasilnew/6.JPG, dibaca dari gambarnya, urut kanan->kiri
# mengikuti urutan baca yang sama dengan pipeline.
REF = [
    "WELL",
    "IN THE END EVERYONE GOT SO WORKED UP THEY CAME BY THEMSELVES ♥",
    "WELL, NO WONDER",
    "WE'VE NEVER SEEN ANYONE SO SLUTTY ♥",
    "UM... BEING DESCRIBED THAT WAY IS EMBARRASSING...",
    "PLEASE DON'T TEASE ME SO MUCH...",
    "EH-? BUT I'M PRAISING YOU FROM THE HEART!",
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

print(f"min_font={SETTINGS.min_font_size} "
      f"REF_CAP_PER_MIN={typeset._REF_CAP_PER_MIN} CAP_PER_SIZE={typeset._CAP_PER_SIZE}")
print(f"{'r':>2} {'interior':>10} {'cap':>4} "
      + " ".join(f"{'@'+str(s):>6}" for s in (18, 16, 14, 13, 12, 11))
      + "   ref_len  feas(ref)")
for r in regions:
    mask = typeset._region_box_mask(r)[1]
    cap = typeset.region_font_cap(mask)
    row = [f"{typeset.char_budget(mask, s, fp):>6}" for s in (18, 16, 14, 13, 12, 11)]
    ref = REF[r.idx] if r.idx < len(REF) else ""
    feas = typeset._max_feasible(ref.upper(), mask, fp) if ref else 0
    print(f"{r.idx:>2} {str(mask.shape[1])+'x'+str(mask.shape[0]):>10} {cap:>4} "
          + " ".join(row) + f"   {len(ref):>7}  {feas:>8}")
