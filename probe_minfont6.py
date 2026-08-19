#!/usr/bin/env python3
"""Verifikasi lantai font berskala resolusi (typeset.min_font()) pada jp_6.

Pertanyaan yang dijawab, offline dan nol token: setelah lantai ikut lebar
halaman, apakah anggaran yang dikirim ke model (soft/hard) sepanjang wording
typeset referensi hasilnew/6.JPG, dan apakah _max_feasible() atas wording itu
berhenti mengembalikan 0?
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
from config import SETTINGS                        # noqa: E402

# Wording typeset referensi hasilnew/6.JPG, urut baca sama seperti pipeline.
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
h, w = img.shape[:2]
regions, _ = detect.detect(img)
soft_mask = textmask.ctd_soft_mask(img)
for r in regions:
    textmask.build_region_mask(img, r, soft_mask)
textmask.partition_shared_interiors(img, regions)
textmask.disjoin_overlapping_interiors(img, regions)

print(f"halaman {w}x{h}  min_font_size(kalibrasi)={SETTINGS.min_font_size} "
      f"ref_width={SETTINGS.min_font_ref_width} abs={SETTINGS.min_font_abs}")
print(f"min_font() sebelum set_page_width = {typeset.min_font()}")
typeset.set_page_width(w)
print(f"min_font() sesudah set_page_width  = {typeset.min_font()} "
      f"darurat={typeset.emergency_floor()}")
print(f"cek skala: 1134px -> {typeset.min_font(1134)}  1600px -> {typeset.min_font(1600)}  "
      f"400px -> {typeset.min_font(400)}")

print(f"\n{'r':>2} {'interior':>9} {'cap':>4} {'prefer':>7} {'maxch':>6} {'word':>5} "
      f"{'ref_len':>7} {'feas':>5} {'fit':>4} {'lulus':>6}")
lo = typeset.min_font()
bad = 0
for r in regions:
    m = typeset._region_box_mask(r)[1]
    b = typeset.region_budget(r, fp)
    ref = REF[r.idx].upper() if r.idx < len(REF) else ""
    feas = typeset._max_feasible(ref, m, fp) if ref else 0
    ok, size = typeset.renders_ok(ref, m, fp) if ref else (True, 0)
    if ref and not ok:
        bad += 1
    print(f"{r.idx:>2} {str(m.shape[1])+'x'+str(m.shape[0]):>9} {b['cap']:>4} "
          f"{b['soft']:>7} {b['hard']:>6} {b['word_hard']:>5} {len(ref):>7} "
          f"{feas:>5} {size:>4} {'ya' if ok else 'TIDAK':>6}")
print(f"\nbalon yang MENOLAK wording referensi: {bad}/{len(REF)}")
