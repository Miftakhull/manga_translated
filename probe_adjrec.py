#!/usr/bin/env python3
"""Apakah halaman uji balon-bertetangga punya kasus 'tercekik' sama sekali?

Setelah reclaim hanya melayani region yang ber-tanda-hubung atau luber, test
selftest yang lama tidak lagi sah: kalau KEDUA region halaman uji itu sudah rapi,
reclaim memang tidak boleh mengubah apa pun, dan 'berubah=0' adalah perilaku
benar — bukan cacat.

Yang dicari di sini kalimat uji yang membuat SATU region tercekik sementara
tetangganya tidak, supaya kontraknya bisa diuji seperti di jp_6. Dicetak: untuk
tiap kandidat panjang teks, fit() kedua region + apakah ada lebar yang disandera
(fill_mask di luar bubble_mask sendiri tapi di dalam interior tetangga).
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

CASES = [
    ("pendek", "OH, IS THIS THE MILKING CLUB ROOM, PREZ?",
     "YES, THE STUDENT COUNCIL RECORDS ARE HERE."),
    ("A satu kata mustahil", "PNEUMONOULTRAMICROSCOPICSILICOVOLCANOCONIOSIS "
     "OTORHINOLARYNGOLOGICAL", "YES."),
    ("A luber", " ".join(["MILKING CLUB RECORDS ARE UTTERLY CONFIDENTIAL"] * 12),
     "YES."),
]

fp = typeset.setup_fonts(verbose=False)
for name, ta, tb in CASES:
    clean, img, inner, regions = selftest.make_adjacent_bubbles_page()
    for r in regions:
        textmask.build_region_mask(img, r, None)
    textmask.disjoin_overlapping_interiors(img, regions)
    typeset.set_page_width(img.shape[1])
    H, W = img.shape[:2]
    regions[0].translation, regions[1].translation = ta, tb
    maps = [typeset._paste_mask(*typeset._region_box_mask(r), H, W) > 0
            for r in regions]
    fills = [typeset._paste_mask(r.fill_bbox, r.fill_mask, H, W) > 0
             if r.fill_mask is not None else np.zeros((H, W), bool)
             for r in regions]
    hostage = [int((fills[i] & ~maps[i] & maps[1 - i]).sum()) for i in (0, 1)]
    print(f"\n=== {name}  disandera={hostage}")
    for i, r in enumerate(regions):
        m = typeset._region_box_mask(r)[1]
        s, ls, _y, ov = typeset.fit(r.translation.upper(), m,
                                    typeset.region_font_cap(m), fp)
        hy = sum(1 for x in ls if x.endswith("-"))
        print(f"  r{i} interior {m.shape[1]}x{m.shape[0]} plafon="
              f"{typeset.region_font_cap(m)} -> size={s} hyph={hy} "
              f"luber={int(bool(ov))}  {' | '.join(ls)}")
    moved = typeset.reclaim_unused_interiors(img, regions)
    print(f"  reclaim mengubah {moved} region")
    if moved:
        for i, r in enumerate(regions):
            m = typeset._region_box_mask(r)[1]
            s, ls, _y, ov = typeset.fit(r.translation.upper(), m,
                                        typeset.region_font_cap(m), fp)
            hy = sum(1 for x in ls if x.endswith("-"))
            print(f"    r{i} sesudah: size={s} hyph={hy} luber={int(bool(ov))}"
                  f"  {' | '.join(ls)}")
