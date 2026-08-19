#!/usr/bin/env python3
"""Cari kalimat uji yang tanda hubungnya BENAR-BENAR disebabkan lebar tersandera.

probe_adjrec.py menunjukkan kenapa kandidat pertama gagal jadi test: satu kata
'PNEUMONOULTRA...' mustahil di lebar berapa pun, jadi menolak klaimnya memang
perilaku benar — bukan cacat reclaim. Yang dibutuhkan test adalah kasus seperti
r3 jp_6: tanda hubungnya hilang begitu lebar sanderaan dikembalikan.

Kriteria kandidat, diuji langsung, bukan ditebak:
    fit(mask_setelah_disjoin)      -> ADA tanda hubung
    fit(mask | sanderaan)          -> TANPA tanda hubung, ukuran tidak turun
Teks r1 dijaga pendek supaya tidak ikut tercekik (biar jelas siapa pelepas).
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

WORDS = ["CONFIDENTIAL", "EXTRAORDINARY", "UNCOMFORTABLE", "MISUNDERSTANDING",
         "RESPONSIBILITY", "ACKNOWLEDGEMENT", "CIRCUMSTANCES", "REGRETTABLY",
         "INTERJECTION", "UNCONDITIONALLY", "PRESIDENT", "COMMITTEE"]
TAILS = ["", " PREZ?", " I THINK.", " AGAIN, PREZ?",
         " AND THE RECORDS TOO.", " BUT THE RECORDS ARE HERE, PREZ."]

fp = typeset.setup_fonts(verbose=False)
clean, img, inner, base = selftest.make_adjacent_bubbles_page()
for r in base:
    textmask.build_region_mask(img, r, None)
textmask.disjoin_overlapping_interiors(img, base)
typeset.set_page_width(img.shape[1])
H, W = img.shape[:2]

maps = [typeset._paste_mask(*typeset._region_box_mask(r), H, W) > 0 for r in base]
fills = [typeset._paste_mask(r.fill_bbox, r.fill_mask, H, W) > 0
         if r.fill_mask is not None else np.zeros((H, W), bool) for r in base]
host = [fills[i] & ~maps[i] & maps[1 - i] for i in (0, 1)]
print(f"sanderaan: r0={int(host[0].sum())} px  r1={int(host[1].sum())} px")


def fit_map(mp: np.ndarray, text: str):
    ys, xs = np.nonzero(mp)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    m = np.where(mp[box[1]:box[3], box[0]:box[2]], 255, 0).astype(np.uint8)
    s, ls, _y, ov = typeset.fit(text.upper(), m, typeset.region_font_cap(m), fp)
    return s, sum(1 for x in ls if x.endswith("-")), int(bool(ov)), ls


found = 0
for w in WORDS:
    for tail in TAILS:
        for rep in (1, 2, 3):
            t = (" ".join([w] * rep) + tail).strip()
            s0, h0, o0, l0 = fit_map(maps[0], t)
            if h0 == 0 and o0 == 0:
                continue
            s1, h1, o1, l1 = fit_map(maps[0] | host[0], t)
            if h1 < h0 and s1 >= s0 and o1 == 0:
                found += 1
                print(f"\n[KANDIDAT] {t!r}")
                print(f"  disjoin : size={s0} hyph={h0} luber={o0}  {' | '.join(l0)}")
                print(f"  +sandera: size={s1} hyph={h1} luber={o1}  {' | '.join(l1)}")
                if found >= 6:
                    raise SystemExit(0)
print(f"\ntotal kandidat = {found}")
