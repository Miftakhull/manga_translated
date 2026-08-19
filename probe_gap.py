#!/usr/bin/env python3
"""Jarak tinta ke garis balon di jalur PRODUKSI (fit + line_axis), per region.

Gerbang wajib untuk setiap perubahan tata letak: melonggarkan apa pun demi
keseimbangan tidak boleh menggerus jarak tinta ke garis balon. Sebelum sumbu
blok dipasang angkanya median 3 px, min 0 px (probe_margin.py varian
"sekarang").

HATI-HATI membaca angka nol di sini. Yang diukur edge_gap() adalah KOTAK baris —
setinggi band tinta, selebar advance font — bukan glyph-nya. Kotak selalu lebih
longgar dari isinya: 'A' tidak mengisi sudut kotaknya, dan advance menyertakan
side bearing kanan yang kosong. Jadi tepi=0 di sini BELUM berarti tinta
menyentuh garis balon. Setelah sumbu blok dipasang probe ini melaporkan
"nol=1/13 [10]" dan itu terbaca seperti regresi, padahal diukur pada tinta yang
benar-benar dirender (probe_inkcmp.py, jalur lama V0 vs jalur produksi):

  jalur   tinta di luar interior   jarak tinta->luar
  lama    5 px di r12              min 0  median 5
  baru    0 px                     min 1  median 7

Jadi yang mengikat adalah probe_inkgap.py / probe_inkcmp.py; probe ini tetap
berguna sebagai gerbang cepat yang konservatif, bukan sebagai putusan akhir.

    TEXTS=probe_opus5_clean.json python probe_gap.py
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
TEXTS = ROOT / os.environ.get("TEXTS", "probe_opus5_clean.json")
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

import typeset   # noqa: E402
from probe_margin import edge_gap  # noqa: E402


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    print(f"{'idx':>3} {'size':>4} {'nb':>3} {'cx':>4} {'dx':>4} {'tepi':>5}  baris")
    gaps, nol = [], []
    for r in regions:
        t = str(texts.get(str(r.idx), "")).upper()
        if not t:
            continue
        m = typeset._region_box_mask(r)[1]
        size, lines, sy, _ov = typeset.fit(t, m, typeset.region_font_cap(m), fp)
        if not lines:
            continue
        font = typeset._font(fp, size)
        lh = typeset._line_height(font)
        it, ib = typeset._ink_band(fp, size)
        cx = typeset.line_axis(m, lines, sy, size, fp)
        cx0 = typeset._centroid(m)[0]
        gap = edge_gap(m, lines, sy, lh, it, ib, cx, font)
        gaps.append(gap)
        if gap == 0:
            nol.append(r.idx)
        print(f"{r.idx:>3} {size:>4} {len(lines):>3} {cx:>4} {cx - cx0:>+4} "
              f"{gap:>5}  {' / '.join(lines)[:40]}")
    print(f"\njarak_tepi: min={min(gaps)} median={np.median(gaps):.0f} "
          f"nol={len(nol)}/{len(gaps)} {nol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
