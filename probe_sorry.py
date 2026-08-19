#!/usr/bin/env python3
"""Kenapa 'SORRY.' (r6) tampak jatuh ke bawah padahal probe_center bilang +0.

probe_center mengukur ketimpangan pakai block_slack, yang menghitung ruang bebas
di dalam MASK INTERIOR region. Kalau mask itu sendiri tidak mencakup seluruh
rongga balon yang terlihat mata — misalnya karena dipotong tetangga di
disjoin_overlapping_interiors — maka teks bisa "terpusat" menurut mask dan tetap
terlihat melorot di dalam balon.

Yang dicetak: kotak balon, kotak interior efektif, dan untuk kolom tempat teks
berada, di mana rongga balon yang SEBENARNYA mulai dan berakhir versus rongga
menurut mask region. Selisih keduanya = besar potongan yang tidak kelihatan.

    IDX=6 python probe_sorry.py
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / ".stage"
IDX = int(os.environ.get("IDX", "6"))
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

import cv2       # noqa: E402
import imgio     # noqa: E402
import textmask  # noqa: E402
import typeset   # noqa: E402


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    typeset.setup_fonts(verbose=False)
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    r = next(x for x in regions if x.idx == IDX)
    box, m = typeset._region_box_mask(r)
    mh, mw = m.shape[:2]
    cx, cy = typeset._centroid(m)
    print(f"r{IDX}  bbox={r.bbox}  bubble_bbox={r.bubble_bbox}")
    print(f"     kotak mask={box}  ukuran={mw}x{mh}  centroid=({cx},{cy})")
    print(f"     piksel interior={int((m > 0).sum())} "
          f"({100.0 * (m > 0).mean():.1f}% dari kotak)")

    # Rongga balon SEBENARNYA di kolom cx: dari citra, bukan dari mask region.
    # Otsu + ambang gelap yang sama dipakai textmask untuk garis balon.
    gx = box[0] + cx
    col = img[:, gx].mean(1)
    dark = col < textmask._LINE_DARK
    y0b, y1b = r.bubble_bbox[1], r.bubble_bbox[3]
    runs = []
    y = y0b
    while y < y1b:
        if dark[y]:
            y += 1
            continue
        s = y
        while y < y1b and not dark[y]:
            y += 1
        runs.append((s, y - 1, y - s))
    runs.sort(key=lambda t: -t[2])
    print(f"\nkolom x={gx} di dalam bubble_bbox y={y0b}..{y1b}:")
    for s, e, n in runs[:3]:
        print(f"     rongga terang y={s}..{e}  ({n} px)  tengahnya y={(s + e) // 2}")

    # Rongga menurut mask region, kolom yang sama.
    mcol = m[:, cx] > 0
    ys = np.flatnonzero(mcol)
    if ys.size:
        a, b = int(ys[0]) + box[1], int(ys[-1]) + box[1]
        print(f"\nmask region kolom sama: y={a}..{b} ({b - a + 1} px) "
              f"tengahnya y={(a + b) // 2}")

    # Di mana teks berada sekarang.
    ok, lines, y_top = typeset.layout("SORRY.", m, r.final_font_size or 11,
                                      typeset.FONT_USED)
    font = typeset._font(typeset.FONT_USED, r.final_font_size or 11)
    it, ib = typeset._ink_band(typeset.FONT_USED, r.final_font_size or 11)
    print(f"\nlayout: ok={ok} lines={lines} y_top={y_top} "
          f"tinta y={box[1] + y_top + it}..{box[1] + y_top + ib} "
          f"tengah tinta y={box[1] + y_top + (it + ib) // 2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
