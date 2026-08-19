#!/usr/bin/env python3
"""Faktor tinggi-kapital Anime Ace: cap_height / ukuran_font.

Semua kesimpulan kalibrasi bergantung angka ini (dipakai untuk menerjemahkan
cap_height terukur di CONTOH/2.webp menjadi ukuran font yang harus kita minta),
jadi diukur, bukan diasumsikan 0.72.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / ".stage"
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

import cv2      # noqa: E402
import typeset  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    print(f"font = {Path(fp).name}")
    print(f"  {'size':>4} {'ink_band':>12} {'cap_ukur':>9} {'cap/size':>9} "
          f"{'lh':>4} {'lebar/huruf':>11}")
    rs = []
    for size in (11, 13, 15, 17, 19, 21, 24, 28, 32):
        f = typeset._font(fp, size)
        top, bot = typeset._ink_band(fp, size)
        im = Image.new("L", (size * 14, size * 4), 255)
        ImageDraw.Draw(im).text((size, size), "HAMBURG", font=f, fill=0)
        a = np.asarray(im, np.uint8)
        ink = (a < 128).astype(np.uint8)
        n, _l, st, _ = cv2.connectedComponentsWithStats(ink, 8)
        hs = [st[i][3] for i in range(1, n) if st[i][4] >= 4]
        cap = float(np.median(hs)) if hs else 0.0
        lh = typeset._line_height(f)
        adv = typeset._measure("HAMBURG", f) / 7
        rs.append(cap / size)
        print(f"  {size:>4} {f'{top}..{bot}':>12} {cap:>9.1f} {cap/size:>9.3f} "
              f"{lh:>4} {adv:>11.2f}")
    r = float(np.median(rs))
    print(f"\ncap/size median = {r:.3f}")
    # cap referensi 13..27 px pada halaman 1812 -> ukuran font di halaman 1577.
    sc = 1577 / 1812
    for cap in (13, 14, 16, 19, 22, 27):
        print(f"  cap_ref {cap:>2} px  ->  cap_kita {cap*sc:>5.1f} px  "
              f"->  ukuran font {cap*sc/r:>5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
