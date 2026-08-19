#!/usr/bin/env python3
"""Apakah bubble_bbox r6 memotong balonnya di sisi kanan?

Interior r6 menyentuh kolom terakhir kotak (x=67 dari mw=68) sepanjang
y=60..115, sementara sisi kirinya berhenti di x=13-15. Kalau balon aslinya
memang berlanjut ke kanan melewati bubble_bbox, maka centroid horizontalnya
salah karena separuh rongga dipotong — dan itu penyebab lain dari 'SORRY.'
yang melorot, bukan cuma metrik keseimbangan.

Dicetak: profil gelap/terang di beberapa baris melintasi kotak balon yang
DILEBARKAN 40 px ke kiri-kanan, plus crop PNG untuk dilihat.
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

PAD = 40


def main() -> int:
    typeset.setup_fonts(verbose=False)
    img = imgio.load_any(ROOT / "jepang_002.webp")
    h, w = img.shape[:2]
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    r = next(x for x in regions if x.idx == IDX)
    bx0, by0, bx1, by1 = r.bubble_bbox
    gray = img.mean(2)
    print(f"r{IDX} bubble_bbox=({bx0},{by0},{bx1},{by1})  ambang gelap="
          f"{textmask._LINE_DARK}")
    x0, x1 = max(bx0 - PAD, 0), min(bx1 + PAD, w)
    print(f"jendela x={x0}..{x1}; '|' menandai batas bubble_bbox\n")
    print("   y  profil ('#'=gelap '.'=terang)")
    for y in range(by0, by1, 8):
        row = gray[y, x0:x1] < textmask._LINE_DARK
        s = "".join("#" if v else "." for v in row)
        a, b = bx0 - x0, bx1 - x0 - 1
        s = s[:a] + "|" + s[a + 1:b] + "|" + s[b + 1:]
        print(f"{y:>4} {s}")

    crop = img[max(by0 - PAD, 0):min(by1 + PAD, h), x0:x1]
    out = ROOT / "_cmp" / f"zz_wide_r{IDX}.png"
    cv2.imwrite(str(out), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print(f"\n-> {out.name} ({crop.shape[1]}x{crop.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
