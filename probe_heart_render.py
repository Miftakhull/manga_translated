"""Bukti visual: baris 'I LOVE YOU ♥' lewat jalur gambar typeset sungguhan.

Bukan lewat PIL langsung — lewat typeset._draw_line(), supaya yang diuji adalah
jalur yang benar-benar dipakai render_region(). Cetak ASCII 2 warna: kalau
bentuknya hati, cacat hasilnew/6.JPG beres; kalau masih huruf, belum.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(ROOT / ".stage"))
sys.path.insert(0, str(ROOT / ".stage"))

from PIL import Image, ImageDraw  # noqa: E402

import typeset  # noqa: E402

typeset.setup_fonts(verbose=False)
SIZE = 26
font = typeset._font(typeset.FONT_USED, SIZE)
cmap = typeset._cmap(typeset.FONT_USED)

for line in ("LOVE ♥", "OK ☆", "LA ♪"):
    w = int(typeset._line_width(line, font, cmap, SIZE)) + 6
    im = Image.new("L", (w, 34), 255)
    typeset._draw_line(ImageDraw.Draw(im), (2, 2), line, font, (0,), cmap, SIZE)
    px = im.load()
    print(f"--- {line!r}  lebar_ukur={typeset._measure(line, font):.1f} "
          f"lebar_gambar={typeset._line_width(line, font, cmap, SIZE):.1f} ---")
    for y in range(34):
        row = "".join("#" if px[x, y] < 128 else "." for x in range(w))
        if row.strip("."):
            print(row)
    print()
