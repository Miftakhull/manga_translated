"""Ukur jalur WARNA HURUF di balon gelap, tanpa model dan tanpa halaman nyata.

Aturan V mengarahkan balon HITAM ke jalur erase+typeset untuk pertama kalinya.
Keputusan warnanya satu baris, typeset.py:1486:

    fill = (255,255,255) if _bg_luminance(img, region) < 128 else (0,0,0)

dan tidak ada SATU pun uji di selftest.py yang memanggil _bg_luminance —
digrep, nol pemanggil. Jadi jalur ini tidak punya penjaga regresi sama sekali.

Probe ini membangun halaman sintetis berisi DUA balon: satu HITAM berhuruf
Jepang PUTIH (bentuk r8), satu PUTIH berhuruf Jepang HITAM (kontrol). Keduanya
lewat build_region_mask(img, r, None) — tanpa ONNX, jadi bisa masuk
verify_local.py nanti. Tujuannya mengambil ANGKA-nya dulu supaya ambang
penjaganya terukur; ambang median 160 yang saya karang di _h5type.py gagal
justru karena dikarang (anti-alias glyph 11 px menarik median ke tengah).

Yang diukur:
  1. _bg_luminance tiap balon -> arah keputusan warnanya
  2. build_region_mask menemukan tinta PUTIH di balon hitam (kalau tidak,
     erase tidak akan menghapusnya — cacat lain di jalur yang sama)
  3. sebaran kelabu tinta yang TERGAMBAR di tiap balon: median, p90, maks
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / ".stage"
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))
STAGE.mkdir(exist_ok=True)
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
for p in sorted((ROOT / "_nbsrc").glob("*.py")):
    (STAGE / p.name).write_text(
        _MAGIC.sub("", p.read_text(encoding="utf-8"), count=1), encoding="utf-8")
sys.path.insert(0, str(STAGE))

import textmask                       # noqa: E402
import typeset                        # noqa: E402
from config import Region             # noqa: E402
import selftest as st                 # noqa: E402

W, H = 620, 430
_STROKE = 3


def make_dark_bubble_page():
    """(clean, img, inner, regions) — balon HITAM + balon PUTIH bersebelahan.

    Balon hitam meniru r8 hasilnew5: interior nyaris 0, huruf Jepang PUTIH.
    Balon putih jadi KONTROL: kalau keputusan warna tertukar arah, balon ini
    yang memergokinya (hurufnya harus tetap gelap).
    """
    page = Image.new("RGB", (W, H), (250, 249, 247))
    d = ImageDraw.Draw(page)
    for y in range(0, H, 9):
        d.line([(0, y), (W, y)], fill=(216, 216, 216), width=1)
    arr = np.asarray(page, np.uint8).copy()

    bubs = ((150, 215, 96, 132), (450, 215, 96, 132))
    fills, inners = [], []
    for cx, cy, ax, ay in bubs:
        f = np.zeros((H, W), np.uint8)
        cv2.ellipse(f, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        k = 2 * _STROKE + 1
        fills.append(f)
        inners.append(cv2.erode(f, np.ones((k, k), np.uint8)))
    # kiri HITAM pekat (r8: _bg_luminance terukur 1.0), kanan PUTIH
    arr[inners[0] > 0] = (6, 6, 7)
    arr[inners[1] > 0] = (255, 255, 255)
    for f, inn in zip(fills, inners):
        arr[(f > 0) & (inn == 0)] = (0, 0, 0)

    clean = arr.copy()
    page = Image.fromarray(arr)
    d = ImageDraw.Draw(page)
    f = st._jp_font(30)
    # Huruf PUTIH di balon hitam, HITAM di balon putih — seperti manga aslinya.
    for i, ((cx, cy, _, _), rows) in enumerate(zip(bubs, (("ヒ", "ッ"), ("そう", "だね")))):
        col = (255, 255, 255) if i == 0 else (0, 0, 0)
        for j, row in enumerate(rows):
            d.text((cx, cy - 22 + j * 44), row, font=f, fill=col, anchor="mm")

    inner = np.maximum(inners[0], inners[1])
    regions = []
    for i, (cx, cy, ax, ay) in enumerate(bubs):
        ys, xs = np.nonzero(fills[i])
        regions.append(Region(
            idx=i, bbox=(cx - 40, cy - 52, cx + 40, cy + 52),
            det_class="text_bubble",
            bubble_bbox=(int(xs.min()), int(ys.min()),
                         int(xs.max()) + 1, int(ys.max()) + 1)))
    return clean, np.asarray(page, np.uint8), inner, regions


clean, img, inner, regions = make_dark_bubble_page()
typeset.set_page_width(W)
typeset.setup_fonts(verbose=False)
if not typeset.FONT_USED:
    sys.exit("FONT_USED kosong -> render tidak menggambar, angkanya tak berarti")

for r in regions:
    textmask.build_region_mask(img, r, None)

print(f"halaman sintetis {W}x{H}  font={Path(typeset.FONT_USED).name}\n")
print("1) arah keputusan warna (typeset.py:1486)")
lums = []
for r in regions:
    lum = typeset._bg_luminance(clean, r)
    lums.append(lum)
    print(f"   r{r.idx}: _bg_luminance={lum:6.1f} -> "
          f"fill={'PUTIH' if lum < 128 else 'HITAM'}")

print("\n2) build_region_mask menemukan tinta di kedua polaritas")
for r in regions:
    im = r.ink_mask
    n = int((im > 0).sum()) if im is not None else -1
    print(f"   r{r.idx}: ink_mask={n} px  bentuk={None if im is None else im.shape}")

print("\n3) sebaran kelabu tinta yang TERGAMBAR")
regions[0].translation = "EEP!?"
regions[1].translation = "YEAH, RIGHT."
for r in regions:
    out = typeset.render_page(clean.copy(), [r])
    ink = st._ink_of(out, clean)
    if not ink.any():
        print(f"   r{r.idx}: TIDAK MENGGAMBAR APA PUN "
              f"(font={r.final_font_size} overflow={r.overflowed})")
        continue
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[ink]
    print(f"   r{r.idx}: latar={lums[r.idx]:5.1f} n={ink.sum():5d} "
          f"min={g.min():3d} med={np.median(g):5.1f} "
          f"p90={np.percentile(g, 90):5.1f} maks={g.max():3d} "
          f"font={r.final_font_size}")
    cv2.imwrite(str(ROOT / f"_h5dark_r{r.idx}.png"),
                cv2.resize(out, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST))
print("\n -> _h5dark_r0.png / _h5dark_r1.png")
