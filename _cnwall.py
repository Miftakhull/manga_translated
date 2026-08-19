"""Lihat interior kandidat R, dan uji apakah garis balon perlu jadi DINDING.

_cnband.py memutuskan aturan polaritas: kelas interior = kelas yang memuat
cincin latar tinta region ("cincin"). Itu memperbaiki tiga region cacat
(tinta 0.05/0.08 -> 1.00/0.98) dan TIDAK mengubah satu pun dari 12 region
balon putih bersih di jp_6 + jp_13 (coverR == coverA persis).

Tapi ada risiko yang belum diukur: pada balon KELABU, membalik polaritas
menaruh interior kelabu, tinta hitam, DAN garis balon di kelas yang sama.
Flood fill lalu bisa menembus garis dan mengisi art gelap di luar balon.
coverR 0.80/0.86 untuk oval di dalam kotaknya terasa terlalu besar.

Jadi di sini diuji R melawan R2: sama, tapi piksel yang JAUH lebih gelap
dari latar tinta dikeluarkan dulu supaya garis balon tetap jadi dinding.
Keduanya digambar ke PNG untuk dilihat, bukan cuma diangkakan.
"""
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

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

import detect      # noqa: E402
import imgio       # noqa: E402
import textmask    # noqa: E402
import typeset     # noqa: E402

typeset.setup_fonts(verbose=False)
K9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def ink_in_crop(r, ox, oy, shape):
    hh, ww = shape
    out = np.zeros((hh, ww), np.uint8)
    ink = r.ink_mask
    if ink is None:
        return out
    x1, y1 = r.bbox[0] - ox, r.bbox[1] - oy
    mh, mw = ink.shape[:2]
    sy1, sx1 = max(y1, 0), max(x1, 0)
    sy2, sx2 = min(y1 + mh, hh), min(x1 + mw, ww)
    if sy2 > sy1 and sx2 > sx1:
        out[sy1:sy2, sx1:sx2] = ink[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1]
    return out


def flood_chain(binv, seed, el):
    hh, ww = binv.shape
    ff = binv.copy()
    m = np.zeros((hh + 2, ww + 2), np.uint8)
    cv2.floodFill(ff, m, textmask._white_seed(binv, seed), 128)
    interior = (ff == 128).astype(np.uint8) * 255
    if int((interior > 0).sum()) < hh * ww * 0.05:
        interior = binv
    interior = textmask._fill_holes(interior)
    closed = cv2.morphologyEx(interior, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return cv2.erode(cv2.bitwise_and(closed, cv2.bitwise_or(interior, binv)), el)


def satu(name, folder):
    img = imgio.load_any(ROOT / folder / f"{name}.JPG")
    typeset.set_page_width(img.shape[1])
    regions, _ = detect.detect(img)
    soft = textmask.ctd_soft_mask(img)
    for r in regions:
        textmask.build_region_mask(img, r, soft)
    print(f"\n===== {folder}/{name}", flush=True)
    h, w = img.shape[:2]
    petaR = np.zeros((h, w), np.uint8)
    petaR2 = np.zeros((h, w), np.uint8)
    for r in regions:
        if r.bubble_bbox is None:
            continue
        bx1, by1, bx2, by2 = r.bubble_bbox
        crop = img[by1:by2, bx1:bx2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, binv = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = ink_in_crop(r, bx1, by1, gray.shape)
        ring = (cv2.dilate(ink, K9) > 0) & (ink == 0)
        if not ring.any():
            ring = np.ones(gray.shape, bool)
        bg = float(np.median(gray[ring]))
        mad = float(np.median(np.abs(gray[ring].astype(np.int16) - bg)))
        balik = float((binv[ring] > 0).mean()) < 0.5

        seed = textmask._ink_center(r, bx1, by1)
        stroke = textmask._stroke_px(r.est_font_size)
        k = 2 * max(stroke, 1) + 1
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        bR = cv2.bitwise_not(binv) if balik else binv
        R = flood_chain(bR, seed, el)
        # R2: garis balon jadi dinding. Hanya perlu saat polaritas dibalik —
        # kalau tidak dibalik, kelas gelap (tinta + garis) sudah di luar binv.
        if balik:
            lantai = bg - max(3.0 * mad, 20.0)
            bR2 = cv2.bitwise_and(bR, (gray > lantai).astype(np.uint8) * 255)
        else:
            bR2 = bR
        R2 = flood_chain(bR2, seed, el)

        def tin(m):
            return -1.0 if not ink.any() else float((m[ink > 0] > 0).mean())

        print(f" r{r.idx} bg={bg:.0f} mad={mad:.0f} balik={balik} "
              f"lantai={(bg - max(3.0*mad, 20.0)) if balik else float('nan'):.0f}"
              f" | coverR={(R>0).mean():.2f} coverR2={(R2>0).mean():.2f}"
              f" | tintaR={tin(R):.2f} tintaR2={tin(R2):.2f}", flush=True)
        textmask._paste(petaR, (R > 0).astype(np.uint8) * 255, r.bubble_bbox)
        textmask._paste(petaR2, (R2 > 0).astype(np.uint8) * 255, r.bubble_bbox)

    base = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2BGR)
    for peta, tag, col in ((petaR, "R", (0, 0, 255)), (petaR2, "R2", (0, 200, 0))):
        vis = base.copy()
        sel = peta > 0
        vis[sel] = (vis[sel] * 0.5
                    + np.array(col, np.float32) * 0.5).astype(np.uint8)
        out = ROOT / f"_cr_{name}_{tag}.png"
        cv2.imwrite(str(out), cv2.resize(vis, None, fx=3, fy=3,
                                         interpolation=cv2.INTER_NEAREST))
        print(f" -> {out.name}", flush=True)


for _n in ("jp_cacatnew1", "jp_cacatnew2"):
    satu(_n, "cacatbaru")
