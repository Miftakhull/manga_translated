"""Regresi halaman BERSIH: apakah perbaikan polaritas + pemulihan tinta
mengubah interior balon PUTIH biasa?

_cnband.py sudah mengukur bahwa KEPUTUSAN polaritas aturan cincin sama dengan
aturan lama di 12 region balon putih (jp_6 + jp_13). Tapi edit yang mendarat
menambah satu hal lagi yang berlaku di SEMUA balon, bukan cuma yang dibalik:
tinta region di-OR balik ke interior. Di balon putih tinta itu GELAP, jadi
memang di luar `binv` — artinya interior bisa MELEBAR sebesar glyph.

Melebar sebesar glyph tidak otomatis salah (tinta ada di dalam balon menurut
definisi, dan _fill_holes memang sudah mencoba menambal lubang yang sama),
tapi harus DIUKUR, bukan diasumsikan: interior juga dipakai typeset untuk
memutuskan baris mana yang muat, dan interior yang melebar melewati garis balon
akan membuat teks menyentuh garis.

Yang diukur per region balon putih:
  cover        fraksi kotak yang terisi (lama vs baru)
  tinta        fraksi tinta region yang tercakup interior (lama vs baru)
  luar_garis   fraksi piksel interior BARU yang jatuh di piksel gelap
               (<= _LINE_DARK): kalau interior melebar menembus garis balon,
               angka ini naik. Inilah uji yang sesungguhnya.
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


def lama(crop, stroke, seed):
    """Salinan _interior_from_crop SEBELUM perbaikan — pembanding."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vals = gray[binv > 0]
    if vals.size < binv.size - vals.size:
        vals = gray[binv == 0]
    if np.median(vals) < 128:
        binv = cv2.bitwise_not(binv)
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
    interior = cv2.bitwise_and(closed, cv2.bitwise_or(interior, binv))
    k = 2 * max(stroke, 1) + 1
    return cv2.erode(interior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def satu(name, folder):
    img = imgio.load_any(ROOT / folder / f"{name}.JPG")
    typeset.set_page_width(img.shape[1])
    regions, _ = detect.detect(img)
    soft = textmask.ctd_soft_mask(img)
    for r in regions:
        textmask.build_region_mask(img, r, soft)
    print(f"\n===== {folder}/{name}", flush=True)
    print("  r | cover_lama cover_baru | tinta_lama tinta_baru"
          " | gelap_lama gelap_baru", flush=True)
    worst = 0.0
    for r in regions:
        if r.bubble_bbox is None:
            continue
        bx1, by1, bx2, by2 = r.bubble_bbox
        crop = img[by1:by2, bx1:bx2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        stroke = textmask._stroke_px(r.est_font_size)
        seed = textmask._ink_center(r, bx1, by1)
        ink = textmask._ink_in_crop(r, bx1, by1, crop.shape[:2])
        A = lama(crop, stroke, seed)
        B = textmask._interior_from_crop(crop, stroke, seed, ink)
        dark = gray <= textmask._LINE_DARK

        def f(m):
            return float((m > 0).mean())

        def t(m):
            return -1.0 if not ink.any() else float((m[ink > 0] > 0).mean())

        def g(m):
            s = m > 0
            return 0.0 if not s.any() else float(dark[s].mean())

        worst = max(worst, g(B) - g(A))
        print(f" r{r.idx} | {f(A):10.3f} {f(B):10.3f} | {t(A):10.3f} {t(B):10.3f}"
              f" | {g(A):10.3f} {g(B):10.3f}", flush=True)
    print(f"  kenaikan piksel-gelap TERBURUK di halaman ini: {worst:+.3f}",
          flush=True)


for _n in ("jp_6", "jp_13"):
    satu(_n, "hasilnew")
