"""Uji dua aturan polaritas pengganti pada balon SCREENTONE, plus regresi.

Temuan _cnpol.py: pada tiga kotak balon cacat baru, kelas Otsu yang dipilih
sebagai 'interior' justru kelas yang TIDAK memuat latar tinta region —
`latar_tinta ada di kelas TERANG` = 0.000 / 0.014 / 0.025. Sebabnya aturan
`np.median(vals) < 128` di _interior_from_crop: balon di sini ber-screentone
kelabu (median 140-150) di atas halaman PUTIH (239). Kelas mayoritas memang
balonnya, tapi mediannya 147 — DI ATAS 128 — jadi tidak dibalik, interior
diambil dari kelas terang = halaman di LUAR balon.

Yang diukur di sini, dua kandidat aturan:

  R  polaritas dari CINCIN LATAR TINTA: kelas interior = kelas yang memuat
     mayoritas piksel di sekeliling tinta region. Ini alasan struktural yang
     sama dengan _keep_ink_lobes ("isian hanya boleh mengisi rongga tempat
     tinta Jepangnya berada"), bukan ambang absolut.
     Risikonya: pada balon kelabu, ambang Otsu tunggal menaruh tinta GELAP dan
     GARIS balon di kelas yang sama dengan interior kelabu -> flood fill bisa
     menembus garis dan mengisi art gelap di luar balon. Itu yang diukur lewat
     'cover' dan 'tembus'.

  B  PITA di sekitar latar tinta: interior = piksel yang kecerahannya dekat
     dengan latar tinta (|blur(g) - bg| <= tol). Ini memisahkan tiga kelas
     sekaligus (tinta gelap / interior kelabu / halaman putih), jadi garis
     balon tetap jadi dinding.

Regresi diuji pada hasilnew/jp_6.JPG dan jp_13.JPG (balon PUTIH biasa): aturan
baru harus memberi keputusan yang SAMA dengan sekarang di sana.
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
    """ink_mask region dipetakan ke koordinat crop."""
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


def flood(binv, seed):
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
    return cv2.bitwise_and(closed, cv2.bitwise_or(interior, binv))


def satu(name, folder):
    img = imgio.load_any(ROOT / folder / f"{name}.JPG")
    typeset.set_page_width(img.shape[1])
    regions, _ = detect.detect(img)
    soft = textmask.ctd_soft_mask(img)
    for r in regions:
        textmask.build_region_mask(img, r, soft)
    print(f"\n===== {folder}/{name}", flush=True)
    print("  r  otsu   bg  MAD | sekarang  cincin | coverA coverB coverR"
          " | tintaA tintaB tintaR", flush=True)
    for r in regions:
        if r.bubble_bbox is None:
            continue
        bx1, by1, bx2, by2 = r.bubble_bbox
        crop = img[by1:by2, bx1:bx2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        thr, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = ink_in_crop(r, bx1, by1, gray.shape)
        ring = (cv2.dilate(ink, K9) > 0) & (ink == 0)
        if not ring.any():
            ring = np.ones(gray.shape, bool)
        bg = float(np.median(gray[ring]))
        mad = float(np.median(np.abs(gray[ring].astype(np.int16) - bg)))

        vals = gray[binv > 0]
        if vals.size < binv.size - vals.size:
            vals = gray[binv == 0]
        skr = "balik" if float(np.median(vals)) < 128 else "tetap"
        # Aturan R: kelas interior = kelas yang memuat mayoritas cincin latar.
        cincin_terang = float((binv[ring] > 0).mean())
        cin = "tetap" if cincin_terang >= 0.5 else "balik"

        seed = textmask._ink_center(r, bx1, by1)
        stroke = textmask._stroke_px(r.est_font_size)
        k = 2 * max(stroke, 1) + 1
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # A = kode sekarang
        A = textmask._interior_from_crop(crop, stroke, seed)
        # R = polaritas dari cincin
        bR = cv2.bitwise_not(binv) if cin == "balik" else binv
        R = cv2.erode(flood(bR, seed), el)
        # B = pita di sekitar bg
        tol = max(24.0, 3.0 * mad)
        blur = cv2.medianBlur(gray, 5)
        band = (np.abs(blur.astype(np.float32) - bg) <= tol).astype(np.uint8) * 255
        B = cv2.erode(flood(band, seed), el)

        def cov(m):
            return float((m > 0).mean())

        def tin(m):
            return -1.0 if not ink.any() else float((m[ink > 0] > 0).mean())

        print(f" r{r.idx} {thr:5.0f} {bg:4.0f} {mad:4.0f} | {skr:8s} {cin:6s}"
              f" | {cov(A):6.2f} {cov(B):6.2f} {cov(R):6.2f}"
              f" | {tin(A):6.2f} {tin(B):6.2f} {tin(R):6.2f}", flush=True)


for _n in ("jp_cacatnew1", "jp_cacatnew2"):
    satu(_n, "cacatbaru")
for _n in ("jp_6", "jp_13"):
    satu(_n, "hasilnew")
