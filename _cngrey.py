"""Halaman uji balon KELABU ber-screentone: apakah bentuknya benar-benar
memancing cacat aturan polaritas LAMA?

Sebuah penjaga regresi hanya bernilai kalau ia GAGAL pada kode lama. Kedua
pembangun balon ganda yang ada (`make_double_bubble_page`,
`make_adjacent_bubbles_page`) mengecat balon PUTIH (`arr[fill > 0] =
(255,255,255)`), jadi tidak satu pun menyentuh kasus yang baru diperbaiki.

Yang diukur di sini, per region:
  lama_*  interior menurut salinan verbatim _interior_from_crop SEBELUM
          perbaikan (aturan absolut `median(kelas mayoritas) < 128`)
  baru_*  interior menurut kode yang mendarat
  tinta   fraksi tinta region yang tercakup interior. Inilah yang menentukan
          apakah tinta Jepang terhapus (build_fill_mask memakai interior ini).
  luar    fraksi piksel interior yang jatuh di LUAR balon (inner == 0).
          Interior yang bocor keluar = terjemahan ditata di atas art.

Lolos berarti: lama gagal (tinta rendah ATAU luar tinggi) dan baru benar.
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

import selftest    # noqa: E402
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


def paste(mask, box, shape):
    x1, y1, _, _ = box
    out = np.zeros(shape, np.uint8)
    mh, mw = mask.shape[:2]
    y2, x2 = min(y1 + mh, shape[0]), min(x1 + mw, shape[1])
    if y2 > y1 and x2 > x1:
        out[y1:y2, x1:x2] = mask[: y2 - y1, : x2 - x1]
    return out


clean, img, inner, regions = selftest.make_grey_bubble_page()
print(f"halaman {img.shape[1]}x{img.shape[0]}  region={len(regions)}", flush=True)
for r in regions:
    textmask.build_region_mask(img, r, None)

print("  r |  med_kotak may_med | tinta_lama tinta_baru | luar_lama luar_baru",
      flush=True)
for r in regions:
    bx1, by1, bx2, by2 = r.bubble_bbox
    crop = img[by1:by2, bx1:bx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    stroke = textmask._stroke_px(r.est_font_size)
    seed = textmask._ink_center(r, bx1, by1)
    ink = textmask._ink_in_crop(r, bx1, by1, crop.shape[:2])
    A = lama(crop, stroke, seed)
    B = textmask._interior_from_crop(crop, stroke, seed, ink)

    # Angka yang menjelaskan MENGAPA aturan lama salah: median kelas mayoritas.
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v = gray[binv > 0]
    if v.size < binv.size - v.size:
        v = gray[binv == 0]
    may_med = float(np.median(v))

    def t(m):
        return -1.0 if not ink.any() else float((m[ink > 0] > 0).mean())

    def luar(m):
        pm = paste(m, r.bubble_bbox, inner.shape)
        s = pm > 0
        return 0.0 if not s.any() else float((inner[s] == 0).mean())

    print(f" r{r.idx} | {float(np.median(gray)):9.1f} {may_med:7.1f} |"
          f" {t(A):10.3f} {t(B):10.3f} | {luar(A):9.3f} {luar(B):9.3f}",
          flush=True)

vis = img.copy()
for r, col in zip(regions, ((255, 0, 0), (0, 160, 255))):
    bx1, by1, _, _ = r.bubble_bbox
    pm = paste(r.bubble_mask, r.bubble_bbox, inner.shape) if r.bubble_mask is not None \
        else np.zeros(inner.shape, np.uint8)
    vis[pm > 0] = (0.45 * np.array(col) + 0.55 * vis[pm > 0]).astype(np.uint8)
cv2.imwrite(str(ROOT / "_cngrey.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print(" -> _cngrey.png", flush=True)
