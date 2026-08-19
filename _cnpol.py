"""Bongkar _interior_from_crop langkah demi langkah pada tiga kotak balon nyata.

Probe sebelumnya (_cacatnew.py) mengukur interior r0 cacatnew1 = 1214 px dari
7980 (15%) dan petanya menaruh warna itu di SUDUT LUAR balon, bukan di dalam.
Yang diuji di sini: pada langkah mana polaritas jadi terbalik — ambang Otsu,
aturan kelas-mayoritas, atau benih flood fill.

Kuncinya satu angka: kelas mana yang MEMUAT tinta region. Interior balon
menurut definisi mengelilingi teksnya sendiri, jadi kalau kelas yang dipilih
255 justru bukan kelas yang memuat tinta, polaritasnya salah.
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


def bongkar(name):
    img = imgio.load_any(ROOT / "cacatbaru" / f"{name}.JPG")
    typeset.set_page_width(img.shape[1])
    regions, _ = detect.detect(img)
    soft = textmask.ctd_soft_mask(img)
    for r in regions:
        textmask.build_region_mask(img, r, soft)
    print(f"\n===== {name}", flush=True)
    for r in regions:
        if r.bubble_bbox is None:
            print(f" r{r.idx}: bubble_bbox None", flush=True)
            continue
        bx1, by1, bx2, by2 = r.bubble_bbox
        crop = img[by1:by2, bx1:bx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        thr, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hi = binv > 0
        n_hi, n_lo = int(hi.sum()), int((~hi).sum())
        med_hi = float(np.median(gray[hi])) if n_hi else -1
        med_lo = float(np.median(gray[~hi])) if n_lo else -1

        # Kelas yang dipakai aturan sekarang: yang MAYORITAS.
        vals = gray[hi]
        if vals.size < binv.size - vals.size:
            vals = gray[~hi]
        balik_sekarang = float(np.median(vals)) < 128

        # Kelas yang MEMUAT tinta region: lihat piksel tinta di dalam crop.
        stroke = textmask._stroke_px(r.est_font_size)
        sx, sy = textmask._ink_center(r, bx1, by1)
        ink = r.ink_mask
        x1, y1, x2, y2 = r.bbox
        # Cincin di sekeliling tinta = latar tempat tinta itu berdiri.
        latar = np.zeros(gray.shape, bool)
        if ink is not None and ink.any():
            pad = np.zeros(gray.shape, np.uint8)
            iy2, ix2 = min(by2, y2), min(bx2, x2)
            iy1, ix1 = max(by1, y1), max(bx1, x1)
            if iy2 > iy1 and ix2 > ix1:
                pad[iy1 - by1:iy2 - by1, ix1 - bx1:ix2 - bx1] = ink[
                    iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1]
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            latar = (cv2.dilate(pad, k) > 0) & (pad == 0)
        f_hi = float(hi[latar].mean()) if latar.any() else -1

        interior = textmask._interior_from_crop(crop, stroke, (sx, sy))
        print(f" r{r.idx} box={r.bubble_bbox} {bx2-bx1}x{by2-by1}", flush=True)
        print(f"    otsu={thr:.0f}  kelas_TERANG n={n_hi} med={med_hi:.0f}"
              f"   kelas_GELAP n={n_lo} med={med_lo:.0f}", flush=True)
        print(f"    mayoritas={'GELAP' if n_lo > n_hi else 'TERANG'}"
              f"  aturan_sekarang_membalik={balik_sekarang}", flush=True)
        print(f"    latar_tinta ada di kelas TERANG sebanyak {f_hi:.3f}"
              f"   -> polaritas benar kalau angka ini > 0.5", flush=True)
        print(f"    benih={(sx, sy)} nilai_binv_di_benih="
              f"{int(binv[np.clip(sy,0,gray.shape[0]-1), np.clip(sx,0,gray.shape[1]-1)])}"
              f"  interior={int((interior>0).sum())}/{gray.size}"
              f" ({(interior>0).mean():.2f})", flush=True)


for _n in ("jp_cacatnew1", "jp_cacatnew2"):
    bongkar(_n)
