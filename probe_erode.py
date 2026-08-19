#!/usr/bin/env python3
"""Apakah mask hapus memakan GARIS balon? Diukur, bukan dikira-kira.

Temuan yang memulai probe ini: pada hasil akhir ada piksel gelap halaman asli
yang jadi terang di dalam pita 4 px di LUAR interior balon — 76 px di r7 (9.7%
panjang garisnya), 49 px di r11 (21.3%), 15 px di r12. Semuanya 100% jatuh di
dalam erase_mask, jadi bukan bocoran inpaint melainkan mask hapusnya sendiri
yang melewati garis.

Penyebabnya bisa ditebak dari kode: _adaptive_dilate memekarkan ink_mask dengan
kernel sampai 31 px (textmask.py:140), dan teks Jepang vertikal di balon sempit
duduk cuma beberapa piksel dari garisnya. Dilasi itu menyeberang garis, lalu
erase menghapus apa yang tersentuh.

Yang diuji di sini bukan cuma penyebabnya tapi juga harga obatnya. Mengurung
mask hapus ke interior balon menghentikan pengikisan, TAPI kalau dikurung ke
interior yang sudah dikikis stroke (_interior_from_crop mengikis segitu di
baris 353-354), halo abu-abu huruf yang menempel di garis ikut selamat — dan
itu persis ghost outline yang jadi penyebab nomor satu di komentar _halo_pass.
Jadi dua varian diukur berdampingan:

  ketat  : ink_mask & interior                 (seperti tersimpan, sudah dikikis)
  pas    : ink_mask & (interior dilasi stroke) (sampai garis, tidak menembus)
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

RING = 4  # tebal pita "garis balon" di luar interior, px


def _page(mask: np.ndarray, box: tuple[int, int, int, int],
          shape: tuple[int, int]) -> np.ndarray:
    """Mask lokal -> kanvas halaman. `box` HARUS kotak asal mask itu sendiri.

    Versi pertama probe ini memasang bubble_mask pada region.bbox dan angkanya
    salah semua: bubble_mask hidup di koordinat bubble_bbox (textmask.py:413-417
    memotong crop dari bubble_bbox), yang untuk r8 mulai 29 px lebih ke atas.
    Mask yang bergeser membuat pita "garis balon" jatuh di tempat yang bukan
    garis — r8 dilaporkan kehilangan 37 px garis padahal pitanya sendiri salah
    letak. Pakai typeset._region_box_mask(), sumber yang sama dengan renderer.
    """
    x1, y1, _, _ = box
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    mh, mw = mask.shape[:2]
    y2, x2 = min(y1 + mh, h), min(x1 + mw, w)
    if y2 > y1 and x2 > x1:
        out[y1:y2, x1:x2] = mask[: y2 - y1, : x2 - x1]
    return out


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    h, w = img.shape[:2]
    dark = img.mean(2) < 110
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    print(f"{'idx':>3} {'stroke':>6} | {'garis':>6} {'kikis_kini':>10} {'ketat':>6} "
          f"{'pas':>5} | {'halo_lolos_ketat':>16} {'pas':>5}")
    tot = np.zeros(4, int)
    for r in regions:
        if r.ink_mask is None or r.bubble_mask is None:
            continue
        stroke = textmask._stroke_px(r.est_font_size or 20)
        ink = _page(r.ink_mask, r.bbox, (h, w)) > 0
        bbox, bmask = typeset._region_box_mask(r)
        inner = (_page(bmask, bbox, (h, w)) > 0).astype(np.uint8)
        if not inner.any():
            continue
        kk = 2 * stroke + 1
        grown = cv2.dilate(
            inner, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))) > 0
        ring = (cv2.dilate(inner, k3, iterations=RING) - inner).astype(bool)

        line = int((dark & ring).sum())
        now = int((dark & ring & ink).sum())          # garis yang kini dihapus
        tight = int((dark & ring & ink & (inner > 0)).sum())
        fitp = int((dark & ring & ink & grown).sum())
        # halo = tinta yang SEHARUSNYA dihapus tapi lolos oleh pengurungan
        halo_t = int((dark & ink & ~(inner > 0)).sum())
        halo_f = int((dark & ink & ~grown).sum())
        tot += (line, now, tight, fitp)
        print(f"{r.idx:>3} {stroke:>6} | {line:>6} {now:>10} {tight:>6} {fitp:>5} | "
              f"{halo_t:>16} {halo_f:>5}")
    print(f"\ntotal garis={tot[0]} kini_terkikis={tot[1]} "
          f"ketat={tot[2]} pas={tot[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
