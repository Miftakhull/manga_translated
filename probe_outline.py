#!/usr/bin/env python3
"""Bisakah GARIS balon dijaga dari mask hapus tanpa meninggalkan hantu teks?

Dua kandidat obat diukur, karena yang pertama sudah terbukti salah:

  kurung  : mask hapus dikurung ke interior balon (dilasi setebal stroke).
            Menghentikan pengikisan hampir seluruhnya (110 -> 5 px) TAPI
            probe_erode.py mengukur harganya: 79/148/36/187 px tinta Jepang di
            r7/r8/r11/r12 ikut selamat. Itu bukan halo, itu GLYPH — zoom
            (177,1173)-(188,1208) memperlihatkan 'すか' utuh. Penyebabnya
            interior bukan selubung yang bisa dipercaya: glyph yang MENEMPEL di
            garis balon membuat takik yang terbuka ke tepi, jadi flood fill tak
            bisa mengelilinginya dan _fill_holes tak bisa menambalnya (hanya
            lubang tertutup). Jadi 'di luar interior' != 'di luar balon'.

  guard   : lindungi GARISNYA saja. Di gambar ASLI, garis balon adalah satu
            struktur gelap besar yang menyambung di pita sekitar tepi interior;
            pecahan glyph di pita yang sama kecil-kecil. Jadi: ambil piksel
            gelap di pita itu, dan lindungi yang termasuk komponen besar.
            Glyph yang menempel garis cuma menyumbang beberapa piksel sentuh —
            sisa yang kelihatan paling berupa bintik, bukan huruf.

Yang dilaporkan per region: garis yang terkikis sekarang, sesudah guard, dan
berapa tinta glyph yang ikut terlindungi (itulah risiko hantu).
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

DARK = 110
# Pita "garis balon" = dari tepi interior sampai sejauh ini keluar. Harus
# stroke-aware: _interior_from_crop mengikis interior sebesar stroke, jadi
# garisnya duduk stroke..2*stroke px di luar. Pita tetap 4 px di r8 (stroke 4)
# melaporkan garis=0 px — pitanya sendiri yang belum sampai ke garis.
_BAND_EXTRA = 4
# Komponen gelap di pita dianggap GARIS kalau bentang terpanjangnya (lebar atau
# tinggi kotak pembatas) minimal sekian kali ketebalan stroke. Garis balon
# selalu membentang jauh di sepanjang tepi; pecahan glyph yang menyeberang pita
# ringkas. Dipakai bentang, bukan luas relatif: di r11/r12 garisnya terputus
# jadi beberapa potong, dan ambang "0.25 x komponen terbesar" membuang potongan
# yang sah.
_MIN_SPAN = 8


def _page(mask: np.ndarray, box, shape) -> np.ndarray:
    x1, y1 = box[0], box[1]
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    mh, mw = mask.shape[:2]
    y2, x2 = min(y1 + mh, h), min(x1 + mw, w)
    if y2 > y1 and x2 > x1:
        out[y1:y2, x1:x2] = mask[: y2 - y1, : x2 - x1]
    return out


def outline_guard(img: np.ndarray, regions) -> np.ndarray:
    """Piksel garis balon yang harus lolos dari erase, skala halaman."""
    h, w = img.shape[:2]
    dark = img.mean(2) < DARK
    guard = np.zeros((h, w), bool)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rows = []
    for r in regions:
        if r.bubble_mask is None:
            continue
        box, bm = typeset._region_box_mask(r)
        if bm.min() == 255:          # persegi penuh: bukan balon, tak ada garis
            continue
        inner = (_page(bm, box, (h, w)) > 0).astype(np.uint8)
        if not inner.any():
            continue
        st = textmask._stroke_px(r.est_font_size or 20)
        band = (cv2.dilate(inner, k3, iterations=st + _BAND_EXTRA) - inner).astype(bool)
        sel = (dark & band).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(sel, 8)
        if n <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        # Panjang komponen, bukan luasnya: garis balon melengkung sepanjang tepi
        # interior jadi kotak pembatasnya besar, sedangkan pecahan glyph yang
        # menyeberang pita tetap ringkas. Ambang luas relatif (0.25 * terbesar)
        # gagal di r11/r12: di sana garisnya terputus jadi beberapa potong
        # sehingga potongan sah pun jatuh di bawah ambang.
        span = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
        keep = 1 + np.flatnonzero(span >= _MIN_SPAN * max(st, 1))
        guard |= np.isin(lab, keep)
        rows.append((r.idx, st, int(areas.max()), sorted(span.tolist(), reverse=True)[:6]))
    if os.environ.get("VERBOSE"):
        for idx, st, amax, sp in rows:
            print(f"  r{idx} st={st} luas_maks={amax} span6={sp}")
    return guard


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    h, w = img.shape[:2]
    dark = img.mean(2) < DARK
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    typeset.setup_fonts(verbose=False)
    guard = outline_guard(img, regions)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    print(f"{'idx':>3} {'st':>2} | {'garis':>6} {'kikis_kini':>10} {'kikis_guard':>11} "
          f"| {'glyph_terjaga':>13}")
    tot = np.zeros(3, int)
    for r in regions:
        if r.ink_mask is None or r.bubble_mask is None:
            continue
        box, bm = typeset._region_box_mask(r)
        if bm.min() == 255:
            continue
        inner = (_page(bm, box, (h, w)) > 0).astype(np.uint8)
        if not inner.any():
            continue
        st = textmask._stroke_px(r.est_font_size or 20)
        band = (cv2.dilate(inner, k3, iterations=st + _BAND_EXTRA) - inner).astype(bool)
        ink = _page(r.ink_mask, r.bbox, (h, w)) > 0
        line = int((dark & band).sum())
        now = int((dark & band & ink).sum())
        aft = int((dark & band & ink & ~guard).sum())
        # Risiko hantu: tinta yang MEMANG glyph tapi ikut terlindungi guard.
        # Diukur di dalam interior saja — di sana tidak ada garis balon, jadi
        # apa pun yang gelap di situ adalah huruf.
        ghost = int((dark & ink & guard & (inner > 0)).sum())
        tot += (line, now, aft)
        print(f"{r.idx:>3} {st:>2} | {line:>6} {now:>10} {aft:>11} | {ghost:>13}")
    print(f"\ntotal garis={tot[0]} kikis_kini={tot[1]} kikis_guard={tot[2]}")
    print(f"guard px={int(guard.sum())}  di dalam interior mana pun="
          f"{int((guard & dark).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
