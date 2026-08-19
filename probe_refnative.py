#!/usr/bin/env python3
"""Ukur typeset referensi DI FRAME-NYA SENDIRI, bukan lewat crop halaman kita.

probe_align.py menunjukkan CONTOH/2.webp bukan hasil resize halaman kita
(rasio 1.1287 x 1.1490, geseran per-kuadran -28..+9 px), jadi crop per-balon di
koordinat kita bisa jatuh di balon sebelah — angka margin dari probe_margin.py
tidak bisa dipakai.

Di sini detector+textmask dijalankan pada gambar REFERENSI sendiri, lalu diukur
besaran yang TIDAK bergantung skala:
  cap_height / min(interior_h, interior_w)   -> untuk menentukan ukuran font
  margin_sisi / min(...)                     -> untuk menentukan pad_ratio
  lebar_tinta / lebar_interior_di_baris_itu  -> berapa penuh baris diisi
Rasio itu langsung bisa diterapkan ke balon kita di resolusi berapa pun.
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
CACHE = ROOT / ".probe_ref_native.pkl"
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
import detect    # noqa: E402
import imgio     # noqa: E402
import textmask  # noqa: E402
import typeset   # noqa: E402


def staged(img: np.ndarray):
    if CACHE.exists():
        with CACHE.open("rb") as f:
            return pickle.load(f)
    regions, _ = detect.detect(img)
    soft = textmask.ctd_soft_mask(img)
    for r in regions:
        textmask.build_region_mask(img, r, soft)
    textmask.partition_shared_interiors(img, regions)
    textmask.disjoin_overlapping_interiors(img, regions)
    with CACHE.open("wb") as f:
        pickle.dump(regions, f)
    return regions


def main() -> int:
    img = imgio.load_any(ROOT / "CONTOH" / "2.webp")
    h, w = img.shape[:2]
    typeset.setup_fonts(verbose=False)
    regions = staged(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(f"referensi {w}x{h}, {len(regions)} region terdeteksi\n")

    print(f"  {'idx':>3} {'interior':>9} {'baris':>5} {'cap_h':>6} "
          f"{'cap/min':>8} {'sisi':>5} {'sisi/min':>9} {'isi_lebar':>9}")
    caps, sides, fills, ratios = [], [], [], []
    for r in regions:
        box, mask = typeset._region_box_mask(r)
        bx1, by1, bx2, by2 = box
        mh, mw = mask.shape[:2]
        y2, x2 = min(by1 + mh, h), min(bx1 + mw, w)
        if y2 - by1 < 10 or x2 - bx1 < 10:
            continue
        m = mask[: y2 - by1, : x2 - bx1] > 0
        g = gray[by1:y2, bx1:x2]
        ink = (m & (g < 110)).astype(np.uint8)
        if int(ink.sum()) < 40:
            continue

        # Tinggi kapital = tinggi komponen tinta yang wajar sebagai huruf.
        n, _lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
        hs = [st[i][3] for i in range(1, n)
              if 4 <= st[i][3] <= 60 and 2 <= st[i][2] <= 60 and st[i][4] >= 6]
        if len(hs) < 4:
            continue
        cap = float(np.median(hs))

        ys, xs = np.nonzero(ink)
        rows = slice(int(ys.min()), int(ys.max()) + 1)
        bxs = np.nonzero(m[rows].any(0))[0]
        left, right = int(xs.min() - bxs.min()), int(bxs.max() - xs.max())
        side = (left + right) / 2
        mn = min(mh, mw)
        # Jumlah baris dari profil baris tinta: celah antar baris = 0 px tinta.
        rowink = ink.any(1).astype(np.int8)
        nlines = int(((rowink[1:] == 1) & (rowink[:-1] == 0)).sum() + rowink[0])
        fill = (xs.max() - xs.min() + 1) / max(len(bxs), 1)
        caps.append(cap); sides.append(side / mn); fills.append(fill)
        ratios.append(cap / mn)
        print(f"  {r.idx:>3} {f'{mw}x{mh}':>9} {nlines:>5} {cap:>6.1f} "
              f"{cap/mn:>8.3f} {side:>5.1f} {side/mn:>9.3f} {fill*100:>8.0f}%")

    print(f"\nn={len(caps)} balon terukur")
    print(f"cap_height        : median={np.median(caps):.1f} px "
          f"(pada halaman {w}x{h})")
    print(f"cap_h / min(sisi) : median={np.median(ratios):.3f} "
          f"p25={np.percentile(ratios,25):.3f} p75={np.percentile(ratios,75):.3f}")
    print(f"margin sisi / min : median={np.median(sides):.3f} "
          f"p25={np.percentile(sides,25):.3f} p75={np.percentile(sides,75):.3f}")
    print(f"lebar tinta / lebar interior: median={np.median(fills)*100:.0f}%")
    # Anime Ace cap height ~ 0.72 * ukuran font; ukuran font setara di halaman
    # kita = cap_ref * (tinggi_kita / tinggi_ref) / 0.72.
    scale = 1577 / h
    print(f"\n=> ukuran font setara di halaman 1134x1577: "
          f"{np.median(caps) * scale / 0.72:.1f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
