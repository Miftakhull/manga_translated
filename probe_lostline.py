#!/usr/bin/env python3
"""Garis balon yang HILANG di plat bersih — diukur pada hasil, bukan pada mask.

probe_erode.py mengukur irisan ink_mask dengan pita garis, jadi angkanya tidak
bergerak walau erase_mask sudah dilindungi: penjaganya bekerja di tingkat
HALAMAN (compose_page_mask), sesudah ink_mask per region terbentuk. Yang
menentukan hasil akhir cuma satu: piksel yang gelap di gambar ASLI dan jadi
terang di 09_cleaned.png.

Karena itu probe ini membandingkan dua gambar, bukan dua mask, dan memisahkan
kehilangan di dalam pita garis balon dari kehilangan di dalam interior (yang
memang harus hilang — itu teks Jepangnya).
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

LIGHT = 170   # "sudah jadi latar" — di atas ambang gelap dengan margin lebar


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    clean = imgio.load_any(ROOT / "debug" / "jepang_002" / "09_cleaned.png")
    if clean.shape[:2] != img.shape[:2]:
        print(f"[gagal] ukuran beda {img.shape[:2]} vs {clean.shape[:2]}")
        return 1
    h, w = img.shape[:2]
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)

    dark = img.mean(2) < textmask._LINE_DARK
    lost = dark & (clean.mean(2) > LIGHT)
    guard = textmask.bubble_outline_guard(img, regions) > 0
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    print(f"{'idx':>3} {'st':>2} | {'garis':>6} {'hilang':>6} {'%':>5} | "
          f"{'terjaga':>7} {'sisa_teks':>9}")
    tot = np.zeros(2, int)
    for r in regions:
        if r.bubble_mask is None:
            continue
        box, bm = textmask._eff_box_mask(r)
        if bm.size == 0 or bm.min() == 255:
            continue
        inner = np.zeros((h, w), np.uint8)
        textmask._paste(inner, (bm > 0).astype(np.uint8), box)
        if not inner.any():
            continue
        st = textmask._stroke_px(r.est_font_size or 20)
        band = (cv2.dilate(inner, k3, iterations=st + textmask._LINE_BAND)
                - inner).astype(bool)
        line = int((dark & band).sum())
        gone = int((lost & band).sum())
        kept = int((guard & band).sum())
        # Sisa teks: piksel gelap di dalam interior yang TIDAK jadi terang.
        # Kalau penjaga garis kebablasan, angka inilah yang naik.
        stay = int((dark & (inner > 0) & ~lost).sum())
        tot += (line, gone)
        pct = 100.0 * gone / line if line else 0.0
        print(f"{r.idx:>3} {st:>2} | {line:>6} {gone:>6} {pct:>5.1f} | "
              f"{kept:>7} {stay:>9}")
    print(f"\ntotal garis={tot[0]} hilang={tot[1]} "
          f"({100.0 * tot[1] / max(tot[0], 1):.2f}%)")
    print(f"tinta hilang seluruh halaman={int(lost.sum())} "
          f"di dalam pita garis={tot[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
