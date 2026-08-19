#!/usr/bin/env python3
"""Bandingkan per-balon: crop kita vs crop CONTOH/2.webp di kotak balon SAMA.

Tujuan kalibrasi: melihat berapa baris dan sebesar apa font yang dipakai
typesetter referensi di balon yang persis sama, bukan menebak dari angka
persentil global.

Keluaran:
  _cmp/ref_r<idx>.png   kiri = render kita, kanan = referensi, kotak identik
  tabel tinggi kapital terukur DI DALAM tiap balon referensi
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
CACHE = ROOT / ".probe_pre.pkl"
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

import cv2      # noqa: E402
import imgio    # noqa: E402
from PIL import Image  # noqa: E402


def cap_heights(crop: np.ndarray) -> list[int]:
    """Tinggi komponen tinta yang plausibel sebagai huruf kapital."""
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    ink = (g < 110).astype(np.uint8)
    n, _lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    return [st[i][3] for i in range(1, n)
            if 5 <= st[i][3] <= 40 and 2 <= st[i][2] <= 40 and st[i][4] >= 8]


def main() -> int:
    if not CACHE.exists():
        print("jalankan probe_interior.py dulu (butuh .probe_pre.pkl)")
        return 1
    with CACHE.open("rb") as f:
        regions = pickle.load(f)

    ours_p = ROOT / "_cmp" / "probe_font.png"
    ours = np.asarray(Image.open(ours_p).convert("RGB"), np.uint8)
    h, w = ours.shape[:2]
    ref = np.asarray(
        Image.open(ROOT / "CONTOH" / "2.webp").convert("RGB").resize((w, h), Image.LANCZOS),
        np.uint8,
    )

    print(f"  {'idx':>3} {'kotak balon':<26} {'lebar':>5} {'tinggi':>6} "
          f"{'cap_ref p50':>11} {'n':>4}")
    for r in regions:
        bx1, by1, bx2, by2 = r.bubble_bbox or r.bbox
        bx1, by1 = max(bx1, 0), max(by1, 0)
        bx2, by2 = min(bx2, w), min(by2, h)
        if bx2 - bx1 < 4 or by2 - by1 < 4:
            continue
        a, b = ours[by1:by2, bx1:bx2], ref[by1:by2, bx1:bx2]
        hs = cap_heights(b)
        p50 = float(np.median(hs)) if hs else 0.0
        print(f"  {r.idx:>3} {str((bx1, by1, bx2, by2)):<26} {bx2-bx1:>5} "
              f"{by2-by1:>6} {p50:>11.1f} {len(hs):>4}")

        # Sisi-sisian dengan pemisah merah, di-upscale 3x supaya glyph terbaca.
        gap = np.full((b.shape[0], 3, 3), (220, 40, 40), np.uint8)
        pair = np.hstack([a, gap, b])
        big = Image.fromarray(pair).resize(
            (pair.shape[1] * 3, pair.shape[0] * 3), Image.LANCZOS
        )
        big.save(ROOT / "_cmp" / f"ref_r{r.idx:02d}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
