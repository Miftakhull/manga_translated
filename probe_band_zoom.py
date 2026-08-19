#!/usr/bin/env python3
"""Zoom piksel pita yang MASIH terkikis di r11/r12: garis balon atau glyph?

probe_outline.py melaporkan guard menyisakan 36 px di r11 dan 16 px di r12 —
pecahan gelap di pita yang bentangnya di bawah ambang, jadi tidak dilindungi.
Angka itu tidak cukup untuk memutuskan: 36 px bisa berarti garis balon
berlubang, atau bisa berarti coretan glyph yang menyeberang pita dan memang
harus dihapus. Jadi dilihat.

Tiap petak: ASLI | BERSIH | pita+tanda, diperbesar 8x. Merah = piksel yang
masih terkikis, hijau = piksel yang dilindungi guard.
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
from probe_outline import _page, outline_guard, DARK, _BAND_EXTRA  # noqa: E402

Z = 8
WATCH = [7, 8, 11, 12]


def main() -> int:
    from PIL import Image  # noqa: PLC0415

    img = imgio.load_any(ROOT / "jepang_002.webp")
    clean = imgio.load_any(ROOT / "debug" / "jepang_002" / "09_cleaned.png")
    h, w = img.shape[:2]
    dark = img.mean(2) < DARK
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    typeset.setup_fonts(verbose=False)
    guard = outline_guard(img, regions)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = ROOT / "_cmp"
    out.mkdir(exist_ok=True)

    for r in regions:
        if r.idx not in WATCH or r.bubble_mask is None or r.ink_mask is None:
            continue
        box, bm = typeset._region_box_mask(r)
        inner = (_page(bm, box, (h, w)) > 0).astype(np.uint8)
        st = textmask._stroke_px(r.est_font_size or 20)
        band = (cv2.dilate(inner, k3, iterations=st + _BAND_EXTRA) - inner).astype(bool)
        ink = _page(r.ink_mask, r.bbox, (h, w)) > 0
        left = dark & band & ink & ~guard          # masih terkikis
        kept = guard & band                        # dilindungi
        ys, xs = np.nonzero(left if left.any() else kept)
        if ys.size == 0:
            continue
        m = 12
        y1, y2 = max(int(ys.min()) - m, 0), min(int(ys.max()) + m + 1, h)
        x1, x2 = max(int(xs.min()) - m, 0), min(int(xs.max()) + m + 1, w)
        mark = clean[y1:y2, x1:x2].copy()
        mark[kept[y1:y2, x1:x2]] = (0, 200, 0)
        mark[left[y1:y2, x1:x2]] = (255, 0, 0)
        strip = np.concatenate(
            [img[y1:y2, x1:x2], clean[y1:y2, x1:x2], mark], axis=1)
        big = cv2.resize(strip, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
        p = out / f"band_r{r.idx}.png"
        Image.fromarray(big).save(p)
        print(f"r{r.idx} sisa={int(left.sum())} terjaga={int(kept.sum())} "
              f"kotak=({x1},{y1})-({x2},{y2}) -> {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
