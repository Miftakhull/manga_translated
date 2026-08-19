#!/usr/bin/env python3
"""Kenapa disjoin memangkas balon: laporan luas interior per region + overlay.

Menyimpan cache SEBELUM disjoin, jadi efek disjoin bisa diukur berulang tanpa
menjalankan detector+CTD lagi (keduanya di CPU, ~2 menit).

Keluaran:
  _cmp/interior_before.png / _cmp/interior_after.png  overlay warna per region
  tabel luas interior sebelum/sesudah + berapa persen hilang
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

import detect      # noqa: E402
import imgio       # noqa: E402
import textmask    # noqa: E402
import typeset     # noqa: E402


def pre_disjoin_regions(img: np.ndarray):
    """detect + mask + partisi balon ganda. TANPA disjoin — itu yang diuji."""
    if CACHE.exists():
        with CACHE.open("rb") as f:
            return pickle.load(f)
    regions, _ = detect.detect(img)
    soft = textmask.ctd_soft_mask(img)
    for r in regions:
        textmask.build_region_mask(img, r, soft)
    textmask.partition_shared_interiors(img, regions)
    with CACHE.open("wb") as f:
        pickle.dump(regions, f)
    return regions


def page_map(r, h: int, w: int) -> np.ndarray:
    (bx1, by1, _, _), m = typeset._region_box_mask(r)
    o = np.zeros((h, w), np.uint8)
    mh, mw = m.shape[:2]
    y2, x2 = min(by1 + mh, h), min(bx1 + mw, w)
    if y2 > by1 and x2 > bx1:
        o[by1:y2, bx1:x2] = m[: y2 - by1, : x2 - bx1]
    return o


_PALETTE = np.array([
    (230, 60, 60), (60, 160, 230), (70, 200, 90), (240, 170, 40),
    (180, 90, 220), (30, 200, 200), (230, 110, 180), (140, 140, 60),
    (90, 110, 230), (200, 80, 30), (60, 200, 150), (200, 200, 60),
    (150, 60, 120),
], np.uint8)


def overlay(img: np.ndarray, regions, path: Path) -> None:
    from PIL import Image
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    for r in regions:
        m = page_map(r, h, w) > 0
        col = _PALETTE[r.idx % len(_PALETTE)].astype(np.float32)
        out[m] = out[m] * 0.55 + col * 0.45
    Image.fromarray(out.astype(np.uint8)).save(path)


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    h, w = img.shape[:2]
    regions = pre_disjoin_regions(img)
    (ROOT / "_cmp").mkdir(exist_ok=True)

    before = {r.idx: int((page_map(r, h, w) > 0).sum()) for r in regions}
    overlay(img, regions, ROOT / "_cmp" / "interior_before.png")

    fixed = textmask.disjoin_overlapping_interiors(img, regions)
    after = {r.idx: int((page_map(r, h, w) > 0).sum()) for r in regions}
    overlay(img, regions, ROOT / "_cmp" / "interior_after.png")

    print(f"disjoin menyentuh {fixed} region\n")
    print(f"  {'idx':>3} {'bbox_region':<26} {'sebelum':>8} {'sesudah':>8} {'hilang':>7}")
    for r in regions:
        b, a = before[r.idx], after[r.idx]
        pct = 0.0 if not b else (b - a) / b * 100
        print(f"  {r.idx:>3} {str(r.bbox):<26} {b:>8} {a:>8} {pct:>6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
