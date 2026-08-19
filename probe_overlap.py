#!/usr/bin/env python3
"""Diagnostik irisan interior balon di halaman nyata — tanpa DeepL, tanpa OCR.

Menjawab satu pertanyaan: pasangan region mana yang interiornya beririsan,
kenapa (mask asli vs fallback persegi), dan apakah
textmask.disjoin_overlapping_interiors() menghabiskannya.
"""

from __future__ import annotations

import os
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

import detect      # noqa: E402
import imgio       # noqa: E402
import textmask    # noqa: E402
import typeset     # noqa: E402


def main() -> int:
    img = imgio.load_any(ROOT / (sys.argv[1] if len(sys.argv) > 1 else "jepang_002.webp"))
    h, w = img.shape[:2]
    regions, _bubbles = detect.detect(img)
    soft = textmask.ctd_soft_mask(img)
    for r in regions:
        textmask.build_region_mask(img, r, soft)
    textmask.partition_shared_interiors(img, regions)

    def pmap(r) -> np.ndarray:
        (bx1, by1, _, _), m = typeset._region_box_mask(r)
        o = np.zeros((h, w), np.uint8)
        mh, mw = m.shape[:2]
        y2, x2 = min(by1 + mh, h), min(bx1 + mw, w)
        if y2 > by1 and x2 > bx1:
            o[by1:y2, bx1:x2] = m[: y2 - by1, : x2 - bx1]
        return o

    def report(tag: str) -> int:
        worst = 0
        maps = {r.idx: pmap(r) for r in regions}
        for i, a in enumerate(regions):
            for b in regions[i + 1:]:
                ov = int(((maps[a.idx] > 0) & (maps[b.idx] > 0)).sum())
                if ov:
                    print(f"  {tag} {a.idx}-{b.idx} ov={ov} "
                          f"sq={_is_square(a)}/{_is_square(b)} "
                          f"shared={a.shared_bubble_bbox is not None}"
                          f"/{b.shared_bubble_bbox is not None}")
                worst = max(worst, ov)
        print(f"{tag}: worst={worst}")
        return worst

    def _is_square(r) -> bool:
        """True kalau interiornya fallback persegi 255 penuh, bukan bentuk balon."""
        box = r.bubble_bbox or r.bbox
        bh, bw = box[3] - box[1], box[2] - box[0]
        m = r.bubble_mask
        return m is None or m.shape[:2] != (bh, bw)

    before = report("BEFORE")
    fixed = textmask.disjoin_overlapping_interiors(img, regions)
    print(f"disjoin memperbaiki {fixed} region")
    after = report("AFTER")
    print(f"\nRINGKAS: {before} -> {after} px")
    return 0 if after == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
