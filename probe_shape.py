#!/usr/bin/env python3
"""Silhouette ASCII mask interior region + rentang kiri/kanan per baris.

Dipakai untuk melihat bentuk sebenarnya interior — apakah oval, pita
melengkung, atau irisan — dan seberapa jauh pusat rongga bergeser dari
centroid yang dipakai layout() sebagai satu-satunya sumbu tengah.

    IDX=6 python probe_shape.py
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
IDX = int(os.environ.get("IDX", "6"))
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

import typeset   # noqa: E402


def main() -> int:
    typeset.setup_fonts(verbose=False)
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    r = next(x for x in regions if x.idx == IDX)
    box, m = typeset._region_box_mask(r)
    mh, mw = m.shape[:2]
    cx, cy = typeset._centroid(m)
    on = m > 0
    print(f"r{IDX} {mw}x{mh} centroid=({cx},{cy}) interior={int(on.sum())}")
    print(f"     bbox={r.bbox} bubble_bbox={r.bubble_bbox}")
    print("\n     baris: '#'=interior '.'=luar; | = kolom centroid")
    for y in range(mh):
        row = on[y]
        xs = np.flatnonzero(row)
        s = "".join("#" if row[x] else "." for x in range(mw))
        s = s[:cx] + ("|" if not row[cx] else "#") + s[cx + 1:]
        if xs.size:
            print(f"{y:>4} {s} L{int(xs[0]):>3} R{int(xs[-1]):>3} "
                  f"n{int(row.sum()):>3} c{(int(xs[0]) + int(xs[-1])) // 2:>3}")
        else:
            print(f"{y:>4} {s}   -")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
