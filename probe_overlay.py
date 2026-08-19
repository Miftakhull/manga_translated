#!/usr/bin/env python3
"""Overlay interior mask region ke atas potongan halaman, plus tetangga.

Kalau interior r6 lebih sempit dari rongga balon yang terlihat, penyebabnya
salah satu dari: (a) dipotong tetangga oleh disjoin_overlapping_interiors,
(b) bubble_bbox memotong balon, (c) garis balon dikira interior lalu dierode.
Overlay ini membedakan ketiganya dalam satu pandangan.

    IDX=6 python probe_overlay.py
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
PAD = int(os.environ.get("PAD", "40"))
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
import typeset   # noqa: E402


def main() -> int:
    typeset.setup_fonts(verbose=False)
    img = imgio.load_any(ROOT / "jepang_002.webp")
    h, w = img.shape[:2]
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    r = next(x for x in regions if x.idx == IDX)
    bx0, by0, bx1, by1 = r.bubble_bbox or r.bbox

    # Tetangga: region lain yang kotak balonnya bersinggungan dengan r ini.
    print(f"r{IDX} bubble_bbox=({bx0},{by0},{bx1},{by1}) bbox={r.bbox}")
    for o in regions:
        if o.idx == IDX:
            continue
        ob = o.bubble_bbox or o.bbox
        if ob[0] < bx1 + PAD and ob[2] > bx0 - PAD and ob[1] < by1 + PAD and ob[3] > by0 - PAD:
            print(f"   tetangga r{o.idx} {o.label:<9} bubble={o.bubble_bbox} "
                  f"bbox={o.bbox} src='{o.src_text[:18]}'")

    x0, y0 = max(bx0 - PAD, 0), max(by0 - PAD, 0)
    x1, y1 = min(bx1 + PAD, w), min(by1 + PAD, h)
    vis = img[y0:y1, x0:x1].astype(np.float32).copy()

    box, m = typeset._region_box_mask(r)
    sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
    a0, b0 = box[1] - y0, box[0] - x0
    sub[a0:a0 + m.shape[0], b0:b0 + m.shape[1]] = m
    on = sub > 0
    vis[on] = vis[on] * 0.55 + np.array([0.0, 255.0, 0.0]) * 0.45
    # Kotak bubble_bbox (merah) supaya kelihatan kalau balonnya terpotong.
    cv2.rectangle(vis, (bx0 - x0, by0 - y0), (bx1 - x0 - 1, by1 - y0 - 1),
                  (255, 0, 0), 1)
    cx, cy = typeset._centroid(m)
    cv2.line(vis, (box[0] - x0 + cx, 0), (box[0] - x0 + cx, y1 - y0),
             (0, 0, 255), 1)
    cv2.line(vis, (0, box[1] - y0 + cy), (x1 - x0, box[1] - y0 + cy),
             (0, 0, 255), 1)
    big = cv2.resize(vis.astype(np.uint8), None, fx=3, fy=3,
                     interpolation=cv2.INTER_NEAREST)
    out = ROOT / "_cmp" / f"zz_ov_r{IDX}.png"
    cv2.imwrite(str(out), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    print(f"\ncentroid=({cx},{cy}) -> {out.name} ({big.shape[1]}x{big.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
