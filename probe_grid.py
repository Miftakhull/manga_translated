#!/usr/bin/env python3
"""Cari (line_spacing, pad_ratio) yang paling mendekati typeset referensi.

Angka acuan diukur di frame referensi sendiri (probe_refnative.py, probe_lines.py):
  cap_height / min(sisi interior) = 0.117   -> seberapa besar font relatif balon
  pitch / cap_height              = 1.36    -> jarak baris
  margin sisi / min(sisi)         = 0.165

Untuk tiap kombinasi dilaporkan: berapa region yang tidak dapat ukuran sama
sekali, sebaran (max/min), dan simpangan rata-rata cap/min dari 0.117 — jadi
pilihannya berdasar ukuran, bukan selera.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / ".stage"
PRE = ROOT / ".probe_pre.pkl"
TEXTS = ROOT / "probe_font_texts.json"
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

import imgio     # noqa: E402
import textmask  # noqa: E402
import typeset   # noqa: E402
from config import SETTINGS  # noqa: E402

REF_CAP_RATIO = 0.117    # cap_height / min(mh, mw) di referensi
CAP_PER_SIZE = 0.844     # cap_height / ukuran font, Anime Ace (probe_cap.py)


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with PRE.open("rb") as f:
        regions = pickle.load(f)
    textmask.disjoin_overlapping_interiors(img, regions)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    masks = {r.idx: typeset._region_box_mask(r)[1] for r in regions}
    want = {}
    for r in regions:
        mh, mw = masks[r.idx].shape[:2]
        want[r.idx] = min(mh, mw) * REF_CAP_RATIO / CAP_PER_SIZE
    print("ukuran font yang setara referensi, per region:")
    print("   " + "  ".join(f"{r.idx}:{want[r.idx]:.0f}" for r in regions))

    keep = (SETTINGS.line_spacing, SETTINGS.pad_ratio)
    print(f"\n  {'ls':>5} {'pad':>5} {'nol':>4} {'p35':>4} {'spread':>7} "
          f"{'|cap-ref|':>10}  ukuran per region")
    best = None
    for ls in (1.28, 1.10, 1.00, 0.96, 0.90):
        for pr in (0.10, 0.08, 0.06, 0.04):
            SETTINGS.line_spacing, SETTINGS.pad_ratio = ls, pr
            feas = {}
            for r in regions:
                t = str(texts.get(str(r.idx), "")).upper()
                m = masks[r.idx]
                mh, mw = m.shape[:2]
                pad = int(min(mh, mw) * pr)
                hi = int(np.clip(mh - 2 * pad, SETTINGS.min_font_size,
                                 SETTINGS.max_font_size))
                p = typeset._search(t, m, SETTINGS.min_font_size, hi, fp, False)
                hy = typeset._search(t, m, SETTINGS.min_font_size, hi, fp, True)
                feas[r.idx] = max(p[0] if p else 0, hy[0] if hy else 0)
            nz = [v for v in feas.values() if v]
            if not nz:
                continue
            tgt = int(np.clip(round(np.percentile(nz, 35)),
                              SETTINGS.min_font_size, SETTINGS.max_font_size))
            # Ukuran yang benar-benar dipakai: target kalau muat, kalau tidak
            # turun ke ukuran layak region itu.
            used = {i: (tgt if v >= tgt else v) for i, v in feas.items() if v}
            sp = max(used.values()) / min(used.values())
            dev = float(np.mean([abs(used[i] - want[i]) for i in used]))
            zeros = sum(1 for v in feas.values() if not v)
            print(f"  {ls:>5} {pr:>5} {zeros:>4} {tgt:>4} {sp:>7.2f} {dev:>10.2f}"
                  f"  {[feas[r.idx] for r in regions]}")
            score = (zeros, round(sp, 2), round(dev, 2))
            if best is None or score < best[0]:
                best = (score, ls, pr, tgt)
    SETTINGS.line_spacing, SETTINGS.pad_ratio = keep
    if best:
        print(f"\nterbaik: line_spacing={best[1]} pad_ratio={best[2]} "
              f"target={best[3]} (nol={best[0][0]} spread={best[0][1]} "
              f"simpangan={best[0][2]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
