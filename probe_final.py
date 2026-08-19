#!/usr/bin/env python3
"""Apa yang SUNGGUH keluar dari fit() per region: ukuran, baris, hyphen, luber.

Tabel probe_cal.py hanya menghitung agregat. Sebelum memilih pad_ratio, harus
jelas region MANA yang membayar harganya — sebuah tanda hubung di balon sempit
yang memang mustahil (terpaksa, sesuai keputusan user) beda maknanya dengan
tanda hubung di balon lapang (tidak terpaksa, itu cacat #4).

Sekaligus memeriksa jalur degradasi: sejak layout() bisa menggeser anchor,
apakah region 6 masih jatuh ke cabang di bawah min_font_size, dan kalau ya,
ukuran berapa yang keluar.
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

CAP_PER_SIZE, REF_RATIO = 0.844, 0.117


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with PRE.open("rb") as f:
        regions = pickle.load(f)
    textmask.disjoin_overlapping_interiors(img, regions)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    masks = {r.idx: typeset._region_box_mask(r)[1] for r in regions}
    SETTINGS.line_spacing = float(os.environ.get("LS", 1.00))
    SETTINGS.pad_ratio = float(os.environ.get("PAD", 0.06))
    print(f"line_spacing={SETTINGS.line_spacing} pad_ratio={SETTINGS.pad_ratio} "
          f"min_font_size={SETTINGS.min_font_size}")
    for r in regions:
        t = str(texts.get(str(r.idx), "")).upper()
        m = masks[r.idx]
        mn = min(m.shape[:2])
        cap = int(round(mn * REF_RATIO / CAP_PER_SIZE))
        size, lines, _sy, ov = typeset.fit(t, m, cap, fp)
        tag = []
        if any(ln.endswith("-") for ln in lines):
            tag.append("HYPHEN")
        if ov:
            tag.append("LUBER")
        if size < SETTINGS.min_font_size:
            tag.append("DI-BAWAH-MIN")
        print(f"  r{r.idx:<2} min={mn:>3} plafon={cap:>2} -> size={size:>2} "
              f"baris={len(lines)} {' '.join(tag)}")
        if tag:
            for ln in lines:
                print(f"        {ln!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
