#!/usr/bin/env python3
"""Jarak antar-baris referensi vs milik kita, dan sweep line_spacing.

probe_refnative.py sudah memberi tinggi kapital dan margin sisi referensi.
Yang belum: JARAK BARIS. Itu penentu berapa banyak baris muat di balon, jadi
penentu ukuran font — dan kandidat utama kenapa lima region kita tidak bisa
memakai ukuran target.

Bagian 1: pitch baris referensi (dari profil baris tinta) dibagi tinggi kapital.
Bagian 2: sweep SETTINGS.line_spacing pada mask kita, laporkan ukuran layak.
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
REFC = ROOT / ".probe_ref_native.pkl"
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

import cv2       # noqa: E402
import imgio     # noqa: E402
import textmask  # noqa: E402
import typeset   # noqa: E402
from config import SETTINGS  # noqa: E402


def ref_pitch() -> None:
    img = imgio.load_any(ROOT / "CONTOH" / "2.webp")
    h, w = img.shape[:2]
    with REFC.open("rb") as f:
        regions = pickle.load(f)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(f"REFERENSI {w}x{h} — pitch baris terukur")
    print(f"  {'idx':>3} {'baris':>5} {'cap':>5} {'pitch':>6} {'pitch/cap':>10}")
    rows = []
    for r in regions:
        box, mask = typeset._region_box_mask(r)
        bx1, by1, bx2, by2 = box
        mh, mw = mask.shape[:2]
        y2, x2 = min(by1 + mh, h), min(bx1 + mw, w)
        if y2 - by1 < 10 or x2 - bx1 < 10:
            continue
        m = mask[: y2 - by1, : x2 - bx1] > 0
        ink = (m & (gray[by1:y2, bx1:x2] < 110)).astype(np.uint8)
        if int(ink.sum()) < 40:
            continue
        prof = ink.any(1).astype(np.int8)
        starts = [i for i in range(len(prof)) if prof[i] and (i == 0 or not prof[i - 1])]
        if len(starts) < 3:
            continue
        pitch = float(np.median(np.diff(starts)))
        n, _l, st, _ = cv2.connectedComponentsWithStats(ink, 8)
        hs = [st[i][3] for i in range(1, n)
              if 4 <= st[i][3] <= 60 and 2 <= st[i][2] <= 60 and st[i][4] >= 6]
        if len(hs) < 4:
            continue
        cap = float(np.median(hs))
        rows.append(pitch / cap)
        print(f"  {r.idx:>3} {len(starts):>5} {cap:>5.1f} {pitch:>6.1f} "
              f"{pitch/cap:>10.2f}")
    print(f"\npitch/cap referensi: median={np.median(rows):.2f} "
          f"p25={np.percentile(rows,25):.2f} p75={np.percentile(rows,75):.2f}")
    f13 = typeset._font(typeset.FONT_USED, 13)
    asc, desc = f13.getmetrics()
    cap13 = 11.0
    print(f"kita: line_spacing={SETTINGS.line_spacing} -> lh={typeset._line_height(f13)} "
          f"px pada size 13 (cap {cap13:.0f}) = pitch/cap "
          f"{typeset._line_height(f13)/cap13:.2f}")
    print(f"      line_spacing yang menyamai referensi ~= "
          f"{np.median(rows) * cap13 / (asc + desc):.2f}")


def sweep_ours() -> None:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    with PRE.open("rb") as f:
        regions = pickle.load(f)
    textmask.disjoin_overlapping_interiors(img, regions)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    fp = typeset.FONT_USED
    masks = {r.idx: typeset._region_box_mask(r)[1] for r in regions}

    keep_ls, keep_pr = SETTINGS.line_spacing, SETTINGS.pad_ratio
    for ls in (1.28, 1.15, 1.05, 1.00):
        SETTINGS.line_spacing = ls
        for pr in (0.10, 0.05):
            SETTINGS.pad_ratio = pr
            out = []
            for r in regions:
                t = str(texts.get(str(r.idx), "")).upper()
                m = masks[r.idx]
                mh, mw = m.shape[:2]
                pad = int(min(mh, mw) * pr)
                hi = int(np.clip(mh - 2 * pad, SETTINGS.min_font_size,
                                 SETTINGS.max_font_size))
                p = typeset._search(t, m, SETTINGS.min_font_size, hi, fp, False)
                hy = typeset._search(t, m, SETTINGS.min_font_size, hi, fp, True)
                out.append(max(p[0] if p else 0, hy[0] if hy else 0))
            nz = [v for v in out if v]
            print(f"  line_spacing={ls:<5} pad_ratio={pr:<5} "
                  f"nol={sum(1 for v in out if not v)} "
                  f"p35={int(np.percentile(nz, 35)) if nz else 0:<3} "
                  f"min={min(nz) if nz else 0:<3} max={max(nz) if nz else 0:<3} "
                  f"{out}")
    SETTINGS.line_spacing, SETTINGS.pad_ratio = keep_ls, keep_pr


def main() -> int:
    typeset.setup_fonts(verbose=False)
    ref_pitch()
    print("\nKITA — ukuran layak (utuh atau ber-hyphen) per region")
    sweep_ours()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
