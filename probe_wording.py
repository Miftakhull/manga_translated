#!/usr/bin/env python3
"""Margin & keterisian kalau WORDING-nya wording referensi.

Ini memisahkan dua sebab yang selama ini tercampur. Margin sisi kita 0.082
sementara referensi 0.165, dan isi baris kita 83% sementara referensi 70% —
tapi teks kita juga JAUH lebih panjang ('THIS IS A SUMMARY OF OUR ACTIVITIES
SINCE THIS SPRING, ISN'T IT?' vs 'A SUMMARY OF EVERYTHING WE'VE DONE SINCE THIS
SPRING.'). Kalau margin longgar referensi datang dari wording pendek, menaikkan
pad_ratio untuk mengejar angka itu justru salah: harganya ukuran font (cacat #3)
dan tanda hubung (cacat #4), dua cacat yang eksplisit di plan.txt, sedangkan
marginnya akan datang sendiri begitu tahap wording dikerjakan.

Jadi: skor yang sama, dijalankan dua kali — teks DeepL kita, lalu teks referensi.
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
REF_SCALE = 1577 / 1812
REF_CAP = {0: 17, 1: 20, 2: 19, 3: 26, 4: 19, 5: 27, 6: 15,
           7: 13, 8: 14, 9: 14, 10: 13, 11: 14, 12: 16}

# Wording yang benar-benar dipakai typesetter referensi (probe_pair.py).
REF_TEXT = {
    0: "AH! FINALLY FOUND YOU!",
    1: "SO THIS IS WHERE YOU WERE!",
    2: "I'VE BEEN LOOKING ALL OVER FOR YOU.",
    3: "PREZ!",
    4: "OH MY.",
    5: "SHIZUKU-SAN.",
    6: "SORRY.",
    7: "IT'S JUST THAT IT'S QUIET AND RELAXING HERE AFTER SCHOOL...",
    8: "WHAT WERE YOU DOING IN A PLACE LIKE THIS?",
    9: "I WAS PUTTING TOGETHER THE STUDENT COUNCIL'S ACTIVITY RECORDS.",
    10: "A SUMMARY OF EVERYTHING WE'VE DONE SINCE THIS SPRING.",
    11: "OH, IS THAT FOR THE MILKING CLUB?",
    12: "COME ON, LET ME SEE~!",
}


def score(regions, masks, get_text, fp) -> dict:
    ref = {i: REF_CAP[i] * REF_SCALE / CAP_PER_SIZE for i in REF_CAP}
    errs, sides, fills, hyph, over, sizes, low = [], [], [], 0, 0, [], 0
    for r in regions:
        t = get_text(r.idx)
        if not t:
            continue
        m = masks[r.idx]
        mn = min(m.shape[:2])
        cap = int(round(mn * REF_RATIO / CAP_PER_SIZE))
        size, lines, sy, ov = typeset.fit(t, m, cap, fp)
        if ov:
            over += 1
        if not lines:
            continue
        sizes.append(size)
        if size < SETTINGS.min_font_size:
            low += 1
        if any(ln.endswith("-") for ln in lines):
            hyph += 1
        if r.idx in ref:
            errs.append(abs(size - ref[r.idx]))
        font = typeset._font(fp, size)
        lh = typeset._line_height(font)
        ink_top, ink_bot = typeset._ink_band(fp, size)
        widest, avail = 0.0, 1
        for k, ln in enumerate(lines):
            lw = typeset._measure(ln, font)
            band = m[max(sy + k * lh + ink_top, 0):max(sy + k * lh + ink_bot, 1)] > 0
            cols = max(int(band.any(0).sum()) if band.size else 1, 1)
            if lw > widest:
                widest, avail = lw, cols
        sides.append((avail - widest) / 2 / mn)
        fills.append(widest / avail)
    return {"galat": float(np.mean(errs)) if errs else 99.0, "hyph": hyph,
            "luber": over, "low": low,
            "sisi": float(np.median(sides)) if sides else 0.0,
            "isi": float(np.median(fills)) if fills else 0.0,
            "min": min(sizes) if sizes else 0, "max": max(sizes) if sizes else 0}


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with PRE.open("rb") as f:
        regions = pickle.load(f)
    textmask.disjoin_overlapping_interiors(img, regions)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    masks = {r.idx: typeset._region_box_mask(r)[1] for r in regions}
    ours = lambda i: str(texts.get(str(i), "")).upper()       # noqa: E731
    refw = lambda i: REF_TEXT.get(i, "")                      # noqa: E731

    print("referensi terukur: sisi=0.165 isi=70% hyph=0 luber=0")
    for name, gt in (("teks DeepL kita", ours), ("teks REFERENSI", refw)):
        print(f"\n{name}")
        print(f"  {'ls':>5} {'pad':>5} | {'galat':>6} {'hyph':>5} {'luber':>6} "
              f"{'<min':>5} {'sisi':>6} {'isi':>5} {'ukuran':>9}")
        for ls in (1.00,):
            for pad in (0.04, 0.06, 0.08, 0.10, 0.12):
                SETTINGS.line_spacing, SETTINGS.pad_ratio = ls, pad
                s = score(regions, masks, gt, fp)
                rng = f"{s['min']}..{s['max']}"
                print(f"  {ls:>5.2f} {pad:>5.2f} | {s['galat']:>6.2f} "
                      f"{s['hyph']:>5} {s['luber']:>6} {s['low']:>5} "
                      f"{s['sisi']:>6.3f} {s['isi']*100:>4.0f}% {rng:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
