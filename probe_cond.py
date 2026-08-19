#!/usr/bin/env python3
"""Berapa condense horizontal yang dibutuhkan supaya jp_6 nol tanda hubung?

Diuji offline, nol token: mask asli halaman jp_6 dibangun ulang, lalu untuk tiap
kandidat faktor condense diukur (a) ukuran font hasil fit(), (b) jumlah baris
ber-tanda-hubung, memakai DUA wording — terjemahan yang sudah ada di
report.json dan wording typeset referensi CONTOH/6.JPG.

Condense disimulasikan dengan mengecilkan lebar advance: itu tepat yang akan
dilakukan implementasinya (_measure dikali faktor, tile digambar lalu diperkecil
horizontal), jadi angka di sini langsung berlaku.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NBSRC, STAGE = ROOT / "_nbsrc", ROOT / ".stage"
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))
STAGE.mkdir(exist_ok=True)
for s in sorted(NBSRC.glob("*.py")):
    body = _MAGIC.sub("", s.read_text(encoding="utf-8"), count=1)
    d = STAGE / s.name
    if not d.exists() or d.read_text(encoding="utf-8") != body:
        d.write_text(body, encoding="utf-8")
sys.path.insert(0, str(STAGE))

import numpy as np                                  # noqa: E402
import detect, imgio, textmask, typeset             # noqa: E402,E401

REPORT = json.load(open(ROOT / "debug/jp_6/report.json", encoding="utf-8"))
OURS = {r["idx"]: r["translation"] for r in REPORT["regions"]}
# Wording typeset referensi CONTOH/6.JPG, dibaca dari zoom (urut idx pipeline).
REF = {
    0: "AND IN THE END...",
    1: "THEY ALL GOT TOO EXCITED AND MOSTLY CAME BY THEMSELVES!♥",
    2: "WELL, THEY'D NEVER SEEN ANYONE AS INSANELY SEXY AS YOU,",
    3: "SO IT'S ONLY NATURAL, RIGHT?♥",
    4: "IT'S EMBARASSING TO BE DESCRIBED THAT WAY...",
    5: "PLEASE DON'T TEASE ME SO MUCH...",
    6: "EHHH~? I'M PRAISING YOU FROM THE BOTTOM OF MY HEART, THOUGH?",
    7: "AH!",
}

fp = typeset.setup_fonts(verbose=False)
img = imgio.load_any(ROOT / "hasilnew/jp_6.JPG")
typeset.set_page_width(img.shape[1])
regions, _ = detect.detect(img)
soft = textmask.ctd_soft_mask(img)
for r in regions:
    textmask.build_region_mask(img, r, soft)
textmask.partition_shared_interiors(img, regions)
textmask.disjoin_overlapping_interiors(img, regions)
masks = {r.idx: typeset._region_box_mask(r)[1] for r in regions}

print(f"halaman {img.shape[1]}x{img.shape[0]}  min_font()={typeset.min_font()} "
      f"darurat={typeset.emergency_floor()}")
print("\nlebar interior EFEKTIF per region (kolom bebas terlebar di band tengah):")
for idx in sorted(masks):
    m = masks[idx]
    run = typeset._band_run(m, m.shape[0] // 2 - 3, m.shape[0] // 2 + 3)
    print(f"  r{idx}: mask {m.shape[1]}x{m.shape[0]}  band tengah bebas = "
          f"{'None' if run is None else run[1]-run[0]+1}  "
          f"cap={typeset.region_font_cap(m)}")

_orig = typeset._measure


def patch(factor: float) -> None:
    typeset._measure = (lambda t, f, _c=factor: _orig(t, f) * _c)


def run(label: str, texts: dict[int, str]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'cond':>5} {'hyphen':>7} {'size min/med/max':>17} {'luber':>6}  detail")
    for c in (1.00, 0.92, 0.88, 0.85, 0.82, 0.78, 0.72):
        patch(c)
        sizes, hy, ov, det = [], 0, 0, []
        for idx in sorted(masks):
            t = texts.get(idx, "").upper()
            if not t:
                continue
            size, lines, _y, over = typeset.fit(t, masks[idx],
                                                typeset.region_font_cap(masks[idx]), fp)
            sizes.append(size)
            ov += int(over)
            bad = [ln for ln in lines if ln.endswith("-")]
            hy += len(bad)
            if bad:
                det.append(f"r{idx}:{'/'.join(bad)}")
        print(f"{c:>5.2f} {hy:>7} {f'{min(sizes)}/{int(np.median(sizes))}/{max(sizes)}':>17} "
              f"{ov:>6}  {' '.join(det)}")


run("terjemahan kita sekarang (report.json)", OURS)
run("wording typeset referensi CONTOH/6.JPG", REF)
typeset._measure = _orig
