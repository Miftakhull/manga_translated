#!/usr/bin/env python3
"""Kenapa reclaim produksi TIDAK menolong r3, padahal simulasi bilang menolong.

Bedanya satu hal, dan itu memang disengaja: probe_reclaim.py hanya MENAMBAH
(comb = maps | reclaim), sedangkan produksi MEMINDAH (penerima menambah, pelepas
mengurangi). Konsekuensinya urutan idx jadi penentu: r2 mengklaim dari interior
r3 sebelum r3 dapat giliran, jadi r3 bisa kehilangan justru lebar yang
dibutuhkannya.

Yang dicetak, per region: piksel sebelum/sesudah, yang DIDAPAT, yang DILEPAS,
lalu fit() sebelum/sesudah. Ditambah potret baris-demi-baris r3: lebar bebas
interiornya pada tiap ketinggian, sebelum vs sesudah reclaim, supaya jelas
apakah lebar tambahannya jatuh di band yang dipakai 'WONDER' atau tidak.

Pakai .probe_cache6.pkl (dibuat probe_final6.py) -> tanpa detect, hitungan detik.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NBSRC, STAGE = ROOT / "_nbsrc", ROOT / ".stage"
CACHE = ROOT / ".probe_cache6.pkl"
PAGE = ROOT / "hasilnew" / "jp_6.JPG"
REPORT = ROOT / "debug" / "jp_6" / "report.json"
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))
STAGE.mkdir(exist_ok=True)
for _s in sorted(NBSRC.glob("*.py")):
    _b = _MAGIC.sub("", _s.read_text(encoding="utf-8"), count=1)
    _d = STAGE / _s.name
    if not _d.exists() or _d.read_text(encoding="utf-8") != _b:
        _d.write_text(_b, encoding="utf-8")
sys.path.insert(0, str(STAGE))

import numpy as np                                          # noqa: E402
import imgio, typeset                                       # noqa: E402,E401

FOCUS = int(os.environ.get("FOCUS", 3))

fp = typeset.setup_fonts(verbose=False)
img = imgio.load_any(PAGE)
H, W = img.shape[:2]
typeset.set_page_width(W)
with CACHE.open("rb") as f:
    regions = pickle.load(f)
texts = {r["idx"]: r["translation"] for r in
         json.loads(REPORT.read_text(encoding="utf-8"))["regions"]}
for r in regions:
    r.translation = texts.get(r.idx) or ""


def snap() -> dict[int, np.ndarray]:
    return {r.idx: typeset._paste_mask(*typeset._region_box_mask(r), H, W) > 0
            for r in regions}


def fit_of(r):
    m = typeset._region_box_mask(r)[1]
    t = r.translation.upper()
    return typeset.fit(t, m, typeset.region_font_cap(m), fp)


before = snap()
fit0 = {r.idx: fit_of(r) for r in regions if r.translation}
box0 = {r.idx: typeset._region_box_mask(r)[0] for r in regions}
mask0 = {r.idx: typeset._region_box_mask(r)[1].copy() for r in regions}

moved = typeset.reclaim_unused_interiors(img, regions)
after = snap()
fit1 = {r.idx: fit_of(r) for r in regions if r.translation}

print(f"reclaim mengubah {moved} region\n")
print(f"{'r':>2} {'sebelum':>8} {'sesudah':>8} {'dapat':>7} {'lepas':>7} "
      f"{'size':>7} {'hyph':>7}  baris sesudah")
for r in sorted(regions, key=lambda q: q.idx):
    i = r.idx
    got = int((after[i] & ~before[i]).sum())
    lost = int((before[i] & ~after[i]).sum())
    if i in fit0:
        s0, l0, _y0, _o0 = fit0[i]
        s1, l1, _y1, _o1 = fit1[i]
        h0 = sum(1 for x in l0 if x.endswith("-"))
        h1 = sum(1 for x in l1 if x.endswith("-"))
        sz, hy = f"{s0}->{s1}", f"{h0}->{h1}"
        txt = " | ".join(l1)
    else:
        sz, hy, txt = "-", "-", "(tanpa terjemahan)"
    print(f"{i:>2} {int(before[i].sum()):>8} {int(after[i].sum()):>8} "
          f"{got:>7} {lost:>7} {sz:>7} {hy:>7}  {txt}")

# ---- potret baris r-FOCUS ----------------------------------------------------
tgt = next((r for r in regions if r.idx == FOCUS), None)
if tgt is not None:
    b1, m1 = typeset._region_box_mask(tgt)
    b0, m0 = box0[FOCUS], mask0[FOCUS]
    s1, l1, y1, _o = fit1[FOCUS]
    font = typeset._font(fp, s1)
    cmap = typeset._cmap(fp)
    need = {ln: typeset._line_width(ln, font, cmap, s1) for ln in l1}
    full = typeset._line_width("WONDER", font, cmap, s1)
    print(f"\n=== r{FOCUS} kotak {b0} -> {b1}   size={s1} "
          f"'WONDER' butuh {full:.1f} px")
    print(f"{'y':>5} {'bebas0':>7} {'bebas1':>7}  keterangan")
    ys = sorted({b0[1] + k for k in range(m0.shape[0])}
                | {b1[1] + k for k in range(m1.shape[0])})
    for y in range(min(ys), max(ys) + 1, 4):
        r0 = int((m0[y - b0[1]] > 0).sum()) if 0 <= y - b0[1] < m0.shape[0] else 0
        r1 = int((m1[y - b1[1]] > 0).sum()) if 0 <= y - b1[1] < m1.shape[0] else 0
        note = "cukup untuk WONDER" if r1 >= full else ""
        print(f"{y:>5} {r0:>7} {r1:>7}  {note}")
