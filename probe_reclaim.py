#!/usr/bin/env python3
"""Simulasi 'lebar tersedia per BARIS' sebelum disentuhkan ke kode produksi.

Aturannya, dan tiap syaratnya ada alasan:

    reclaim_i = fill_map_i
                & (gabungan interior region LAIN sekarang)
                - (gabungan KOTAK TINTA semua region, fase 1)
                - reclaim region ber-indeks lebih kecil

  fill_map_i        interior balon region i SEBELUM disjoin (Region.fill_mask,
                    direkam build_fill_mask sebelum pemangkasan). Karena sumber
                    reclaim adalah balon MILIK SENDIRI, cacat 'keluar bubble'
                    tidak mungkin muncul dari langkah ini.
  & interior lain   piksel itu harus yang DIAMBIL tetangga, bukan piksel tepi
                    baru: fill_mask dikikis lebih tipis daripada bubble_mask
                    (fill_erode_stroke), jadi tanpa syarat ini reclaim bisa
                    memakan jarak aman ke garis balon.
  - kotak tinta     tetangga yang benar-benar MEMAKAI lebar itu tetap menang.
  - reclaim lebih   satu piksel milik satu region; tanpa ini dua region bisa
    kecil           mengklaim piksel yang sama dan hasilnya saling timpa.

Yang dicetak: reclaim tiap region, lalu fit() SEBELUM vs SESUDAH, plus dua
kontrak keras — tinta tidak saling timpa dan tinta tidak keluar dari interior
balon sendiri (fill_map).
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

import numpy as np                                     # noqa: E402
import detect, imgio, textmask, typeset                # noqa: E402,E401

REPORT = json.load(open(ROOT / "debug/jp_6/report.json", encoding="utf-8"))
OURS = {r["idx"]: r["translation"] for r in REPORT["regions"]}

fp = typeset.setup_fonts(verbose=False)
img = imgio.load_any(ROOT / "hasilnew/jp_6.JPG")
H, W = img.shape[:2]
typeset.set_page_width(W)
regions, _ = detect.detect(img)
soft = textmask.ctd_soft_mask(img)
for r in regions:
    textmask.build_region_mask(img, r, soft)
textmask.partition_shared_interiors(img, regions)
textmask.disjoin_overlapping_interiors(img, regions)
for r in regions:
    r.translation = OURS.get(r.idx, "")


def page(box, m) -> np.ndarray:
    """Mask lokal -> kanvas halaman."""
    out = np.zeros((H, W), np.uint8)
    x1, y1 = box[0], box[1]
    mh, mw = m.shape[:2]
    sy1, sx1 = max(y1, 0), max(x1, 0)
    sy2, sx2 = min(y1 + mh, H), min(x1 + mw, W)
    if sy2 > sy1 and sx2 > sx1:
        out[sy1:sy2, sx1:sx2] = m[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1]
    return out


def fit_of(box, mask, text):
    size, lines, sy, over = typeset.fit(text.upper(), mask,
                                        typeset.region_font_cap(mask), fp)
    return size, lines, sy, over


def bands_of(box, mask, size, lines, sy) -> list[tuple[int, int, int, int]]:
    """Kotak tinta tiap baris, koordinat halaman. Sama dengan render_region."""
    if not lines:
        return []
    font = typeset._font(fp, size)
    cmap = typeset._cmap(fp)
    lh = typeset._line_height(font)
    it, ib = typeset._ink_band(fp, size)
    ax = typeset.line_axis(mask, lines, sy, size, fp)
    out = []
    for k, ln in enumerate(lines):
        w = typeset._line_width(ln, font, cmap, size)
        out.append((box[1] + sy + k * lh + it, box[1] + sy + k * lh + ib + 1,
                    box[0] + int(ax - w / 2), box[0] + int(ax + w / 2) + 1))
    return out


# ---- fase 1: tata letak pada mask sekarang ----------------------------------
boxes, masks, maps, fills, phase1 = {}, {}, {}, {}, {}
for r in regions:
    b, m = typeset._region_box_mask(r)
    boxes[r.idx], masks[r.idx] = b, m
    maps[r.idx] = page(b, m)
    fills[r.idx] = (page(r.fill_bbox, r.fill_mask)
                    if r.fill_mask is not None and r.fill_bbox is not None
                    else np.zeros((H, W), np.uint8))
    phase1[r.idx] = fit_of(b, m, r.translation) if r.translation else None

ink = np.zeros((H, W), bool)
for r in regions:
    if phase1[r.idx] is None:
        continue
    s, ls, sy, _o = phase1[r.idx]
    for y0, y1, x0, x1 in bands_of(boxes[r.idx], masks[r.idx], s, ls, sy):
        ink[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True

# ---- reclaim -----------------------------------------------------------------
print("=== reclaim per region ===")
print(f"{'r':>2} {'lepas':>7} {'diambil':>8} {'reclaim':>8}  kotak baru")
taken = np.zeros((H, W), bool)
reclaim: dict[int, np.ndarray] = {}
for r in regions:
    i = r.idx
    others = np.zeros((H, W), bool)
    for q in regions:
        if q.idx != i:
            others |= maps[q.idx] > 0
    lost = (fills[i] > 0) & ~(maps[i] > 0)
    cand = lost & others & ~ink & ~taken
    reclaim[i] = cand
    taken |= cand
    nb = "-"
    if cand.any():
        ys, xs = np.nonzero((maps[i] > 0) | cand)
        nb = f"({xs.min()}, {ys.min()}, {xs.max() + 1}, {ys.max() + 1})"
    print(f"{i:>2} {int(lost.sum()):>7} {int((lost & others).sum()):>8} "
          f"{int(cand.sum()):>8}  {nb}")

# ---- fase 2: tata letak pada mask yang sudah diperluas ----------------------
print("\n=== fit() sebelum vs sesudah ===")
print(f"{'r':>2} {'size':>9} {'hyph':>7} {'luber':>7}  baris sesudah")
newbox, newmask, phase2 = {}, {}, {}
for r in regions:
    i = r.idx
    if phase1[i] is None:
        continue
    comb = (maps[i] > 0) | reclaim[i]
    ys, xs = np.nonzero(comb)
    b = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    m = np.where(comb[b[1]:b[3], b[0]:b[2]], 255, 0).astype(np.uint8)
    newbox[i], newmask[i] = b, m
    phase2[i] = fit_of(b, m, r.translation)
    s1, l1, _y1, o1 = phase1[i]
    s2, l2, _y2, o2 = phase2[i]
    h1 = sum(1 for x in l1 if x.endswith("-"))
    h2 = sum(1 for x in l2 if x.endswith("-"))
    print(f"{i:>2} {f'{s1}->{s2}':>9} {f'{h1}->{h2}':>7} "
          f"{f'{int(o1)}->{int(o2)}':>7}  {' | '.join(l2)}")

# ---- kontrak keras ----------------------------------------------------------
print("\n=== kontrak ===")
ink2 = {}
for r in regions:
    i = r.idx
    if i not in phase2:
        continue
    s, ls, sy, _o = phase2[i]
    mm = np.zeros((H, W), bool)
    for y0, y1, x0, x1 in bands_of(newbox[i], newmask[i], s, ls, sy):
        mm[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    ink2[i] = mm & ((maps[i] > 0) | reclaim[i])   # seperti _clip_to_mask

clash = 0
for a in ink2:
    for b in ink2:
        if a < b:
            clash += int((ink2[a] & ink2[b]).sum())
print(f"tinta saling timpa           : {clash} px")
out_own = {i: int((ink2[i] & ~(fills[i] > 0) & ~(maps[i] > 0)).sum()) for i in ink2}
print(f"tinta di luar balon sendiri  : {out_own}")
ov = {i: int(phase2[i][3]) for i in phase2}
print(f"luber                        : {ov}")
