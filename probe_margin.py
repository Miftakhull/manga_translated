#!/usr/bin/env python3
"""Margin horizontal build() vs yang benar-benar ditegakkan _verify().

Ditemukan lewat r12 yang timpang 21 px pada wording baru. Sebabnya BUKAN
penyeimbang: pemecahan 4 baris ['IS IT?','LEMME','SEE,',"C'MON~!"] muat di
y=110 dengan ketimpangan 0, tapi tidak pernah diusulkan. Dua rem berturut-turut:

  1. n melompat 3 -> 5. `nxt = len(lines) + 1` mengambil jumlah baris hasil build
     di y terpusat (4) lalu menambah 1, jadi n=4 TIDAK PERNAH dicoba.
  2. Walau n=4 dicoba, build() memotong lebar tiap baris sebesar pad*2 px
     sementara _verify() tidak memotong apa pun. Baris terakhir butuh 74 px dari
     77.9 px bebas — sah menurut _verify, ditolak build. Jadi layout yang legal
     tidak pernah lahir.

Probe ini mengukur akibat mengubah margin itu untuk SEMUA region, karena
melonggarkan margin berarti teks boleh lebih dekat ke garis balon — dan itu
salah satu cacat yang justru sedang diberantas. Yang dicetak per varian:
ukuran font, jumlah baris, ketimpangan atas-bawah, dan jarak tinta terdekat ke
tepi interior (px). Varian dipilih dari angka, bukan dari selera.

    TEXTS=probe_llm2_seekai-claude-opus-5.json python probe_margin.py
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
TEXTS = ROOT / os.environ.get("TEXTS", "probe_llm2_seekai-claude-opus-5.json")
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
import typeset   # noqa: E402
from config import SETTINGS  # noqa: E402

# (nama, margin_horizontal, langkah_n)
VARIANTS = (
    ("sekarang  pad*2, n lompat", 2.0, False),
    ("A  pad*2, n +1", 2.0, True),
    ("B  pad*1, n +1", 1.0, True),
    ("C  pad*0, n +1", 0.0, True),
)


def sim(mask: np.ndarray, text: str, size: int, fp: str,
        mar_mult: float, step_one: bool):
    """layout() disalin apa adanya kecuali dua hal yang sedang diuji."""
    font = typeset._font(fp, size)
    words = text.split()
    if not words:
        return None
    lh = typeset._line_height(font)
    it, ib = typeset._ink_band(fp, size)
    cx, cy = typeset._centroid(mask)
    mh, mw = mask.shape[:2]
    pad = int(min(mh, mw) * SETTINGS.pad_ratio)
    mar = pad * mar_mult

    def width_at(y: int) -> float:
        lo, hi = 0.0, float(mw)
        for _ in range(9):
            mid = (lo + hi) / 2
            if typeset._row_free(mask, y + it, y + ib, cx - mid / 2, cx + mid / 2):
                lo = mid
            else:
                hi = mid
        return max(lo - mar, 0.0)

    def build(top: int):
        lines: list[str] = []
        q = list(words)
        i = 0
        for _ in range(64):
            if i >= len(q):
                return lines, True
            av = width_at(top + len(lines) * lh)
            if av < size * 0.9:
                return lines, False
            ln = q[i]
            j = i + 1
            while j < len(q) and typeset._measure(f"{ln} {q[j]}", font) <= av:
                ln = f"{ln} {q[j]}"
                j += 1
            if j == i + 1 and typeset._measure(ln, font) > av:
                return lines, False
            lines.append(ln)
            i = j
        return lines, i >= len(q)

    def center_y(n: int) -> int:
        return cy - ((n - 1) * lh + (ib - it)) // 2 - it

    def tops(n: int) -> list[int]:
        ink_h = (n - 1) * lh + (ib - it)
        lo, hi = pad - it, mh - pad - ink_h - it
        if hi < lo:
            return [center_y(n)]
        c = [int(np.clip(center_y(n), lo, hi))]
        st = max(lh // 2, 2)
        return c + [t for t in range(lo, hi + 1, st) if t != c[0]]

    def verify(ls: list[str], top: int) -> bool:
        for k, l in enumerate(ls):
            w = typeset._measure(l, font)
            y = top + k * lh
            if not typeset._row_free(mask, y + it, y + ib, cx - w / 2, cx + w / 2):
                return False
        return not (top + it < pad
                    or top + (len(ls) - 1) * lh + ib > mh - pad)

    def slack(ls: list[str], top: int):
        return typeset.block_slack(
            mask, cx, pad, typeset._measure(ls[0], font),
            typeset._measure(ls[-1], font),
            top + it, top + (len(ls) - 1) * lh + ib)

    n = max(1, int(np.ceil(typeset._measure(text, font) / max(mw - pad * 2, 1))))
    tol = max(2, lh // 2)
    for _ in range(6):
        hit = None
        first = None
        for top in tops(n):
            cand, ok = build(top)
            if first is None:
                first = (cand, ok)
            if not (ok and len(cand) == n and verify(cand, top)):
                continue
            up, dn = slack(cand, top)
            bal = abs(up - dn)
            if hit is None or bal < hit[0]:
                hit = (bal, top, cand)
            if bal <= tol:
                break
        if hit is not None:
            bal, top, cand = hit
            up, dn = slack(cand, top)
            bt, bb = top, bal
            stp = 1 if dn > up else -1
            for _ in range(abs(dn - up) // 2):
                t = bt + stp
                if not verify(cand, t):
                    break
                u2, d2 = slack(cand, t)
                if abs(u2 - d2) >= bb:
                    break
                bt, bb = t, abs(u2 - d2)
            u, d = slack(cand, bt)
            return cand, bt, u, d, lh, it, ib, cx, font
        cand, ok = first
        nxt = max(1, len(cand) + (0 if ok else 1))
        n = (nxt if nxt < n else n + 1) if step_one else (n + 1 if nxt == n else nxt)
    return None


def edge_gap(mask: np.ndarray, lines, top, lh, it, ib, cx, font) -> int:
    """Jarak px terdekat dari tinta ke tepi interior. 0 = menempel garis."""
    inner = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(inner, cv2.DIST_L2, 3)
    mh, mw = mask.shape[:2]
    best = 10**6
    for k, ln in enumerate(lines):
        w = typeset._measure(ln, font)
        y0, y1 = top + k * lh + it, top + k * lh + ib
        x0, x1 = int(cx - w / 2), int(np.ceil(cx + w / 2))
        y0, y1 = max(y0, 0), min(y1, mh)
        x0, x1 = max(x0, 0), min(x1, mw)
        if y1 <= y0 or x1 <= x0:
            return 0
        box = dist[y0:y1, x0:x1]
        # Tepi kotak baris itulah yang paling dekat ke garis; ambil minimum di
        # perimeter, bukan di seluruh kotak (bagian tengah selalu jauh).
        peri = np.concatenate([box[0], box[-1], box[:, 0], box[:, -1]])
        best = min(best, int(peri.min()))
    return best


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))

    for name, mult, step in VARIANTS:
        print(f"\n=== {name}")
        print(f"  {'idx':>3} {'size':>4} {'nb':>3} {'atas':>5} {'bawah':>5} "
              f"{'timpang':>7} {'jarak_tepi':>10}  baris")
        bals, gaps = [], []
        for r in regions:
            t = str(texts.get(str(r.idx), "")).upper()
            if not t:
                continue
            m = typeset._region_box_mask(r)[1]
            cap = typeset.region_font_cap(m)
            got = None
            # fit() dari besar ke kecil, ambil yang pertama muat — sama seperti
            # _search(); di sini cukup ukuran final yang dipakai pipeline.
            for size in range(cap, SETTINGS.min_font_size - 1, -1):
                got = sim(m, t, size, fp, mult, step)
                if got:
                    break
            if not got:
                print(f"  {r.idx:>3}  TIDAK MUAT di ukuran mana pun")
                continue
            lines, top, up, dn, lh, it, ib, cx, font = got
            gap = edge_gap(m, lines, top, lh, it, ib, cx, font)
            bals.append(abs(up - dn))
            gaps.append(gap)
            print(f"  {r.idx:>3} {size:>4} {len(lines):>3} {up:>5} {dn:>5} "
                  f"{abs(up - dn):>7} {gap:>10}  {lines}")
        print(f"  timpang: median={np.median(bals):.0f} maks={max(bals)}  "
              f"jarak_tepi: min={min(gaps)} median={np.median(gaps):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
