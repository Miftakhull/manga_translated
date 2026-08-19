#!/usr/bin/env python3
"""Sumbu tengah horizontal: centroid global vs pusat rongga blok itu sendiri.

Kenapa 'SORRY.' (r6) melorot 49 px di bawah tengah balon. Bukan penyeimbang,
bukan margin, bukan ukuran font — tapi SUMBU X.

layout() memakai satu cx = _centroid(mask) untuk semua baris, dan max_width_at()
melebarkan pita secara SIMETRIS di sekitar cx itu. Interior r6 sudah dipotong
tetangganya (r7) sehingga bentuknya pita melengkung yang pusat rongganya
BERGESER seiring y: c=27 di atas, c=41 di tengah, c=32 di bawah, sementara
centroid global cx=36. Di ketinggian tengah rongga membentang x=14..67 (54 px)
tapi probe simetris di cx=36 hanya mengakui 2*(36-14)=44 px, dikurangi pad*2
jadi 40 px — 'SORRY.' butuh 49 px, jadi GAGAL. Satu-satunya y yang lolos ada di
y=121..139, tempat rongga cukup lebar walau diukur simetris di cx yang salah.
Persis 49 px di bawah tengah.

Varian X memakai cx per BLOK: pusat rongga terpanjang yang bebas di SEMUA baris
yang ditempati blok itu. Untuk balon oval biasa angkanya sama dengan centroid
(rongga simetris), jadi yang berubah hanya balon yang terpotong.

Yang dicetak: ukuran, jumlah baris, ketimpangan atas-bawah, galat pusat
vertikal (tengah tinta - cy; inilah yang dikeluhkan user), dan jarak tinta
terdekat ke garis balon (edge_gap dari probe_margin.py — melonggarkan apa pun
tidak boleh menggerus angka ini).

    TEXTS=probe_opus5_clean.json python probe_blockcx.py
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
TEXTS = ROOT / os.environ.get("TEXTS", "probe_opus5_clean.json")
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
from config import SETTINGS  # noqa: E402
from probe_margin import edge_gap  # noqa: E402

_COVER = typeset._ROW_COVER


def blk_cx(mask: np.ndarray, y_a: int, y_b: int, fb: int) -> int:
    """Pusat rentang kolom yang bebas di SEMUA baris y_a..y_b; fb kalau nihil."""
    mh, mw = mask.shape[:2]
    a, b = max(int(y_a), 0), min(int(y_b), mh)
    if b <= a:
        return fb
    ok = (mask[a:b] >= 200).all(0)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return fb
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], cuts + 1))
    ends = np.concatenate((cuts, [idx.size - 1]))
    best = max(zip(starts, ends), key=lambda se: idx[se[1]] - idx[se[0]])
    return int((idx[best[0]] + idx[best[1]]) // 2)


def sim(mask: np.ndarray, text: str, size: int, fp: str, per_block: bool):
    """layout() disalin apa adanya kecuali sumbu x-nya."""
    font = typeset._font(fp, size)
    words = text.split()
    if not words:
        return None
    lh = typeset._line_height(font)
    it, ib = typeset._ink_band(fp, size)
    cx0, cy = typeset._centroid(mask)
    mh, mw = mask.shape[:2]
    pad = int(min(mh, mw) * SETTINGS.pad_ratio)

    def axis(top: int, n: int) -> int:
        if not per_block:
            return cx0
        return blk_cx(mask, top + it, top + (n - 1) * lh + ib, cx0)

    def width_at(y: int, cx: int) -> float:
        lo, hi = 0.0, float(mw)
        for _ in range(9):
            mid = (lo + hi) / 2
            if typeset._row_free(mask, y + it, y + ib, cx - mid / 2, cx + mid / 2):
                lo = mid
            else:
                hi = mid
        return max(lo - pad * 2, 0.0)

    def build(top: int, cx: int):
        lines: list[str] = []
        q = list(words)
        i = 0
        for _ in range(64):
            if i >= len(q):
                return lines, True
            av = width_at(top + len(lines) * lh, cx)
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

    def verify(ls: list[str], top: int, cx: int) -> bool:
        for k, ln in enumerate(ls):
            w = typeset._measure(ln, font)
            y = top + k * lh
            if not typeset._row_free(mask, y + it, y + ib, cx - w / 2, cx + w / 2):
                return False
        return not (top + it < pad or top + (len(ls) - 1) * lh + ib > mh - pad)

    def slack(ls: list[str], top: int, cx: int):
        return typeset.block_slack(
            mask, cx, pad, typeset._measure(ls[0], font),
            typeset._measure(ls[-1], font),
            top + it, top + (len(ls) - 1) * lh + ib)

    n0 = max(1, int(np.ceil(typeset._measure(text, font) / max(mw - pad * 2, 1))))
    tol = max(2, lh // 2)
    hit = None
    for n in range(n0, n0 + 5):
        for top in tops(n):
            cx = axis(top, n)
            cand, ok = build(top, cx)
            if not (ok and len(cand) == n and verify(cand, top, cx)):
                continue
            up, dn = slack(cand, top, cx)
            bal = abs(up - dn)
            if hit is None or bal < hit[0]:
                hit = (bal, top, cand, cx)
            if bal <= tol:
                break
        if hit is not None and hit[0] <= tol:
            break
    if hit is None:
        return None
    bal, top, cand, cx = hit
    up, dn = slack(cand, top, cx)
    bt, bb = top, bal
    stp = 1 if dn > up else -1
    for _ in range(abs(dn - up) // 2):
        t = bt + stp
        c2 = axis(t, len(cand))
        if not verify(cand, t, c2):
            break
        u2, d2 = slack(cand, t, c2)
        if abs(u2 - d2) >= bb:
            break
        bt, bb, cx = t, abs(u2 - d2), c2
    u, d = slack(cand, bt, cx)
    return cand, bt, cx, u, d, lh, it, ib, font, cy


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))

    for name, per_block in (("sekarang  cx = centroid global", False),
                            ("X  cx = pusat rongga blok", True)):
        print(f"\n=== {name}")
        print(f"  {'idx':>3} {'size':>4} {'nb':>3} {'cx':>3} {'dx':>4} "
              f"{'atas':>5} {'bawah':>5} {'timpang':>7} {'pusat':>6} "
              f"{'tepi':>4}  baris")
        bals, errs, gaps = [], [], []
        for r in regions:
            t = str(texts.get(str(r.idx), "")).upper()
            if not t:
                continue
            m = typeset._region_box_mask(r)[1]
            cap = typeset.region_font_cap(m)
            cx0 = typeset._centroid(m)[0]
            got = None
            for size in range(cap, SETTINGS.min_font_size - 1, -1):
                got = sim(m, t, size, fp, per_block)
                if got:
                    break
            if not got:
                print(f"  {r.idx:>3}  TIDAK MUAT di ukuran mana pun")
                continue
            lines, top, cx, up, dn, lh, it, ib, font, cy = got
            gap = edge_gap(m, lines, top, lh, it, ib, cx, font)
            # Galat pusat vertikal: tengah tinta blok vs centroid mask. Inilah
            # yang dilihat mata sebagai "di tengah antara atas dan bawah".
            mid = top + it + ((len(lines) - 1) * lh + (ib - it)) // 2
            err = mid - cy
            bals.append(abs(up - dn))
            errs.append(abs(err))
            gaps.append(gap)
            print(f"  {r.idx:>3} {size:>4} {len(lines):>3} {cx:>3} {cx - cx0:>+4} "
                  f"{up:>5} {dn:>5} {abs(up - dn):>7} {err:>+6} {gap:>4}  "
                  f"{' / '.join(lines)[:38]}")
        print(f"  timpang: median={np.median(bals):.0f} maks={max(bals)}  "
              f"|pusat|: median={np.median(errs):.0f} maks={max(errs)}  "
              f"tepi: min={min(gaps)} median={np.median(gaps):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
