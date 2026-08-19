#!/usr/bin/env python3
"""Dua varian sempit untuk r12, tanpa mengorbankan jarak tinta ke garis balon.

probe_margin.py menunjukkan margin nol memang menyembuhkan r12 (timpang 41 -> 1)
tapi median jarak_tepi seluruh halaman jatuh 3 px -> 0 px. Itu menukar satu cacat
dengan cacat yang baru saja diberantas. Dua varian di sini melebarkan PENCARIAN,
bukan melonggarkan batas:

  D  pilih n terbaik lintas percobaan. Sekarang loop berhenti di n pertama yang
     menghasilkan hit, walau n berikutnya jauh lebih seimbang (r7: n=7 timpang 15
     dipilih padahal n=8 timpang 1 — kebetulan tertolong lompatan n yang ada).
     D memindai n0..n0+4, ambil paling seimbang, berhenti dini kalau <= tol.

  E  D + margin longgar HANYA sebagai penyelamat: tiap n dicoba dulu dengan
     margin pad*2 penuh; kalau tak ada satu pun top yang lolos, n itu diulang
     dengan margin 0 — tetap wajib lolos _verify(). Region yang sudah berhasil
     secara ketat tidak berubah sama sekali; yang tadinya tidak punya layout
     legal dapat satu.

    TEXTS=probe_llm2_seekai-claude-opus-5.json python probe_margin2.py
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

from probe_margin import edge_gap  # noqa: E402

VARIANTS = (("sekarang", False, False), ("D  pindai n", True, False),
            ("E  pindai n + relax penyelamat", True, True))


def sim(mask, text, size, fp, scan_n: bool, relax: bool):
    font = typeset._font(fp, size)
    words = text.split()
    if not words:
        return None
    lh = typeset._line_height(font)
    it, ib = typeset._ink_band(fp, size)
    cx, cy = typeset._centroid(mask)
    mh, mw = mask.shape[:2]
    pad = int(min(mh, mw) * SETTINGS.pad_ratio)

    def width_at(y: int, mar: float) -> float:
        lo, hi = 0.0, float(mw)
        for _ in range(9):
            mid = (lo + hi) / 2
            if typeset._row_free(mask, y + it, y + ib, cx - mid / 2, cx + mid / 2):
                lo = mid
            else:
                hi = mid
        return max(lo - mar, 0.0)

    def build(top: int, mar: float):
        lines: list[str] = []
        i = 0
        for _ in range(64):
            if i >= len(words):
                return lines, True
            av = width_at(top + len(lines) * lh, mar)
            if av < size * 0.9:
                return lines, False
            ln, j = words[i], i + 1
            while j < len(words) and typeset._measure(f"{ln} {words[j]}", font) <= av:
                ln, j = f"{ln} {words[j]}", j + 1
            if j == i + 1 and typeset._measure(ln, font) > av:
                return lines, False
            lines.append(ln)
            i = j
        return lines, i >= len(words)

    def center_y(n: int) -> int:
        return cy - ((n - 1) * lh + (ib - it)) // 2 - it

    def tops(n: int) -> list[int]:
        ink_h = (n - 1) * lh + (ib - it)
        lo, hi = pad - it, mh - pad - ink_h - it
        if hi < lo:
            return [center_y(n)]
        c = int(np.clip(center_y(n), lo, hi))
        st = max(lh // 2, 2)
        return [c] + [t for t in range(lo, hi + 1, st) if t != c]

    def verify(ls, top) -> bool:
        for k, l in enumerate(ls):
            w = typeset._measure(l, font)
            y = top + k * lh
            if not typeset._row_free(mask, y + it, y + ib, cx - w / 2, cx + w / 2):
                return False
        return not (top + it < pad or top + (len(ls) - 1) * lh + ib > mh - pad)

    def slack(ls, top):
        return typeset.block_slack(
            mask, cx, pad, typeset._measure(ls[0], font),
            typeset._measure(ls[-1], font),
            top + it, top + (len(ls) - 1) * lh + ib)

    def sweep(n: int, mar: float):
        """Top paling seimbang untuk jumlah baris n. None = tak ada yang legal."""
        best = None
        for top in tops(n):
            cand, ok = build(top, mar)
            if not (ok and len(cand) == n and verify(cand, top)):
                continue
            up, dn = slack(cand, top)
            bal = abs(up - dn)
            if best is None or bal < best[0]:
                best = (bal, top, cand)
            if bal <= tol:
                break
        return best

    n0 = max(1, int(np.ceil(typeset._measure(text, font) / max(mw - pad * 2, 1))))
    tol = max(2, lh // 2)

    if not scan_n:  # perilaku sekarang, apa adanya
        n = n0
        for _ in range(5):
            hit = sweep(n, pad * 2)
            if hit:
                break
            cand, ok = build(center_y(n), pad * 2)
            nxt = max(1, len(cand) + (0 if ok else 1))
            n = n + 1 if nxt == n else nxt
        else:
            hit = None
    else:
        hit = None
        for n in range(n0, n0 + 5):
            got = sweep(n, pad * 2)
            if got is None and relax:
                got = sweep(n, 0.0)
            if got and (hit is None or got[0] < hit[0]):
                hit = got
            if hit and hit[0] <= tol:
                break

    if hit is None:
        return None
    bal, top, cand = hit
    up, dn = slack(cand, top)
    bt, bb, stp = top, bal, 1 if dn > up else -1
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


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))

    for name, scan_n, relax in VARIANTS:
        print(f"\n=== {name}")
        print(f"  {'idx':>3} {'size':>4} {'nb':>3} {'timpang':>7} {'jarak_tepi':>10}  baris")
        bals, gaps, sizes = [], [], []
        for r in regions:
            t = str(texts.get(str(r.idx), "")).upper()
            if not t:
                continue
            m = typeset._region_box_mask(r)[1]
            got = None
            for size in range(typeset.region_font_cap(m), SETTINGS.min_font_size - 1, -1):
                got = sim(m, t, size, fp, scan_n, relax)
                if got:
                    break
            if not got:
                print(f"  {r.idx:>3}  TIDAK MUAT")
                continue
            lines, top, up, dn, lh, it, ib, cx, font = got
            gap = edge_gap(m, lines, top, lh, it, ib, cx, font)
            bals.append(abs(up - dn))
            gaps.append(gap)
            sizes.append(size)
            print(f"  {r.idx:>3} {size:>4} {len(lines):>3} {abs(up - dn):>7} "
                  f"{gap:>10}  {lines}")
        print(f"  timpang median={np.median(bals):.0f} maks={max(bals)} | "
              f"jarak_tepi min={min(gaps)} median={np.median(gaps):.0f} "
              f"nol={sum(g == 0 for g in gaps)}/{len(gaps)} | "
              f"size={sorted(sizes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
