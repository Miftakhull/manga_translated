#!/usr/bin/env python3
"""Sumbu & aturan lebar: empat varian, diukur berdampingan di semua region.

Temuan probe_fitwin.py: yang menahan 'SORRY.' di r6 BUKAN ruang dan bukan
penyeimbang, tapi cara mengukur lebar. max_width_at() melebarkan pita SIMETRIS
di sekitar satu cx = centroid mask. Interior r6 sudah dipotong tetangganya
sehingga rongganya tidak simetris terhadap cx: di y terpusat rongganya
x=14..67 (54 px) tapi ekspansi simetris di cx=36 cuma mengakui 2*(36-14)=44 px,
dikurangi pad*2 jadi 40 px — 'SORRY.' butuh 49. Jadi y terpusat DITOLAK, dan
satu-satunya y yang lolos ada di y=119..140, 39 px di bawah tengah.

  aturan lebar        y yang muat      y terdekat ke terpusat
  simetris di cx      22 (119..140)    +39 px   <- sekarang
  simetris di rongga  71 ( 50..145)     -7 px
  interval apa adanya 78 ( 46..143)     +0 px

Varian yang diuji di sini:
  V0  sekarang        — lebar simetris di centroid, pilih min|atas-bawah|
  V1  lebar interval  — lebar = rongga terlebar di band baris itu, tiap baris
                        dipusatkan di rongganya sendiri
  V2  V1 + sumbu blok — satu x untuk seluruh blok (pusat irisan rongga semua
                        barisnya) supaya blok tidak bergerigi
  V3  V2 + pilih yang paling tengah secara vertikal di antara yang seimbang

Metrik: ukuran, jumlah baris, timpang (atas vs bawah), pusat (tengah tinta -
cy), tepi (jarak tinta terdekat ke garis balon — TIDAK BOLEH turun), gerigi
(sebaran x antar baris), dan tanda hubung.

    TEXTS=probe_opus5_clean.json python probe_axis.py
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

import cv2       # noqa: E402
import typeset   # noqa: E402
from config import SETTINGS  # noqa: E402

# (nama, aturan_lebar, sumbu_blok, pilih_paling_tengah)
VARIANTS = (
    ("V0 sekarang: simetris di centroid", "sym", False, False),
    ("V1 lebar interval, x per baris", "run", False, False),
    ("V2 lebar interval, satu x per blok", "run", True, False),
    ("V3 V2 + pilih paling tengah", "run", True, True),
)


def band_runs(mask: np.ndarray, y0: int, y1: int) -> list[tuple[int, int]]:
    """Interval kolom yang bebas di SELURUH band y0..y1. Ambang = _row_free."""
    mh = mask.shape[0]
    a, b = max(int(y0), 0), min(int(y1), mh)
    if b <= a:
        return []
    ok = (mask[a:b] >= 200).all(0)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    st = np.concatenate(([0], cuts + 1))
    en = np.concatenate((cuts, [idx.size - 1]))
    return [(int(idx[s]), int(idx[e])) for s, e in zip(st, en)]


def widest(mask: np.ndarray, y0: int, y1: int) -> tuple[int, int] | None:
    rr = band_runs(mask, y0, y1)
    return max(rr, key=lambda t: t[1] - t[0]) if rr else None


def sim(mask: np.ndarray, text: str, size: int, fp: str,
        wmode: str, block_axis: bool, prefer_mid: bool):
    font = typeset._font(fp, size)
    words = text.split()
    if not words:
        return None
    lh = typeset._line_height(font)
    it, ib = typeset._ink_band(fp, size)
    cx0, cy = typeset._centroid(mask)
    mh, mw = mask.shape[:2]
    pad = int(min(mh, mw) * SETTINGS.pad_ratio)

    def sym_width(y: int, c: int) -> float:
        lo, hi = 0.0, float(mw)
        for _ in range(9):
            mid = (lo + hi) / 2
            if typeset._row_free(mask, y + it, y + ib, c - mid / 2, c + mid / 2):
                lo = mid
            else:
                hi = mid
        return max(lo - pad * 2, 0.0)

    def line_space(y: int) -> tuple[float, int]:
        """(lebar tersedia, x pusat) untuk satu baris di y."""
        if wmode == "sym":
            return sym_width(y, cx0), cx0
        run = widest(mask, y + it, y + ib)
        if run is None:
            return 0.0, cx0
        a, b = run
        return max((b - a + 1) - pad * 2, 0.0), (a + b) // 2

    def block_x(top: int, n: int, ws: list[float]) -> int:
        """Satu x untuk seluruh blok: pusat irisan rongga semua barisnya.

        Irisan, bukan rata-rata: x yang dipakai harus sah untuk SETIAP baris,
        kalau tidak baris terlebar akan menembus garis balon.
        """
        if wmode == "sym" or not block_axis:
            return cx0
        lo, hi = 0, mw - 1
        for k in range(n):
            run = widest(mask, top + k * lh + it, top + k * lh + ib)
            if run is None:
                return cx0
            lo, hi = max(lo, run[0]), min(hi, run[1])
        if hi - lo + 1 < max(ws, default=0.0):
            return cx0          # irisan lebih sempit dari baris terlebar
        return (lo + hi) // 2

    def build(top: int):
        lines: list[str] = []
        widths: list[float] = []
        q = list(words)
        i = 0
        for _ in range(64):
            if i >= len(q):
                return lines, widths, True
            av, _c = line_space(top + len(lines) * lh)
            if av < size * 0.9:
                return lines, widths, False
            ln = q[i]
            j = i + 1
            while j < len(q) and typeset._measure(f"{ln} {q[j]}", font) <= av:
                ln = f"{ln} {q[j]}"
                j += 1
            if j == i + 1 and typeset._measure(ln, font) > av:
                return lines, widths, False
            lines.append(ln)
            widths.append(av)
            i = j
        return lines, widths, i >= len(q)

    def axes(ls: list[str], top: int) -> list[int]:
        """x pusat tiap baris di bawah varian ini."""
        if wmode == "sym":
            return [cx0] * len(ls)
        ws = [typeset._measure(x, font) for x in ls]
        if block_axis:
            return [block_x(top, len(ls), ws)] * len(ls)
        out = []
        for k in range(len(ls)):
            out.append(line_space(top + k * lh)[1])
        return out

    def verify(ls: list[str], top: int, xs: list[int]) -> bool:
        for k, ln in enumerate(ls):
            w = typeset._measure(ln, font)
            y = top + k * lh
            if not typeset._row_free(mask, y + it, y + ib,
                                     xs[k] - w / 2, xs[k] + w / 2):
                return False
        return not (top + it < pad or top + (len(ls) - 1) * lh + ib > mh - pad)

    def slack(ls: list[str], top: int, xs: list[int]):
        # Diukur di sumbu baris pertama/terakhir masing-masing — itulah kolom
        # yang benar-benar dilewati tinta di ujung atas dan bawah blok.
        up = typeset.block_slack(
            mask, xs[0], pad, typeset._measure(ls[0], font),
            typeset._measure(ls[0], font),
            top + it, top + it)[0]
        dn = typeset.block_slack(
            mask, xs[-1], pad, typeset._measure(ls[-1], font),
            typeset._measure(ls[-1], font),
            top + (len(ls) - 1) * lh + ib, top + (len(ls) - 1) * lh + ib)[1]
        return up, dn

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

    n0 = max(1, int(np.ceil(typeset._measure(text, font) / max(mw - pad * 2, 1))))
    tol = max(2, lh // 2)
    hit = None
    for n in range(n0, n0 + 5):
        for top in tops(n):
            cand, _ws, ok = build(top)
            if not (ok and len(cand) == n):
                continue
            xs = axes(cand, top)
            if not verify(cand, top, xs):
                continue
            up, dn = slack(cand, top, xs)
            bal = abs(up - dn)
            mid = top + it + ((len(cand) - 1) * lh + (ib - it)) // 2
            key = (abs(mid - cy), bal) if prefer_mid else (bal, abs(mid - cy))
            if hit is None or key < hit[0]:
                hit = (key, top, cand, xs, bal)
            if (key[0] <= tol if prefer_mid else bal <= tol):
                break
        if hit is not None and (hit[0][0] <= tol):
            break
    if hit is None:
        return None
    _key, top, cand, xs, bal = hit
    # Penghalusan 1 px: geser ke arah yang menyeimbangkan, sama seperti _polish.
    up, dn = slack(cand, top, xs)
    bt, bx, bb = top, xs, abs(up - dn)
    stp = 1 if dn > up else -1
    for _ in range(abs(dn - up) // 2):
        t = bt + stp
        x2 = axes(cand, t)
        if not verify(cand, t, x2):
            break
        u2, d2 = slack(cand, t, x2)
        if abs(u2 - d2) >= bb:
            break
        bt, bx, bb = t, x2, abs(u2 - d2)
    u, d = slack(cand, bt, bx)
    return cand, bt, bx, u, d, lh, it, ib, font, cy


def edge_gap_x(mask, lines, top, lh, it, ib, xs, font) -> int:
    """probe_margin.edge_gap tapi dengan x per baris."""
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    mh, mw = mask.shape[:2]
    best = 10**6
    for k, ln in enumerate(lines):
        w = typeset._measure(ln, font)
        y0, y1 = top + k * lh + it, top + k * lh + ib
        x0, x1 = int(xs[k] - w / 2), int(np.ceil(xs[k] + w / 2))
        y0, y1 = max(y0, 0), min(y1, mh)
        x0, x1 = max(x0, 0), min(x1, mw)
        if y1 <= y0 or x1 <= x0:
            return 0
        box = dist[y0:y1, x0:x1]
        peri = np.concatenate([box[0], box[-1], box[:, 0], box[:, -1]])
        best = min(best, int(peri.min()))
    return best


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))

    for name, wmode, baxis, pmid in VARIANTS:
        print(f"\n=== {name}")
        print(f"  {'idx':>3} {'size':>4} {'nb':>3} {'timpang':>7} {'pusat':>6} "
              f"{'tepi':>4} {'gerigi':>6}  baris")
        sizes, bals, errs, gaps, jit = [], [], [], [], []
        for r in regions:
            t = str(texts.get(str(r.idx), "")).upper()
            if not t:
                continue
            m = typeset._region_box_mask(r)[1]
            cap = typeset.region_font_cap(m)
            got = None
            for size in range(cap, SETTINGS.min_font_size - 1, -1):
                got = sim(m, t, size, fp, wmode, baxis, pmid)
                if got:
                    break
            if not got:
                print(f"  {r.idx:>3}  TIDAK MUAT di ukuran mana pun")
                continue
            lines, top, xs, up, dn, lh, it, ib, font, cy = got
            gap = edge_gap_x(m, lines, top, lh, it, ib, xs, font)
            mid = top + it + ((len(lines) - 1) * lh + (ib - it)) // 2
            err = mid - cy
            g = max(xs) - min(xs)
            sizes.append(size)
            bals.append(abs(up - dn))
            errs.append(abs(err))
            gaps.append(gap)
            jit.append(g)
            print(f"  {r.idx:>3} {size:>4} {len(lines):>3} {abs(up - dn):>7} "
                  f"{err:>+6} {gap:>4} {g:>6}  {' / '.join(lines)[:36]}")
        print(f"  timpang med={np.median(bals):.0f} maks={max(bals)} | "
              f"|pusat| med={np.median(errs):.0f} maks={max(errs)} | "
              f"tepi min={min(gaps)} med={np.median(gaps):.0f} | "
              f"gerigi maks={max(jit)} | ukuran med={np.median(sizes):.0f} "
              f"min={min(sizes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
