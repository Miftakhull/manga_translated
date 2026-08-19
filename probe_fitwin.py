#!/usr/bin/env python3
"""Di y mana 'SORRY.' MUAT, di bawah tiga aturan lebar yang berbeda?

Tiga aturan, dari yang paling ketat ke paling longgar:
  S  simetris di centroid  — yang dipakai layout() sekarang
  B  simetris di pusat rongga blok — varian X di probe_blockcx.py
  I  interval bebas apa adanya (baris ditaruh di tengah rongga, bukan di
     tengah kotak) — batas fisik sesungguhnya

Kalau aturan I memuatnya di y terpusat sementara S dan B tidak, maka yang
menahan 'SORRY.' bukan ruang, melainkan cara mengukurnya: ekspansi simetris
di sekitar SATU x membuang setengah rongga yang tidak simetris terhadap x itu.

    IDX=6 TEXT=SORRY. python probe_fitwin.py
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / ".stage"
IDX = int(os.environ.get("IDX", "6"))
TEXT = os.environ.get("TEXT", "SORRY.")
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


def runs(mask: np.ndarray, y0: int, y1: int) -> list[tuple[int, int]]:
    """Interval kolom yang bebas di SELURUH band y0..y1 (ambang sama _row_free)."""
    mh, mw = mask.shape[:2]
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


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    r = next(x for x in regions if x.idx == IDX)
    m = typeset._region_box_mask(r)[1]
    mh, mw = m.shape[:2]
    cx0, cy = typeset._centroid(m)
    pad = int(min(mh, mw) * SETTINGS.pad_ratio)
    size = typeset.region_font_cap(m)
    font = typeset._font(fp, size)
    it, ib = typeset._ink_band(fp, size)
    need = typeset._measure(TEXT, font)
    lh = typeset._line_height(font)
    y_mid = cy - (ib - it) // 2 - it
    print(f"r{IDX} {mw}x{mh} cx={cx0} cy={cy} pad={pad} size={size} lh={lh}")
    print(f"'{TEXT}' butuh {need:.1f} px; band tinta {it}..{ib}; "
          f"y_top terpusat={y_mid}")

    def sym(y: int, c: int) -> float:
        lo, hi = 0.0, float(mw)
        for _ in range(9):
            mid = (lo + hi) / 2
            if typeset._row_free(m, y + it, y + ib, c - mid / 2, c + mid / 2):
                lo = mid
            else:
                hi = mid
        return max(lo - pad * 2, 0.0)

    ok_s, ok_b, ok_i = [], [], []
    print(f"\n{'y':>4} {'S':>6} {'B':>6} {'I':>6} {'rongga':>12} {'cx_B':>5} "
          f"{'cx_I':>5}")
    for y in range(pad - it, mh - pad - (ib - it) - it + 1):
        rr = runs(m, y + it, y + ib)
        if not rr:
            continue
        a, b = max(rr, key=lambda t: t[1] - t[0])
        cb = (a + b) // 2
        wi = (b - a + 1) - pad * 2          # interval bebas apa adanya
        ws, wb = sym(y, cx0), sym(y, cb)
        if ws >= need:
            ok_s.append(y)
        if wb >= need:
            ok_b.append(y)
        if wi >= need:
            ok_i.append(y)
        if y % 6 == 0 or y == y_mid:
            mark = "  <- terpusat" if y == y_mid else ""
            print(f"{y:>4} {ws:>6.1f} {wb:>6.1f} {wi:>6.1f} "
                  f"{f'{a}..{b}':>12} {cb:>5} "
                  f"{int(a + pad + need / 2):>5}{mark}")

    def rng(v: list[int]) -> str:
        return f"{len(v):>3} y {v[0]}..{v[-1]}" if v else "  0 (nihil)"

    print(f"\nS simetris di centroid   : {rng(ok_s)}")
    print(f"B simetris di pusat rongga: {rng(ok_b)}")
    print(f"I interval bebas apa adanya: {rng(ok_i)}")
    for nm, v in (("S", ok_s), ("B", ok_b), ("I", ok_i)):
        if v:
            best = min(v, key=lambda y: abs(y - y_mid))
            print(f"   {nm}: y terdekat ke terpusat = {best} "
                  f"(galat {best - y_mid:+d} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
