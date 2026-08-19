#!/usr/bin/env python3
"""Dua perbaikan yang TIDAK menyentuh margin, diukur bersama.

Kesimpulan dua probe sebelumnya:
  - probe_margin: margin 0 menyembuhkan r12 tapi median jarak tinta ke garis
    balon jatuh 3 -> 0 px di seluruh halaman. Ditolak.
  - probe_r12_exhaust: pada margin ketat pad*2, r12 SUDAH punya layout seimbang
    di ukuran 14 (timpang 1, 4 baris) dan 13 (timpang 0, 3 baris); yang mustahil
    seimbang hanya ukuran 15 (paling bagus 43).
  - probe_balsize: layout() sungguhan cuma mencapai 32 di ukuran 14 dan 24 di 13,
    jadi bukan geometrinya yang mentok — pencariannya yang tidak sampai.

Maka dua perbaikan yang diuji di sini:

  N  pindai jumlah baris. `nxt = len(lines) + 1` melompati n: di r12 n0=3, build
     di y terpusat memberi 4 baris dengan ok=False, jadi n langsung 5 dan n=4
     tidak pernah dicoba. N memindai n0..n0+4 dan mengambil yang paling seimbang.

  S  ukuran sadar keseimbangan. fit() sekarang memakai jalur cepat "ukuran plafon
     muat -> pakai" tanpa melihat bentuk bloknya. S menurunkan ukuran selama
     ketimpangannya masih di atas ambang DAN penurunannya tidak lebih dari
     _BAL_MAX_DROP px, supaya balon lain (yang sudah seimbang di plafon) tidak
     ikut mengecil.
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

import typeset   # noqa: E402
from config import SETTINGS  # noqa: E402

from probe_margin import edge_gap      # noqa: E402
from probe_margin2 import sim          # noqa: E402

MAX_DROP = int(os.environ.get("MAX_DROP", "3"))


def pick(mask, text, fp, scan_n: bool, bal_aware: bool):
    """Ukuran + layout terpilih. Margin selalu ketat (pad*2)."""
    cap = typeset.region_font_cap(mask)
    first = None
    for size in range(cap, SETTINGS.min_font_size - 1, -1):
        got = sim(mask, text, size, fp, scan_n, False)
        if got is None:
            continue
        lines, top, up, dn = got[0], got[1], got[2], got[3]
        bal = abs(up - dn)
        lh = typeset._line_height(typeset._font(fp, size))
        tol = max(2, lh // 2)
        if first is None:
            first = (size, got, bal)
            if not bal_aware or bal <= tol:
                return size, got, bal
        # Turun ukuran hanya kalau blok teratas memang timpang, dan hanya
        # sejauh MAX_DROP px supaya teks tidak diam-diam mengecil.
        if first[0] - size > MAX_DROP:
            break
        if bal <= tol:
            return size, got, bal
    return first if first else None


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))

    for name, scan_n, bal in (("sekarang", False, False),
                              ("N  pindai n", True, False),
                              ("N+S  pindai n + ukuran sadar seimbang", True, True)):
        print(f"\n=== {name}")
        print(f"  {'idx':>3} {'size':>4} {'nb':>3} {'timpang':>7} {'jarak_tepi':>10}  baris")
        bals, gaps, sizes = [], [], []
        for r in regions:
            t = str(texts.get(str(r.idx), "")).upper()
            if not t:
                continue
            m = typeset._region_box_mask(r)[1]
            res = pick(m, t, fp, scan_n, bal)
            if res is None:
                print(f"  {r.idx:>3}  TIDAK MUAT")
                continue
            size, got, b = res
            lines, top, up, dn, lh, it, ib, cx, font = got
            gap = edge_gap(m, lines, top, lh, it, ib, cx, font)
            bals.append(b)
            gaps.append(gap)
            sizes.append(size)
            print(f"  {r.idx:>3} {size:>4} {len(lines):>3} {b:>7} {gap:>10}  {lines}")
        print(f"  timpang median={np.median(bals):.0f} maks={max(bals)} | "
              f"jarak_tepi min={min(gaps)} median={np.median(gaps):.0f} "
              f"nol={sum(g == 0 for g in gaps)}/{len(gaps)} | size={sorted(sizes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
