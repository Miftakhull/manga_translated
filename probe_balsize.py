#!/usr/bin/env python3
"""Ketimpangan hasil layout() SUNGGUHAN di tiap ukuran font, per region.

probe_r12_exhaust menunjukkan pada margin ketat pad*2 r12 seimbang di ukuran 14
(timpang 1) sementara ukuran 15 mustahil lebih baik dari 43. Jadi obatnya bukan
melonggarkan margin, melainkan berhenti memaksa ukuran terbesar ketika ukuran
terbesar itu tidak bisa ditata dengan seimbang.

Probe ini memanggil typeset.layout() apa adanya — greedy build, sapuan setengah
baris, lompatan n, _polish, semuanya — supaya angkanya betul-betul yang akan
terjadi, bukan hasil pencarian ideal. Untuk tiap region dicetak tiap ukuran dari
plafon ke min: muat atau tidak, dan kalau muat berapa ketimpangannya.
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


def bal_of(mask, text, size, fp):
    """(muat, timpang, nb, y, baris) dari layout() apa adanya."""
    ok, lines, y = typeset.layout(text, mask, size, fp, hyphenate=False)
    if not ok or not lines:
        return None
    font = typeset._font(fp, size)
    lh = typeset._line_height(font)
    it, ib = typeset._ink_band(fp, size)
    cx, _ = typeset._centroid(mask)
    mh, mw = mask.shape[:2]
    pad = int(min(mh, mw) * SETTINGS.pad_ratio)
    up, dn = typeset.block_slack(
        mask, cx, pad, typeset._measure(lines[0], font),
        typeset._measure(lines[-1], font),
        y + it, y + (len(lines) - 1) * lh + ib)
    return abs(up - dn), len(lines), y, lines


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))

    # Untuk tiap region: ukuran terbesar yang muat (perilaku sekarang) vs ukuran
    # terbesar yang muat DAN timpangnya <= ambang.
    print(f"{'idx':>3} {'plafon':>6} | {'sekarang':>18} | {'seimbang':>18} | turun")
    drop = []
    for r in regions:
        t = str(texts.get(str(r.idx), "")).upper()
        if not t:
            continue
        m = typeset._region_box_mask(r)[1]
        cap = typeset.region_font_cap(m)
        rows = []
        for size in range(cap, SETTINGS.min_font_size - 1, -1):
            got = bal_of(m, t, size, fp)
            if got:
                rows.append((size, *got))
        if not rows:
            print(f"{r.idx:>3} {cap:>6} | tidak muat sama sekali")
            continue
        now = rows[0]
        # tol = setengah tinggi baris, ambang "sudah terpusat" yang sama dipakai
        # layout() untuk berhenti menyapu.
        lh = typeset._line_height(typeset._font(fp, now[0]))
        tol = max(2, lh // 2)
        good = next((x for x in rows if x[1] <= tol), None)
        gs = f"size={good[0]} nb={good[2]} timpang={good[1]}" if good else "-"
        print(f"{r.idx:>3} {cap:>6} | size={now[0]} nb={now[2]} timpang={now[1]:<3} | "
              f"{gs:>18} | {now[0] - good[0] if good else '-'}")
        if good:
            drop.append(now[0] - good[0])
        print(f"      semua: " + "  ".join(
            f"{s}:{b}" for s, b, _n, _y, _l in rows))
    print(f"\nturun ukuran: {drop}  maks={max(drop)} jumlah_yang_turun="
          f"{sum(d > 0 for d in drop)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
