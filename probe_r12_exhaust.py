#!/usr/bin/env python3
"""Apakah r12 punya layout seimbang SAMA SEKALI pada margin ketat pad*2?

probe_margin2 varian E menyembuhkan r12 tapi ikut membesarkan lima region lain
(r5 14->15, r7 13->14, r8 14->15, r9 12->14, r10 13->14) dan median jarak tinta
ke tepi interior jatuh 3 px -> 1 px. Itu menukar satu balon timpang dengan
risiko di enam balon. Sebelum memilih, pertanyaannya harus dijawab dulu: pada
margin ketat, adakah ukuran + jumlah baris yang membuat r12 seimbang?

Dipindai penuh: tiap ukuran font dari plafon sampai min, tiap jumlah baris 1..8,
tiap top yang legal (langkah 1 px, bukan setengah baris), pakai margin pad*2.
Yang dicetak per ukuran: ketimpangan terbaik yang bisa dicapai dan barisnya.
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
IDX = int(os.environ.get("IDX", "12"))
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


def splits(words: list[str], n: int):
    """Semua pemecahan words ke n baris berurutan (tanpa mengubah urutan)."""
    if n == 1:
        yield [" ".join(words)]
        return
    for cut in range(1, len(words) - n + 2):
        for rest in splits(words[cut:], n - 1):
            yield [" ".join(words[:cut])] + rest


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    r = next(x for x in regions if x.idx == IDX)
    mask = typeset._region_box_mask(r)[1]
    text = str(texts[str(IDX)]).upper()
    words = text.split()
    mh, mw = mask.shape[:2]
    pad = int(min(mh, mw) * SETTINGS.pad_ratio)
    cap = typeset.region_font_cap(mask)
    print(f"r{IDX}  interior {mw}x{mh}  pad={pad}  plafon={cap}  {text!r}")
    print(f"\n{'size':>4} {'margin':>6} {'nb':>3} {'y':>4} {'atas':>5} {'bawah':>5} "
          f"{'timpang':>7}  baris")

    for mar_name, mar in (("pad*2", pad * 2.0), ("pad*1", pad * 1.0),
                          ("0", 0.0)):
        for size in range(cap, SETTINGS.min_font_size - 1, -1):
            font = typeset._font(fp, size)
            lh = typeset._line_height(font)
            it, ib = typeset._ink_band(fp, size)
            cx, _cy = typeset._centroid(mask)
            best = None
            for n in range(1, min(9, len(words) + 1)):
                ink_h = (n - 1) * lh + (ib - it)
                lo, hi = pad - it, mh - pad - ink_h - it
                if hi < lo:
                    continue
                for ls in splits(words, n):
                    # Lebar tiap baris wajib muat pada margin yang diuji, dan
                    # blok wajib lolos _verify (batas pad atas-bawah).
                    for top in range(lo, hi + 1):
                        ok = True
                        for k, l in enumerate(ls):
                            w = typeset._measure(l, font)
                            y = top + k * lh
                            if not typeset._row_free(mask, y + it, y + ib,
                                                     cx - (w + mar) / 2,
                                                     cx + (w + mar) / 2):
                                ok = False
                                break
                        if not ok:
                            continue
                        up, dn = typeset.block_slack(
                            mask, cx, pad, typeset._measure(ls[0], font),
                            typeset._measure(ls[-1], font),
                            top + it, top + (n - 1) * lh + ib)
                        bal = abs(up - dn)
                        if best is None or bal < best[0]:
                            best = (bal, size, n, top, up, dn, ls)
                        if bal == 0:
                            break
                    if best and best[0] == 0:
                        break
                if best and best[0] == 0:
                    break
            if best is None:
                print(f"{size:>4} {mar_name:>6}   -    -     -     -       -  TIDAK MUAT")
                continue
            bal, sz, n, top, up, dn, ls = best
            print(f"{sz:>4} {mar_name:>6} {n:>3} {top:>4} {up:>5} {dn:>5} "
                  f"{bal:>7}  {ls}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
