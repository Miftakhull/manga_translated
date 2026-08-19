#!/usr/bin/env python3
"""Kenapa 4 region sisa tidak bisa digeser lagi? Rem-nya diberi nama.

Setelah layout() memilih kandidat paling seimbang lalu menghaluskan 1 px demi
1 px, empat region masih timpang (r2 +12, r7 -7, r9 -18, r11 +9). Ada tiga
kemungkinan rem dan keduanya perlu dibedakan sebelum diperbaiki:

  bentuk   : geseran membuat salah satu baris tidak muat lagi (build berubah /
             _verify gagal) — memang mentok, bukan bug.
  ukur     : geseran diterima tapi ketimpangan yang diukur tidak turun.
  n_baris  : geseran memaksa jumlah baris berubah.

Yang dicetak: langkah pertama yang ditolak dan alasannya, per region.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / ".stage"
TEXTS = ROOT / os.environ.get("TEXTS", "probe_llm2_opus5_clean.json")
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

import numpy as np             # noqa: E402
import typeset                 # noqa: E402
from config import SETTINGS     # noqa: E402

WATCH = {2, 7, 9, 11}


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))

    for r in regions:
        if r.idx not in WATCH:
            continue
        t = str(texts.get(str(r.idx), "")).upper()
        m = typeset._region_box_mask(r)[1]
        size, lines, sy, _ = typeset.fit(t, m, typeset.region_font_cap(m), fp)
        font = typeset._font(fp, size)
        lh = typeset._line_height(font)
        it, ib = typeset._ink_band(fp, size)
        cx, _ = typeset._centroid(m)
        mh, mw = m.shape[:2]
        pad = int(min(mh, mw) * SETTINGS.pad_ratio)

        def slack(ls, top):
            return typeset.block_slack(
                m, cx, pad, typeset._measure(ls[0], font),
                typeset._measure(ls[-1], font),
                top + it, top + (len(ls) - 1) * lh + ib)

        up, dn = slack(lines, sy)
        step = 1 if dn > up else -1
        print(f"\nr{r.idx} size={size} nb={len(lines)} sy={sy} atas={up} bawah={dn} "
              f"lh={lh} pad={pad} mh={mh}")
        top = sy
        cur = abs(up - dn)
        for k in range(abs(dn - up) // 2 + 1):
            t2 = top + step
            # build() tidak diekspor; rem "bentuk" dideteksi dengan menguji
            # ulang tiap baris pada lebarnya sendiri di y kandidat — probe yang
            # sama dipakai _verify().
            widths = [typeset._measure(ln, font) for ln in lines]
            bad = []
            for i, w in enumerate(widths):
                y = t2 + i * lh
                if not typeset._row_free(m, y + it, y + ib, cx - w / 2, cx + w / 2):
                    bad.append((i, lines[i], round(w)))
            over = (t2 + it < pad, t2 + (len(lines) - 1) * lh + ib > mh - pad)
            u2, d2 = slack(lines, t2)
            new = abs(u2 - d2)
            print(f"  +{k+1:>2} y={t2:>4} baris_gagal={bad} luar_pad={over} "
                  f"atas={u2} bawah={d2} timpang {cur}->{new}")
            if bad or any(over) or new >= cur:
                print("  REM: " + ("bentuk" if bad else
                                   "pad" if any(over) else "ukur"))
                break
            top, cur = t2, new
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
