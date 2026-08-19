#!/usr/bin/env python3
"""Sweep pad_ratio: ukuran layak per region, target halaman, sebaran.

Referensi (probe_margin.py) memakai margin sisi ~2% dari sisi terpendek balon
dan mengisi ~96% lebar interior. Kita memakai pad_ratio 0.10 yang DIKURANGI DUA
KALI di layout.max_width_at() -> 20% lebar hilang. Sweep ini mengukur akibatnya
pada ukuran font, bukan menebak.
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
CACHE = ROOT / ".probe_pre.pkl"
TEXTS = ROOT / "probe_font_texts.json"
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

import imgio     # noqa: E402
import textmask  # noqa: E402
import typeset   # noqa: E402
from config import SETTINGS  # noqa: E402


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with CACHE.open("rb") as f:
        regions = pickle.load(f)
    textmask.disjoin_overlapping_interiors(img, regions)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    for r in regions:
        r.translation = texts.get(str(r.idx))

    masks = {r.idx: typeset._region_box_mask(r)[1] for r in regions}
    ref_cap = {0: 15, 1: 17, 2: 16, 3: 22, 4: 16.5, 5: 21, 6: 11, 7: 11,
               8: 12, 9: 12, 10: 11, 11: 12, 12: 13}

    for pr in (0.10, 0.06, 0.04, 0.03, 0.02):
        SETTINGS.pad_ratio = pr
        plain, hyph = {}, {}
        for r in regions:
            text = str(r.translation).upper()
            m = masks[r.idx]
            mh, mw = m.shape[:2]
            pad = int(min(mh, mw) * pr)
            hi = int(np.clip(mh - 2 * pad, SETTINGS.min_font_size,
                             SETTINGS.max_font_size))
            p = typeset._search(text, m, SETTINGS.min_font_size, hi, fp, False)
            hy = typeset._search(text, m, SETTINGS.min_font_size, hi, fp, True)
            plain[r.idx] = p[0] if p else 0
            hyph[r.idx] = hy[0] if hy else 0
        both = [max(plain[i], hyph[i]) for i in plain]
        # DISUPERSEDE oleh probe_cal.py / probe_final.py / probe_tidy.py. Dulu
        # baris ini mengambil persentil halaman (typeset._PAGE_SIZE_PCTL) sebagai
        # 'target'; model itu sudah dibuang — referensi terukur menskalakan teks
        # ke besar balon, bukan menyeragamkannya. Yang tersisa berguna dari sweep
        # ini cuma kolom utuh/hyph per pad_ratio, jadi 'target' diganti median
        # sebagai ringkasan deskriptif saja, bukan keputusan.
        nz = [v for v in both if v]
        tgt = int(np.median(nz)) if nz else 0
        print(f"\npad_ratio={pr:<5} median={tgt:<3} "
              f"nol_utuh={sum(1 for v in plain.values() if not v)} "
              f"nol_apapun={sum(1 for v in both if not v)}")
        print("      idx  utuh  hyph  ref_cap  ~size_ref")
        for r in regions:
            i = r.idx
            # Anime Ace: cap height ~= 0.72 * ukuran font (diukur _ink_band).
            print(f"      {i:>3} {plain[i]:>5} {hyph[i]:>5} "
                  f"{ref_cap.get(i, 0):>8} {ref_cap.get(i, 0)/0.72:>10.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
