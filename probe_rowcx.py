#!/usr/bin/env python3
"""Lebar bebas per baris: dipusatkan di cx tetap vs mengikuti rongga sungguhan.

Dugaan untuk r6 ('SORRY.' melorot ke bawah): masknya bukan oval melainkan pita
tegak melengkung — potongan satu kolom teks vertikal Jepang dari balon yang
dibagi dengan r7. Pusat horizontal pita itu BERGESER seiring y, sementara
max_width_at() selalu memeriksa rongga simetris di sekitar cx yang sama untuk
semua baris. Di ketinggian tengah, pita bergeser menjauh dari cx sehingga
probe simetris gagal, dan satu-satunya y yang lolos ada di bawah tempat pita
melebar — persis 50 px di bawah centroid.

Yang dicetak per baris y: lebar bebas simetris di cx (yang dipakai sekarang),
lebar rongga terpanjang di baris itu, dan pusat rongga tersebut.
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


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    r = next(x for x in regions if x.idx == IDX)
    box, m = typeset._region_box_mask(r)
    mh, mw = m.shape[:2]
    cx, cy = typeset._centroid(m)
    pad = int(min(mh, mw) * SETTINGS.pad_ratio)
    size = r.final_font_size or 11
    font = typeset._font(fp, size)
    it, ib = typeset._ink_band(fp, size)
    text = "SORRY." if IDX == 6 else ""
    need = typeset._measure(text, font) if text else 0.0
    print(f"r{IDX}  mask {mw}x{mh}  cx={cx} cy={cy} pad={pad} size={size}")
    print(f"     '{text}' butuh {need:.1f} px; band tinta {it}..{ib}")

    def sym_width(y: int) -> float:
        lo, hi = 0.0, float(mw)
        for _ in range(9):
            mid = (lo + hi) / 2
            if typeset._row_free(m, y + it, y + ib, cx - mid / 2, cx + mid / 2):
                lo = mid
            else:
                hi = mid
        return max(lo - pad * 2, 0.0)

    def best_run(y: int):
        """Rongga terpanjang di pita baris ini, tanpa dipaksa simetris di cx."""
        flags = np.ones(mw, bool)
        for x in range(mw):
            flags[x] = typeset._row_free(m, y + it, y + ib, x, x + 1)
        b_len, b_c = 0, cx
        x = 0
        while x < mw:
            if not flags[x]:
                x += 1
                continue
            s = x
            while x < mw and flags[x]:
                x += 1
            if x - s > b_len:
                b_len, b_c = x - s, (s + x) // 2
        return b_len, b_c

    print(f"\n{'y':>4} {'sym@cx':>7} {'rongga':>7} {'pusat':>6} {'geser':>6} "
          f"{'muat_sym':>9} {'muat_rongga':>12}")
    for y in range(pad - it, mh - pad - (ib - it) - it + 1, 6):
        sw = sym_width(y)
        bl, bc = best_run(y)
        print(f"{y:>4} {sw:>7.1f} {bl:>7} {bc:>6} {bc - cx:>+6} "
              f"{'ya' if sw >= need else 'TIDAK':>9} "
              f"{'ya' if bl - pad * 2 >= need else 'TIDAK':>12}")
    print(f"\ncy={cy}  y_top terpusat untuk 1 baris = "
          f"{cy - (ib - it) // 2 - it}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
