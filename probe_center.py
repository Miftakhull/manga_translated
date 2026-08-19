#!/usr/bin/env python3
"""Seberapa terpusat blok teks di dalam interior balon — per region, vertikal.

Yang diukur BUKAN jarak ke centroid mask. Centroid balon oval yang sudah
dipotong tetangganya bisa jauh dari tengah RUANG PAKAI-nya, jadi "dekat
centroid" tidak sama dengan "terlihat di tengah". Yang dipakai mata:
sisa ruang di ATAS baris pertama versus di BAWAH baris terakhir, diukur pada
kolom yang benar-benar dilewati teks. Kalau keduanya seimbang, teksnya terlihat
di tengah; kalau salah satu jauh lebih besar, teksnya nempel ke satu sisi.

    TEXTS=probe_llm2_opus5_clean.json python probe_center.py
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
CACHE = ROOT / ".probe_cache.pkl"
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

import typeset                # noqa: E402
from config import SETTINGS   # noqa: E402


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with CACHE.open("rb") as f:
        regions = pickle.load(f)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))

    print(f"pad_ratio={SETTINGS.pad_ratio} line_spacing={SETTINGS.line_spacing}")
    print(f"{'idx':>3} {'size':>4} {'nb':>2} | {'atas':>5} {'bawah':>5} {'geser':>6} "
          f"{'pusat':>6} | teks")
    worst = []
    for r in regions:
        t = str(texts.get(str(r.idx), "")).upper()
        if not t:
            continue
        m = typeset._region_box_mask(r)[1]
        size, lines, sy, _ov = typeset.fit(t, m, typeset.region_font_cap(m), fp)
        font = typeset._font(fp, size)
        lh = typeset._line_height(font)
        ink_top, ink_bot = typeset._ink_band(fp, size)
        if not lines:
            continue
        # Sumbu x yang BENAR-BENAR dipakai render_region() menggambar blok ini.
        # Memakai centroid di sini mengukur sisa ruang di kolom yang tidak
        # dilewati tinta: pada r6 halaman ini centroid dan sumbu blok berjarak
        # 5 px, cukup untuk melaporkan sisa atas -39 px yang tidak pernah ada.
        cx = typeset.line_axis(m, lines, sy, size, fp)
        _cx0, cy = typeset._centroid(m)

        ink_a = sy + ink_top                              # atas tinta baris ke-1
        ink_b = sy + (len(lines) - 1) * lh + ink_bot      # bawah tinta terakhir
        # Sisa ruangnya diukur oleh fungsi yang SAMA dengan yang dipakai layout()
        # untuk menyeimbangkan. Kalau probe punya rumus sendiri, ia mengukur
        # sesuatu yang lain dan angkanya tidak bisa dipakai menilai perbaikan.
        up, dn = typeset.block_slack(
            m, cx, int(min(m.shape[:2]) * SETTINGS.pad_ratio),
            typeset._measure(lines[0], font), typeset._measure(lines[-1], font),
            ink_a, ink_b)
        # geser = ke mana blok harus digeser supaya sisa atas == sisa bawah.
        shift = (dn - up) // 2
        # pusat = tengah tinta blok vs centroid mask. Ini yang dilihat mata
        # sebagai "di tengah antara atas dan bawah", dan tidak selalu sama
        # dengan geser: pada interior yang tidak simetris, sisa ruang bisa
        # seimbang di kolom teks sementara bloknya sendiri jauh dari tengah.
        pusat = (ink_a + ink_b) // 2 - cy
        print(f"{r.idx:>3} {size:>4} {len(lines):>2} | {up:>5} {dn:>5} {shift:>+6} "
              f"{pusat:>+6} | {' / '.join(lines)[:40]}")
        if abs(shift) >= max(4, size // 2):
            worst.append((r.idx, shift))
    print(f"\ntidak terpusat (|geser| >= max(4, size/2)): {worst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
