#!/usr/bin/env python3
"""Sandingkan balon kita dengan balon referensi, MASING-MASING di frame sendiri.

probe_align.py membuktikan CONTOH/2.webp bukan hasil resize halaman kita
(geseran per-kuadran -28..+9 px), jadi crop di koordinat kita bisa jatuh di
balon sebelah. Di sini tiap sisi dipotong pakai kotak balon yang dideteksi di
gambarnya SENDIRI, lalu diskalakan ke tinggi tampil sama.

Gunanya dua: memastikan pasangan idx benar (urutan baca), dan melihat berapa
baris + sebesar apa font yang dipakai referensi di balon yang sama.
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
PRE = ROOT / ".probe_pre.pkl"
REFC = ROOT / ".probe_ref_native.pkl"
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

import imgio    # noqa: E402
import typeset  # noqa: E402
from PIL import Image  # noqa: E402

_H = 210  # tinggi tampil tiap crop


def crops(pkl: Path, page: np.ndarray) -> dict[int, Image.Image]:
    h, w = page.shape[:2]
    with pkl.open("rb") as f:
        regions = pickle.load(f)
    out = {}
    for r in regions:
        bx1, by1, bx2, by2 = r.bubble_bbox or r.bbox
        bx1, by1 = max(bx1 - 4, 0), max(by1 - 4, 0)
        bx2, by2 = min(bx2 + 4, w), min(by2 + 4, h)
        if bx2 - bx1 < 8 or by2 - by1 < 8:
            continue
        c = Image.fromarray(page[by1:by2, bx1:bx2])
        sc = _H / c.height
        out[r.idx] = c.resize((max(int(c.width * sc), 1), _H), Image.LANCZOS)
    return out


def main() -> int:
    typeset.setup_fonts(verbose=False)
    ours_page = np.asarray(
        Image.open(ROOT / "_cmp" / "probe_font.png").convert("RGB"), np.uint8)
    ref_page = imgio.load_any(ROOT / "CONTOH" / "2.webp")
    a, b = crops(PRE, ours_page), crops(REFC, ref_page)

    rows = []
    for i in sorted(set(a) | set(b)):
        ca = a.get(i) or Image.new("RGB", (60, _H), (245, 245, 245))
        cb = b.get(i) or Image.new("RGB", (60, _H), (245, 245, 245))
        strip = Image.new("RGB", (ca.width + cb.width + 6, _H), (220, 40, 40))
        strip.paste(ca, (0, 0))
        strip.paste(cb, (ca.width + 6, 0))
        rows.append((i, strip))

    # Dua baris supaya lembarnya tidak jadi pita panjang tak terbaca.
    half = (len(rows) + 1) // 2
    for n, part in enumerate((rows[:half], rows[half:])):
        wtot = sum(s.width + 10 for _, s in part)
        sheet = Image.new("RGB", (wtot, _H + 18), (255, 255, 255))
        x = 0
        for i, s in part:
            sheet.paste(s, (x, 18))
            x += s.width + 10
        sheet.save(ROOT / "_cmp" / f"pair_{n}.png")
        print(f"_cmp/pair_{n}.png  idx {[i for i, _ in part]} "
              f"(kiri=kita, kanan=referensi)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
