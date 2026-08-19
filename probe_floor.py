#!/usr/bin/env python3
"""Ukuran TERKECIL yang masih memuat teks utuh, untuk region yang layak=0.

Ditulis saat fit() masih jatuh ke `low_lo = max(min_font_size // 2, 4)` begitu
kedua kandidat nihil, dan itulah yang merender region 6 pada 6 px. Sebelum
memilih aturan degradasi, angkanya harus diketahui: pada ukuran berapa teks itu
SUNGGUH muat utuh? Kalau 9-10 px, cukup turunkan lantai; kalau 6-7 px,
masalahnya wording, bukan lantai.

HASIL: keduanya. Lantai sekarang konstanta bernama typeset._MIN_FONT_FLOOR (9),
dan pada pad_ratio 0.04 halaman referensi tidak menyentuhnya sama sekali — r6
dan r10 dirender 11 px. Probe ini disimpan karena kolomnya (P=utuh, h=hyphen,
-=tidak muat) adalah cara tercepat memisahkan 'balonnya sempit' dari 'wording-
nya panjang' untuk halaman lain.
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
PRE = ROOT / ".probe_pre.pkl"
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

# Wording referensi (probe_pair.py) untuk region yang mepet — supaya terlihat
# apakah yang menjepit itu geometri atau panjang kalimat DeepL.
REF_TEXT = {
    6: "SORRY.",
    10: "A SUMMARY OF EVERYTHING WE'VE DONE SINCE THIS SPRING.",
    3: "PREZ!",
    5: "SHIZUKU-SAN.",
    4: "OH MY.",
}


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with PRE.open("rb") as f:
        regions = pickle.load(f)
    textmask.disjoin_overlapping_interiors(img, regions)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    SETTINGS.line_spacing = float(os.environ.get("LS", 0.95))
    SETTINGS.pad_ratio = float(os.environ.get("PAD", 0.05))
    print(f"line_spacing={SETTINGS.line_spacing} pad_ratio={SETTINGS.pad_ratio}")
    want = {int(a) for a in sys.argv[1:]} or {6, 10}

    for r in regions:
        if r.idx not in want:
            continue
        m = typeset._region_box_mask(r)[1]
        mh, mw = m.shape[:2]
        for tag, t in (("kita", str(texts.get(str(r.idx), "")).upper()),
                       ("REF ", REF_TEXT.get(r.idx, "").upper())):
            if not t:
                continue
            row = []
            for size in range(5, 15):
                okp = typeset.layout(t, m, size, fp, hyphenate=False)[0]
                okh = typeset.layout(t, m, size, fp, hyphenate=True)[0]
                row.append(f"{size}:{'P' if okp else ('h' if okh else '-')}")
            print(f"  r{r.idx:<2} {tag} mask={mw}x{mh} {t!r}")
            print(f"        {' '.join(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
