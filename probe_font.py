#!/usr/bin/env python3
"""Kalibrasi ukuran font tanpa OCR & tanpa DeepL.

Terjemahan diambil dari cache (`probe_font_texts.json` — hasil DeepL run
sebelumnya), jadi bagian lambat yang tersisa hanya detect + CTD; hasil keduanya
disimpan ke `.probe_cache.npz` supaya iterasi berikutnya langsung jalan.

Yang dilaporkan: final_font_size tiap region, sebarannya, jumlah tanda hubung,
dan tinggi huruf kapital terukur — dibanding target CONTOH/2.webp (~11.3 px pada
halaman setinggi 1577).
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
# Bisa ditimpa lewat env supaya wording alternatif diuji dengan typeset yang SAMA:
#   TEXTS=probe_llm_texts.json python probe_font.py
TEXTS = ROOT / os.environ.get("TEXTS", "probe_font_texts.json")
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

import cv2         # noqa: E402
import detect      # noqa: E402
import imgio       # noqa: E402
import textmask    # noqa: E402
import typeset     # noqa: E402


def _stage_regions(img: np.ndarray):
    """detect + mask + partisi + disjoin, di-cache karena ORT jalan di CPU."""
    if CACHE.exists():
        with CACHE.open("rb") as f:
            return pickle.load(f)
    regions, _ = detect.detect(img)
    soft = textmask.ctd_soft_mask(img)
    for r in regions:
        textmask.build_region_mask(img, r, soft)
    textmask.partition_shared_interiors(img, regions)
    textmask.disjoin_overlapping_interiors(img, regions)
    textmask.protect_bubble_outline(img, regions)
    with CACHE.open("wb") as f:
        pickle.dump(regions, f)
    return regions


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    h, w = img.shape[:2]
    typeset.setup_fonts(verbose=False)
    regions = _stage_regions(img)

    texts: dict[str, str] = json.loads(TEXTS.read_text(encoding="utf-8"))
    miss = [r.idx for r in regions if str(r.idx) not in texts]
    for r in regions:
        r.translation = texts.get(str(r.idx))
    if miss:
        print(f"[warn] tanpa terjemahan cache: {miss}")

    out = typeset.render_page(img, regions)
    fs = [r.final_font_size for r in regions if r.final_font_size]
    spread = max(fs) / min(fs) if fs else 0.0
    # Ukuran TIDAK lagi seragam per halaman: referensi terukur menskalakan teks
    # ke besar balon (probe_refnative.py, sebaran 2.08x). Jadi yang dinilai bukan
    # sebaran mentah melainkan seberapa dekat ukuran ke plafon proporsionalnya.
    print(f"model: proporsional balon (region_font_cap), "
          f"cap/min={typeset._REF_CAP_PER_MIN}")
    short = []
    for r in regions:
        m = typeset._region_box_mask(r)[1]
        capr = typeset.region_font_cap(m)
        feas = typeset._max_feasible(str(r.translation).upper(), m, typeset.FONT_USED)
        # Yang dinilai BUKAN plafon - ukuran. Plafon proporsional bisa lebih besar
        # daripada yang muat secara geometri (r9: plafon 16, feasible 11), dan
        # region seperti itu sudah dirender sebesar mungkin — bukan cacat. Target
        # yang benar = min(plafon, feasible); kekurangan terhadap ITU yang berarti
        # fit() menyisakan ruang. feasible == 0 berarti teksnya cuma muat dengan
        # penggalan, jadi tidak ada target utuh yang bisa dibandingkan.
        tgt = min(capr, feas) if feas else 0
        if tgt:
            short.append(tgt - r.final_font_size)
        print(f"  {r.idx:>3} min={min(m.shape[:2]):>3} plafon={capr:>3} "
              f"fin={r.final_font_size:>3} feasible={feas:>3} "
              f"target={tgt:>3} {r.lines}")
    print(f"\nfont size : {sorted(fs)}  spread={spread:.2f}  "
          f"(referensi 2.08)")
    print(f"kurang dari target: median={np.median(short):.1f} max={max(short)}")
    hy = [ln for r in regions for ln in (r.lines or []) if ln.endswith("-")]
    print(f"tanda hubung: {len(hy)} {hy}")

    ink = (np.abs(out.astype(np.int16) - img.astype(np.int16)).sum(2) > 120)
    n, _lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    hs = [st[i][3] for i in range(1, n) if 6 <= st[i][3] <= 60 and 3 <= st[i][2] <= 60]
    if hs:
        print(f"tinggi kapital p25/50/75 = {np.percentile(hs, [25, 50, 75]).round(1)} "
              f"(referensi 16.0 px pada 1812 -> ~13.9 px pada 1577)")
    from PIL import Image
    Image.fromarray(out).save(ROOT / "_cmp" / "probe_font.png")
    print(f"render -> _cmp/probe_font.png ({w}x{h})")
    # Lulus = tiap region dirender pada target yang bisa dicapai (min(plafon,
    # feasible)) atau paling banyak 1 px di bawahnya, dan tak ada yang jatuh di
    # bawah min_font_size.
    from config import SETTINGS
    bad = [r.idx for r in regions
           if r.final_font_size and r.final_font_size < SETTINGS.min_font_size]
    print(f"di bawah min_font_size: {bad}")
    return 0 if not bad and max(short) <= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
