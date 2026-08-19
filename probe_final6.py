#!/usr/bin/env python3
"""Render ulang jp_6 di PLAT BERSIH dengan kode typeset sekarang.

Kenapa perlu file sendiri, bukan run_page.py: run_page memanggil OCR + DeepL,
padahal yang diuji di sini murni geometri typeset (per-baris reclaim, condense,
hyphen). Wording-nya diambil apa adanya dari debug/jp_6/report.json — hasil
terjemahan yang SAMA dengan run terakhir — jadi perbedaan gambar hanya bisa
berasal dari perubahan typeset, bukan dari wording yang berubah.

Plat bersihnya debug/jp_6/09_cleaned.png, persis yang dipakai
pipeline.py `typeset.render_page(cleaned, regions)`.

Tahap detect + mask makan >120 s di CPU, jadi hasilnya di-pickle ke
.probe_cache6.pkl; render berikutnya instan. Hapus file itu kalau detect atau
textmask diubah.

    python probe_final6.py            # pakai cache kalau ada
    FRESH=1 python probe_final6.py    # paksa deteksi ulang
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NBSRC, STAGE = ROOT / "_nbsrc", ROOT / ".stage"
CACHE = ROOT / ".probe_cache6.pkl"
PAGE = ROOT / "hasilnew" / "jp_6.JPG"
CLEAN = ROOT / "debug" / "jp_6" / "09_cleaned.png"
REPORT = ROOT / "debug" / "jp_6" / "report.json"
OUT = ROOT / "_cmp" / os.environ.get("OUT", "render6_clean.png")
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))
STAGE.mkdir(exist_ok=True)
for _s in sorted(NBSRC.glob("*.py")):
    _b = _MAGIC.sub("", _s.read_text(encoding="utf-8"), count=1)
    _d = STAGE / _s.name
    if not _d.exists() or _d.read_text(encoding="utf-8") != _b:
        _d.write_text(_b, encoding="utf-8")
sys.path.insert(0, str(STAGE))

import imgio     # noqa: E402
import typeset   # noqa: E402


def build_regions(img):
    """detect + mask + partisi + disjoin: urutan yang sama dengan pipeline.py."""
    import detect, textmask                                  # noqa: PLC0415,E401
    regions, _ = detect.detect(img)
    soft = textmask.ctd_soft_mask(img)
    for r in regions:
        textmask.build_region_mask(img, r, soft)
    textmask.partition_shared_interiors(img, regions)
    textmask.disjoin_overlapping_interiors(img, regions)
    textmask.protect_bubble_outline(img, regions)
    return regions


def main() -> int:
    from PIL import Image                                    # noqa: PLC0415

    typeset.setup_fonts(verbose=False)
    img = imgio.load_any(PAGE)
    typeset.set_page_width(img.shape[1])
    if CACHE.exists() and not os.environ.get("FRESH"):
        with CACHE.open("rb") as f:
            regions = pickle.load(f)
        print(f"[cache] {CACHE.name} ({len(regions)} region)")
    else:
        regions = build_regions(img)
        with CACHE.open("wb") as f:
            pickle.dump(regions, f)
        print(f"[fresh] deteksi ulang -> {CACHE.name} ({len(regions)} region)")

    texts = {r["idx"]: r["translation"] for r in
             json.loads(REPORT.read_text(encoding="utf-8"))["regions"]}
    for r in regions:
        r.translation = texts.get(r.idx) or ""
    miss = [r.idx for r in regions if not r.translation]
    if miss:
        print(f"[warn] tanpa terjemahan: {miss}")

    plate = imgio.load_any(CLEAN)
    out = typeset.render_page(plate, regions)
    OUT.parent.mkdir(exist_ok=True)
    Image.fromarray(out).save(OUT)

    print(f"\n{'r':>2} {'size':>5} {'baris':>6} {'hyph':>5} {'luber':>6}  isi")
    hy = 0
    for r in sorted(regions, key=lambda q: q.idx):
        ln = r.lines or []
        h = sum(1 for x in ln if x.endswith("-"))
        hy += h
        print(f"{r.idx:>2} {r.final_font_size or 0:>5} {len(ln):>6} {h:>5} "
              f"{int(bool(r.overflowed)):>6}  {' | '.join(ln)}")
    fs = [r.final_font_size for r in regions if r.final_font_size]
    print(f"\nsize={sorted(fs)} spread={max(fs) / min(fs):.2f} hyphen_total={hy} "
          f"luber_total={sum(1 for r in regions if r.overflowed)}")
    print(f"-> {OUT.relative_to(ROOT)}  {out.shape[1]}x{out.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
