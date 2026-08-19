#!/usr/bin/env python3
"""Render wording apa pun ke PLAT YANG SUDAH DI-INPAINT, bukan ke halaman asli.

Perlu ada karena probe_font.py memanggil render_page(img, ...) dengan `img` =
halaman ASLI. Itu benar untuk tujuannya (mengukur ukuran font, baris, hyphen —
semuanya geometri, tidak peduli latarnya), tapi gambarnya TIDAK BISA dibanding
mata ke CONTOH/2.webp: teks Jepangnya masih ada di bawah, jadi setiap balon
tampak "saling timpa" padahal itu ulah probe-nya.

Plat bersihnya diambil dari debug/jepang_002/09_cleaned.png — keluaran erase +
LaMa dari run_page.py, persis yang dipakai pipeline di
pipeline.py:164 `typeset.render_page(cleaned, regions)`.

    TEXTS=probe_llm2_opus5_clean.json python probe_render_clean.py
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
CACHE = ROOT / ".probe_cache.pkl"
CLEAN = ROOT / "debug" / "jepang_002" / "09_cleaned.png"
TEXTS = ROOT / os.environ.get("TEXTS", "probe_llm2_opus5_clean.json")
OUT = ROOT / "_cmp" / os.environ.get("OUT", "render_clean.png")
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


def main() -> int:
    from PIL import Image  # noqa: PLC0415

    typeset.setup_fonts(verbose=False)
    with CACHE.open("rb") as f:
        regions = pickle.load(f)
    plate = imgio.load_any(CLEAN)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    for r in regions:
        r.translation = texts.get(str(r.idx))
    miss = [r.idx for r in regions if not r.translation]
    if miss:
        print(f"[warn] tanpa terjemahan: {miss}")
    out = typeset.render_page(plate, regions)
    OUT.parent.mkdir(exist_ok=True)
    Image.fromarray(out).save(OUT)
    fs = sorted(r.final_font_size for r in regions if r.final_font_size)
    hy = [ln for r in regions for ln in (r.lines or []) if ln.endswith("-")]
    print(f"size={fs} spread={max(fs)/min(fs):.2f} hyphen={len(hy)} {hy}")
    print(f"-> {OUT.relative_to(ROOT)}  {out.shape[1]}x{out.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
