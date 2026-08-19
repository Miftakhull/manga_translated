#!/usr/bin/env python3
"""Verifikasi condense 0.85 yang SUDAH terpasang, plus ukur sisa cacat r3.

Tiga hal, semuanya offline dan nol token:

1. fit() dijalankan apa adanya (tanpa patch) pada mask jp_6 yang sungguhan —
   berapa tanda hubung dan ukuran font yang tersisa sekarang.
2. Kesetiaan ukur-vs-gambar: baris dirender ke tile lewat jalur render_region
   yang sama, lalu lebar TINTA-nya diukur dari piksel. Kalau _measure() dan
   transform tile pecah, angka ini yang menangkapnya.
3. Kolom bebas per BARIS di r2 dan r3, sebelum dan sesudah
   disjoin_overlapping_interiors() — untuk melihat di baris mana saja r3
   kehilangan lebar dan apakah r2 benar-benar memakainya.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NBSRC, STAGE = ROOT / "_nbsrc", ROOT / ".stage"
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))
STAGE.mkdir(exist_ok=True)
for s in sorted(NBSRC.glob("*.py")):
    body = _MAGIC.sub("", s.read_text(encoding="utf-8"), count=1)
    d = STAGE / s.name
    if not d.exists() or d.read_text(encoding="utf-8") != body:
        d.write_text(body, encoding="utf-8")
sys.path.insert(0, str(STAGE))

import numpy as np                                        # noqa: E402
from PIL import Image, ImageDraw                          # noqa: E402
import detect, imgio, textmask, typeset                   # noqa: E402,E401
from config import SETTINGS                               # noqa: E402

REPORT = json.load(open(ROOT / "debug/jp_6/report.json", encoding="utf-8"))
OURS = {r["idx"]: r["translation"] for r in REPORT["regions"]}
REF = {
    0: "AND IN THE END...",
    1: "THEY ALL GOT TOO EXCITED AND MOSTLY CAME BY THEMSELVES!♥",
    2: "WELL, THEY'D NEVER SEEN ANYONE AS INSANELY SEXY AS YOU,",
    3: "SO IT'S ONLY NATURAL, RIGHT?♥",
    4: "IT'S EMBARASSING TO BE DESCRIBED THAT WAY...",
    5: "PLEASE DON'T TEASE ME SO MUCH...",
    6: "EHHH~? I'M PRAISING YOU FROM THE BOTTOM OF MY HEART, THOUGH?",
    7: "AH!",
}

fp = typeset.setup_fonts(verbose=False)
img = imgio.load_any(ROOT / "hasilnew/jp_6.JPG")
typeset.set_page_width(img.shape[1])
regions, _ = detect.detect(img)
soft = textmask.ctd_soft_mask(img)
for r in regions:
    textmask.build_region_mask(img, r, soft)
textmask.partition_shared_interiors(img, regions)
pre = {r.idx: typeset._region_box_mask(r) for r in regions}
pre = {k: (b, m.copy()) for k, (b, m) in pre.items()}
textmask.disjoin_overlapping_interiors(img, regions)
post = {r.idx: typeset._region_box_mask(r) for r in regions}
masks = {k: m for k, (_b, m) in post.items()}

print(f"condense={typeset._cond()}  halaman {img.shape[1]}x{img.shape[0]}  "
      f"min_font()={typeset.min_font()}  darurat={typeset.emergency_floor()}")


def sweep(label: str, texts: dict[int, str]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'r':>2} {'size':>4} {'hyph':>4} {'luber':>5}  baris")
    tot_h = tot_o = 0
    sizes = []
    for idx in sorted(masks):
        t = texts.get(idx, "").upper()
        if not t:
            continue
        size, lines, _y, over = typeset.fit(t, masks[idx],
                                            typeset.region_font_cap(masks[idx]), fp)
        bad = [ln for ln in lines if ln.endswith("-")]
        tot_h += len(bad)
        tot_o += int(over)
        sizes.append(size)
        print(f"{idx:>2} {size:>4} {len(bad):>4} {int(over):>5}  {' | '.join(lines)}")
    print(f"TOTAL hyphen={tot_h}  luber={tot_o}  "
          f"size {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")


sweep("terjemahan kita (report.json)", OURS)
sweep("wording typeset referensi CONTOH/6.JPG", REF)

# ---- 2. ukur vs gambar -------------------------------------------------------
print("\n=== kesetiaan ukur vs gambar (lebar tinta tile sungguhan) ===")
print(f"{'teks':>16} {'size':>4} {'_measure':>9} {'tinta':>6} {'selisih':>8}")
cnd = typeset._cond()
cmap = typeset._cmap(fp)
for txt, size in (("WONDER", 6), ("EMBARASSING", 7), ("NATURAL,", 6),
                  ("I'M PRAISING", 8), ("SO MUCH...", 7), ("AH!", 7)):
    font = typeset._font(fp, size)
    w = typeset._line_width(txt, font, cmap, size)
    lh = typeset._line_height(font)
    k = SETTINGS.oblique
    pad = int(abs(k) * lh) + 4
    wn = w / cnd
    tile = Image.new("RGBA", (int(wn) + pad * 2, lh + pad * 2), (0, 0, 0, 0))
    typeset._draw_line(ImageDraw.Draw(tile), (pad, pad), txt, font,
                       (0, 0, 0), cmap, size, 0)
    th = tile.height
    tile = tile.transform(tile.size, Image.AFFINE,
                          (1 / cnd, k / cnd, pad - (pad + k * th / 2) / cnd, 0, 1, 0),
                          resample=Image.BICUBIC)
    a = np.asarray(tile.getchannel("A"))
    cols = np.where((a > 24).any(axis=0))[0]
    ink = int(cols[-1] - cols[0] + 1) if cols.size else 0
    print(f"{txt:>16} {size:>4} {w:>9.1f} {ink:>6} {ink - w:>8.1f}")

# ---- 3. lebar per baris r2 vs r3 --------------------------------------------
print("\n=== kolom bebas per baris (sebelum -> sesudah disjoin) ===")
for idx in (2, 3):
    (pb, pm), (qb, qm) = pre[idx], post[idx]
    print(f"\nr{idx}: box {pb} mask {pm.shape[1]}x{pm.shape[0]}"
          f"  ->  box {qb} mask {qm.shape[1]}x{qm.shape[0]}")
    pad = int(min(qm.shape[:2]) * SETTINGS.pad_ratio)
    print(f"  pad={pad}   y(abs) | bebas_sebelum | bebas_sesudah")
    y0 = min(pb[1], qb[1])
    y1 = max(pb[3], qb[3])
    for ya in range(y0, y1, 6):
        def run(box, m):
            yl = ya - box[1]
            if yl < 0 or yl + 5 >= m.shape[0]:
                return None
            r = typeset._band_run(m, yl, yl + 5)
            return None if r is None else r[1] - r[0] + 1
        a, b = run(pb, pm), run(qb, qm)
        print(f"       {ya:>5} | {str(a):>13} | {str(b):>13}")
