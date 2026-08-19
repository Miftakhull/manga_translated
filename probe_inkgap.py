#!/usr/bin/env python3
"""Piksel TINTA SUNGGUHAN di luar interior balon, dan jaraknya ke garis balon.

edge_gap() di probe_margin.py mengukur KOTAK baris, bukan tintanya: kotak
setinggi band tinta dan selebar advance font selalu lebih besar dari glyph-nya
sendiri, jadi angka 0 di sana belum tentu berarti tinta menyentuh garis. Yang
mengikat adalah kontrak di selftest: "tinta tidak menyentuh garis balon dan
tidak keluar balon". Probe ini mengukur itu pada halaman nyata — glyph digambar
persis seperti render_region (font, ukuran, sumbu, shear) lalu dibandingkan ke
mask interior.

Dicetak per region: jumlah piksel tinta di luar interior, jarak minimum tinta
ke tepi interior, dan jarak minimum ke luar-interior (0 = menempel garis).

    TEXTS=probe_opus5_clean.json python probe_inkgap.py
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
TEXTS = ROOT / os.environ.get("TEXTS", "probe_opus5_clean.json")
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

import cv2       # noqa: E402
import typeset   # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from config import SETTINGS       # noqa: E402


def ink_layer(mask: np.ndarray, lines: list[str], top: int, size: int,
              fp: str, cx: int) -> np.ndarray:
    """Bitmap tinta di koordinat mask — meniru render_region apa adanya."""
    mh, mw = mask.shape[:2]
    font = typeset._font(fp, size)
    cmap = typeset._cmap(fp)
    lh = typeset._line_height(font)
    k = SETTINGS.oblique
    pad = int(abs(k) * lh) + 4
    out = Image.new("L", (mw, mh), 0)
    for i, line in enumerate(lines):
        w = typeset._line_width(line, font, cmap, size)
        tile = Image.new("L", (int(w) + pad * 2, lh + pad * 2), 0)
        typeset._draw_line(ImageDraw.Draw(tile), (pad, pad), line, font,
                           255, cmap, size, 0)
        if k:
            th = tile.height
            tile = tile.transform(tile.size, Image.AFFINE,
                                  (1, k, -k * th / 2, 0, 1, 0),
                                  resample=Image.BICUBIC)
        tx = int(cx - w / 2) - pad
        ty = top + i * lh - pad
        # Tempel dengan klip manual; di luar kotak mask dianggap "di luar balon"
        # dan tetap dihitung, karena itu justru pelanggaran yang dicari.
        x0, y0 = max(0, -tx), max(0, -ty)
        x1 = min(tile.width, mw - tx)
        y1 = min(tile.height, mh - ty)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = tile.crop((x0, y0, x1, y1))
        base = out.crop((tx + x0, ty + y0, tx + x1, ty + y1))
        out.paste(Image.fromarray(np.maximum(np.asarray(base), np.asarray(sub))),
                  (tx + x0, ty + y0))
    return np.asarray(out)


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    print(f"{'idx':>3} {'size':>4} {'nb':>3} {'luar_px':>8} {'jarak_min':>10}  baris")
    outs, dists = [], []
    for r in regions:
        t = str(texts.get(str(r.idx), "")).upper()
        if not t:
            continue
        m = typeset._region_box_mask(r)[1]
        size, lines, sy, _ov = typeset.fit(t, m, typeset.region_font_cap(m), fp)
        if not lines:
            continue
        cx = typeset.line_axis(m, lines, sy, size, fp)
        ink = ink_layer(m, lines, sy, size, fp, cx) > 96
        inner = (m > 0)
        outside = int((ink & ~inner).sum())
        # Jarak tinta terdekat ke LUAR interior: 1 = piksel tinta persis di tepi.
        dist = cv2.distanceTransform(inner.astype(np.uint8), cv2.DIST_L2, 3)
        dmin = int(dist[ink].min()) if ink.any() else -1
        outs.append(outside)
        dists.append(dmin)
        print(f"{r.idx:>3} {size:>4} {len(lines):>3} {outside:>8} {dmin:>10}  "
              f"{' / '.join(lines)[:38]}")
    print(f"\ntinta di luar interior: total={sum(outs)} "
          f"region_bermasalah={[i for i, v in enumerate(outs) if v]}")
    print(f"jarak tinta->luar: min={min(dists)} median={np.median(dists):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
