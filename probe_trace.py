#!/usr/bin/env python3
"""Jejak build() baris-per-baris: kenapa layout() menolak ukuran yang jelas muat.

probe_words.py menunjukkan SEMUA kata di region 10 dan 12 muat di baris terlebar
pada ukuran minimum, tapi layout() tetap gagal — jadi yang menolak bukan lebar
kata melainkan sesuatu di jalur vertikal. Di sini tiap langkah build() dicetak:
n tebakan awal, start_y, lebar tersedia per baris, dan syarat mana yang gagal.
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


def trace(text: str, mask: np.ndarray, size: int, fp: str, hyph: bool) -> None:
    font = typeset._font(fp, size)
    words = text.split()
    lh = typeset._line_height(font)
    ink_top, ink_bot = typeset._ink_band(fp, size)
    cx, cy = typeset._centroid(mask)
    mh, mw = mask.shape[:2]
    pad = int(min(mh, mw) * SETTINGS.pad_ratio)

    def avail_at(y_top: int) -> float:
        lo, hi = 0.0, float(mw)
        for _ in range(9):
            mid = (lo + hi) / 2
            if typeset._row_free(mask, y_top + ink_top, y_top + ink_bot,
                                 cx - mid / 2, cx + mid / 2):
                lo = mid
            else:
                hi = mid
        return max(lo - pad * 2, 0.0)

    def center_y(n: int) -> int:
        ink_h = (n - 1) * lh + (ink_bot - ink_top)
        return cy - ink_h // 2 - ink_top

    print(f"  size={size} lh={lh} ink_band={ink_top}..{ink_bot} "
          f"centroid=({cx},{cy}) pad={pad} mask={mw}x{mh}")
    n = max(1, int(np.ceil(typeset._measure(text, font) / max(mw - pad * 2, 1))))
    print(f"  tebakan awal n={n} (lebar teks {typeset._measure(text, font):.0f} / "
          f"{mw - pad*2})")
    for it in range(5):
        top = center_y(n)
        print(f"   iter{it}: n={n} start_y={top}")
        lines, queue, i = [], list(words), 0
        for _ in range(64):
            if i >= len(queue):
                break
            y = top + len(lines) * lh
            av = avail_at(y)
            if av < size * 0.9:
                print(f"     GAGAL di baris {len(lines)}: y={y} avail={av:.1f} "
                      f"< {size*0.9:.1f}")
                lines = None
                break
            line = queue[i]
            j = i + 1
            while j < len(queue) and typeset._measure(f"{line} {queue[j]}", font) <= av:
                line = f"{line} {queue[j]}"
                j += 1
            if j == i + 1 and typeset._measure(line, font) > av:
                head, tail = (typeset._split_word(line, font, av) if hyph
                              else ("", line))
                if head:
                    queue[i:i + 1] = [head, tail]
                    line, j = head, i + 1
                else:
                    print(f"     GAGAL: '{line}' ({typeset._measure(line, font):.0f}) "
                          f"> avail {av:.1f} dan tidak bisa dipenggal")
                    lines = None
                    break
            print(f"     baris {len(lines)}: y={y} avail={av:.1f} -> {line!r}")
            lines.append(line)
            i = j
        if lines is None:
            n += 1
            continue
        done = i >= len(queue)
        if done and len(lines) == n:
            sy = center_y(len(lines))
            bot = sy + (len(lines) - 1) * lh + ink_bot
            print(f"     n stabil. start_y={sy} tinta {sy+ink_top}..{bot} "
                  f"batas {pad}..{mh-pad}  "
                  f"{'LOLOS' if sy+ink_top >= pad and bot <= mh-pad else 'GAGAL batas vertikal'}")
            return
        n = max(1, len(lines) + (0 if done else 1))
    print("     iterasi habis tanpa n stabil")


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with PRE.open("rb") as f:
        regions = pickle.load(f)
    textmask.disjoin_overlapping_interiors(img, regions)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    SETTINGS.line_spacing = float(os.environ.get("LS", 0.95))
    SETTINGS.pad_ratio = float(os.environ.get("PAD", 0.06))
    want = {int(a) for a in sys.argv[1:]} or {12}
    for r in regions:
        if r.idx not in want:
            continue
        t = str(texts.get(str(r.idx), "")).upper()
        print(f"\n=== r{r.idx} {t!r}")
        trace(t, typeset._region_box_mask(r)[1], SETTINGS.min_font_size, fp, True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
