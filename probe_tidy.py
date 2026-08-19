#!/usr/bin/env python3
"""Margin & keterisian kita, diukur PERSIS seperti probe_refnative.py mengukur referensi.

Ini memperbaiki perbandingan yang tidak setara. probe_refnative.py memakai lebar
interior GABUNGAN sepanjang blok tinta sebagai penyebut (bxs = kolom interior di
rentang baris tinta, disatukan), sedangkan probe_fill/probe_cal memakai lebar
baris terlebar ITU SENDIRI. Penyebut gabungan selalu >= penyebut per-baris, jadi
'isi 83% kita vs 70% referensi' dan 'sisi 0.082 vs 0.165' membandingkan dua
besaran berbeda — tidak boleh dipakai memilih pad_ratio.

Di sini teks benar-benar DIRENDER ke halaman yang interiornya diputihkan, lalu
tintanya diukur dengan kode yang sama seperti sisi referensi: cap_height dari
komponen terhubung, margin sisi dari kolom interior gabungan, isi dari
(xs.max-xs.min+1)/lebar_interior_gabungan.
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

import cv2       # noqa: E402
import imgio     # noqa: E402
import textmask  # noqa: E402
import typeset   # noqa: E402
from config import SETTINGS  # noqa: E402

CAP_PER_SIZE, REF_RATIO = 0.844, 0.117
REF_TEXT = {
    0: "AH! FINALLY FOUND YOU!", 1: "SO THIS IS WHERE YOU WERE!",
    2: "I'VE BEEN LOOKING ALL OVER FOR YOU.", 3: "PREZ!", 4: "OH MY.",
    5: "SHIZUKU-SAN.", 6: "SORRY.",
    7: "IT'S JUST THAT IT'S QUIET AND RELAXING HERE AFTER SCHOOL...",
    8: "WHAT WERE YOU DOING IN A PLACE LIKE THIS?",
    9: "I WAS PUTTING TOGETHER THE STUDENT COUNCIL'S ACTIVITY RECORDS.",
    10: "A SUMMARY OF EVERYTHING WE'VE DONE SINCE THIS SPRING.",
    11: "OH, IS THAT FOR THE MILKING CLUB?", 12: "COME ON, LET ME SEE~!",
}


def blank(img: np.ndarray, regions) -> np.ndarray:
    """Halaman dengan interior balon diputihkan — pengganti hasil inpaint.

    Perlu supaya tinta Inggris bisa dipisahkan dari tinta Jepang tanpa
    menjalankan LaMa: ambang gray<110 di dalam mask lalu hanya menangkap
    huruf yang baru digambar.
    """
    out = img.copy()
    h, w = out.shape[:2]
    for r in regions:
        (bx1, by1, _, _), m = typeset._region_box_mask(r)
        mh, mw = m.shape[:2]
        y2, x2 = min(by1 + mh, h), min(bx1 + mw, w)
        sub = m[: y2 - by1, : x2 - bx1] > 0
        out[by1:y2, bx1:x2][sub] = 255
    return out


def measure(page: np.ndarray, region, gray_before: np.ndarray) -> tuple | None:
    h, w = page.shape[:2]
    (bx1, by1, _, _), mask = typeset._region_box_mask(region)
    mh, mw = mask.shape[:2]
    y2, x2 = min(by1 + mh, h), min(bx1 + mw, w)
    m = mask[: y2 - by1, : x2 - bx1] > 0
    gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)[by1:y2, bx1:x2]
    ink = (m & (gray < 110) & (gray_before[by1:y2, bx1:x2] >= 110)).astype(np.uint8)
    if int(ink.sum()) < 40:
        return None
    n, _lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    hs = [st[i][3] for i in range(1, n)
          if 4 <= st[i][3] <= 60 and 2 <= st[i][2] <= 60 and st[i][4] >= 6]
    if len(hs) < 4:
        return None
    cap = float(np.median(hs))
    ys, xs = np.nonzero(ink)
    rows = slice(int(ys.min()), int(ys.max()) + 1)
    bxs = np.nonzero(m[rows].any(0))[0]
    side = ((xs.min() - bxs.min()) + (bxs.max() - xs.max())) / 2
    mn = min(mh, mw)
    fill = (xs.max() - xs.min() + 1) / max(len(bxs), 1)
    return cap, cap / mn, side / mn, fill


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with PRE.open("rb") as f:
        regions = pickle.load(f)
    textmask.disjoin_overlapping_interiors(img, regions)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))
    base = blank(img, regions)
    gray0 = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    masks = {r.idx: typeset._region_box_mask(r)[1] for r in regions}
    which = os.environ.get("WORDING", "ours")

    print("REFERENSI (probe_refnative.py, metode identik): "
          "cap/min=0.117  sisi/min=0.165  isi=70%")
    print(f"wording={which}  line_spacing={os.environ.get('LS', '1.00')}")
    print(f"  {'pad':>5} | {'cap/min':>8} {'sisi/min':>9} {'isi':>5} {'n':>3}")
    SETTINGS.line_spacing = float(os.environ.get("LS", 1.00))
    for pad in (0.04, 0.05, 0.06, 0.08, 0.10, 0.12):
        SETTINGS.pad_ratio = pad
        page = base.copy()
        for r in regions:
            t = (REF_TEXT.get(r.idx, "") if which == "ref"
                 else str(texts.get(str(r.idx), "")).upper())
            if not t:
                continue
            r.translation, r.label = t, "DIALOGUE"
            mn = min(masks[r.idx].shape[:2])
            page = typeset.render_region(
                page, r, fp, size_cap=int(round(mn * REF_RATIO / CAP_PER_SIZE)))
        got = [measure(page, r, gray0) for r in regions]
        got = [g for g in got if g]
        if not got:
            print(f"  {pad:>5.2f} | (tidak terukur)")
            continue
        arr = np.array(got)
        print(f"  {pad:>5.2f} | {np.median(arr[:,1]):>8.3f} "
              f"{np.median(arr[:,2]):>9.3f} {np.median(arr[:,3])*100:>4.0f}% "
              f"{len(got):>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
