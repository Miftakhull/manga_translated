#!/usr/bin/env python3
"""Uji model ukuran font terhadap ukuran referensi yang TERUKUR.

Temuan probe_refnative.py: referensi TIDAK memakai satu ukuran seragam untuk
seluruh halaman. cap_height-nya 13..27 px (sebaran 2.08x) dan yang jauh lebih
stabil adalah cap_height / min(sisi interior) = 0.117 (p25 0.108, p75 0.150) —
balon besar dapat teks besar. Balon 1-baris ('PREZ!', 'OH MY.') justru rasio
tertinggi karena teksnya pendek dan diisi penuh.

Tiga model dibandingkan terhadap ukuran referensi-setara per region:
  A  seragam-halaman (perilaku sekarang): persentil 35 dari ukuran layak
  B  proporsional balon: 0.117 * min(mh,mw) / 0.844, dibatasi kelayakan
  C  proporsional lalu diselaraskan PER PANEL (kontingensi di plan.txt)

Nilai galat = rata-rata |ukuran_model - ukuran_referensi| dalam piksel font.
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

CAP_PER_SIZE = 0.844      # probe_cap.py, Anime Ace
REF_RATIO = 0.117         # cap / min(sisi) di referensi
REF_SCALE = 1577 / 1812   # tinggi halaman kita / tinggi referensi
# cap_height terukur di CONTOH/2.webp (probe_refnative.py), per idx.
REF_CAP = {0: 17, 1: 20, 2: 19, 3: 26, 4: 19, 5: 27, 6: 15,
           7: 13, 8: 14, 9: 14, 10: 13, 11: 14, 12: 16}


def panels(regions) -> dict[int, int]:
    """Kelompokkan region per panel lewat celah vertikal antar bbox.

    Panel manga dipisah garis mendatar; celah y antar kelompok jauh lebih besar
    dari jarak antar balon di satu panel. Cukup 1-D: di halaman ini tiap panel
    menempati pita y sendiri.
    """
    rs = sorted(regions, key=lambda r: r.bbox[1])
    hs = [r.bbox[3] - r.bbox[1] for r in rs]
    gap = max(int(np.median(hs) * 0.6), 20)
    out, grp, prev_bot = {}, 0, None
    for r in rs:
        if prev_bot is not None and r.bbox[1] - prev_bot > gap:
            grp += 1
        out[r.idx] = grp
        prev_bot = max(prev_bot or 0, r.bbox[3])
    return out


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
    print(f"line_spacing={SETTINGS.line_spacing} pad_ratio={SETTINGS.pad_ratio}")

    pan = panels(regions)
    ref = {i: REF_CAP[i] * REF_SCALE / CAP_PER_SIZE for i in REF_CAP}
    rows = {}
    for r in regions:
        t = str(texts.get(str(r.idx), "")).upper()
        m = typeset._region_box_mask(r)[1]
        mh, mw = m.shape[:2]
        pad = int(min(mh, mw) * SETTINGS.pad_ratio)
        hi = int(np.clip(mh - 2 * pad, SETTINGS.min_font_size, SETTINGS.max_font_size))
        p = typeset._search(t, m, SETTINGS.min_font_size, hi, fp, False)
        hy = typeset._search(t, m, SETTINGS.min_font_size, hi, fp, True)
        rows[r.idx] = {
            "panel": pan[r.idx],
            "min": min(mh, mw),
            "plain": p[0] if p else 0,
            "hyph": hy[0] if hy else 0,
            "ideal": min(mh, mw) * REF_RATIO / CAP_PER_SIZE,
            "ref": ref.get(r.idx, 0.0),
        }

    feas = {i: max(v["plain"], v["hyph"]) for i, v in rows.items()}
    nz = [v for v in feas.values() if v]
    A = int(np.percentile(nz, 35))
    mA = {i: (min(A, v) if v else 0) for i, v in feas.items()}
    mB = {i: (min(round(rows[i]["ideal"]), v) if v else 0) for i, v in feas.items()}
    mC = {}
    for g in set(pan.values()):
        ids = [i for i in rows if rows[i]["panel"] == g]
        cand = [min(rows[i]["ideal"], feas[i]) for i in ids if feas[i]]
        t = int(round(float(np.median(cand)))) if cand else 0
        for i in ids:
            mC[i] = min(t, feas[i]) if feas[i] else 0
    # D: isi penuh — ambil ukuran layak terbesar, tanpa plafon proporsional.
    # Hipotesis: sisa galat model B semuanya di balon berteks PENDEK ('PREZ!',
    # 'OH MY.') yang di referensi dibesarkan sampai memenuhi balon.
    mD = dict(feas)
    # E: plafon proporsional yang dilonggarkan k kali — kompromi antara B dan D.
    mE = {}
    kbest, ebest = 1.0, 1e9
    for k in np.arange(1.0, 2.05, 0.05):
        cand = {i: (min(round(rows[i]["ideal"] * k), v) if v else 0)
                for i, v in feas.items()}
        ok = [i for i in cand if cand[i] and rows[i]["ref"]]
        e = float(np.mean([abs(cand[i] - rows[i]["ref"]) for i in ok]))
        if e < ebest:
            kbest, ebest, mE = float(k), e, cand

    print(f"\n  {'idx':>3} {'pan':>3} {'min':>4} {'utuh':>5} {'hyph':>5} "
          f"{'ideal':>6} {'REF':>5} {'A_seragam':>10} {'B_prop':>7} {'C_panel':>8}"
          f" {'D_isi':>6} {'E_kx':>5}")
    for i in sorted(rows):
        v = rows[i]
        print(f"  {i:>3} {v['panel']:>3} {v['min']:>4} {v['plain']:>5} {v['hyph']:>5} "
              f"{v['ideal']:>6.1f} {v['ref']:>5.1f} {mA[i]:>10} {mB[i]:>7} {mC[i]:>8}"
              f" {mD[i]:>6} {mE[i]:>5}")

    for name, mm in (("A seragam-halaman", mA), ("B proporsional", mB),
                     ("C proporsional per panel", mC), ("D isi-penuh", mD),
                     (f"E proporsional x{kbest:.2f}", mE)):
        ok = [i for i in mm if mm[i] and rows[i]["ref"]]
        err = float(np.mean([abs(mm[i] - rows[i]["ref"]) for i in ok]))
        vals = [mm[i] for i in ok]
        print(f"{name:<26} galat_rata2={err:>5.2f} px  "
              f"sebaran={max(vals)/min(vals):.2f}  nol={sum(1 for v in mm.values() if not v)}")
    refv = [rows[i]["ref"] for i in rows if rows[i]["ref"]]
    print(f"{'REFERENSI itu sendiri':<26} galat_rata2= 0.00 px  "
          f"sebaran={max(refv)/min(refv):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
