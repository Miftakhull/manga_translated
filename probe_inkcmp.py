#!/usr/bin/env python3
"""Jarak TINTA ke garis balon: sebelum vs sesudah sumbu blok, diukur sama.

probe_gap.py melaporkan r10 turun ke 0 px setelah sumbu blok dipasang, dan itu
yang bikin perbaikan r6 belum bisa ditutup. Tapi edge_gap() di sana mengukur
KOTAK baris — setinggi band tinta, selebar advance font — dan kotak selalu lebih
besar dari glyph di dalamnya: huruf 'A' tidak mengisi sudut kotaknya, dan advance
menyertakan side bearing kanan yang kosong. Jadi 0 px di edge_gap belum berarti
tinta menyentuh apa pun.

Probe ini mengukur benda yang diikat kontrak selftest ("tinta tidak menyentuh
garis balon dan tidak keluar balon") pada DUA tata letak sekaligus, dengan alat
yang sama:

  LAMA  = varian V0 probe_axis.sim() — lebar simetris di centroid, satu cx.
          Ini reproduksi jalur sebelum perubahan, bukan tebakan.
  BARU  = jalur produksi sekarang, fit() + line_axis().

Dilaporkan per region: piksel tinta di luar interior, dan jarak tinta terdekat
ke luar-interior (1 = tinta di baris piksel terakhir sebelum tepi; 0 = tinta
sudah di luar).

    TEXTS=probe_opus5_clean.json python probe_inkcmp.py
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
from config import SETTINGS   # noqa: E402
from probe_axis import sim    # noqa: E402
from probe_inkgap import ink_layer  # noqa: E402


def measure(mask: np.ndarray, lines: list[str], top: int, size: int,
            fp: str, xs: list[int]) -> tuple[int, int]:
    """(piksel tinta di luar interior, jarak tinta terdekat ke luar interior).

    Digambar per baris di sumbunya sendiri supaya varian V1/V0 yang punya x
    berbeda per baris tetap terukur apa adanya.
    """
    inner = mask > 0
    dist = cv2.distanceTransform(inner.astype(np.uint8), cv2.DIST_L2, 3)
    outside, dmin = 0, 10**6
    for k, ln in enumerate(lines):
        lh = typeset._line_height(typeset._font(fp, size))
        ink = ink_layer(mask, [ln], top + k * lh, size, fp, xs[k]) > 96
        if not ink.any():
            continue
        outside += int((ink & ~inner).sum())
        dmin = min(dmin, int(dist[ink].min()))
    return outside, (dmin if dmin < 10**6 else -1)


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    texts = json.loads(TEXTS.read_text(encoding="utf-8"))

    print(f"{'idx':>3} | {'LAMA sz':>7} {'luar':>5} {'jarak':>5} "
          f"| {'BARU sz':>7} {'luar':>5} {'jarak':>5} | teks")
    old_o, old_d, new_o, new_d = [], [], [], []
    for r in regions:
        t = str(texts.get(str(r.idx), "")).upper()
        if not t:
            continue
        m = typeset._region_box_mask(r)[1]
        cap = typeset.region_font_cap(m)

        # LAMA: V0 = lebar simetris di centroid, satu cx untuk semua baris.
        got, osize = None, None
        for osize in range(cap, SETTINGS.min_font_size - 1, -1):
            got = sim(m, t, osize, fp, "sym", False, False)
            if got:
                break
        if got:
            ol, otop, oxs = got[0], got[1], got[2]
            oo, od = measure(m, ol, otop, osize, fp, oxs)
        else:
            oo, od, osize = -1, -1, 0

        # BARU: jalur produksi.
        nsize, nl, nsy, _ov = typeset.fit(t, m, cap, fp)
        if nl:
            nax = typeset.line_axis(m, nl, nsy, nsize, fp)
            no, nd = measure(m, nl, nsy, nsize, fp, [nax] * len(nl))
        else:
            no, nd, nsize = -1, -1, 0

        old_o.append(oo); old_d.append(od); new_o.append(no); new_d.append(nd)
        print(f"{r.idx:>3} | {osize:>7} {oo:>5} {od:>5} "
              f"| {nsize:>7} {no:>5} {nd:>5} | {t[:26]}")

    def ring(o, d, tag):
        print(f"{tag}: tinta_di_luar total={sum(v for v in o if v > 0)} "
              f"region={[i for i, v in enumerate(o) if v > 0]} | "
              f"jarak min={min(d)} median={np.median(d):.0f} "
              f"nol={sum(1 for v in d if v == 0)}/{len(d)}")
    print()
    ring(old_o, old_d, "LAMA")
    ring(new_o, new_d, "BARU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
