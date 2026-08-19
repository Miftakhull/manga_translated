#!/usr/bin/env python3
"""Uji kriteria anggaran pada MASK REFERENSI, bukan mask kita.

Kalibrasi versi pertama menjalankan wording referensi melewati mask KITA, dan
menyimpulkan validatornya salah. Kesimpulan itu sendiri belum sah: probe_scale.py
mengukur interior kita rata-rata 8.6 px LEBIH KECIL daripada interior referensi
pada skala yang sama (median; r10 -10.3 px), dan kedua halaman itu bahkan bukan
resize satu sama lain — rasio lebar 0.886 versus rasio tinggi 0.870, jadi
aspeknya beda. Menuntut kalimat referensi muat di mask kita berarti menuntut 53
karakter masuk ke balon yang 9% lebih sempit daripada balon yang dipakai
typesetter aslinya.

Jadi uji yang benar: wording referensi di mask REFERENSI. Kalau di sana ia lolos
kriteria (feasible >= cap - slack), kriterianya sehat dan yang berbeda cuma
kapasitas balon kita. Kalau di sana pun gagal, kriterianya memang keliru.
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

import typeset                # noqa: E402
from config import SETTINGS   # noqa: E402

REF = json.loads((ROOT / "probe_ref_texts.json").read_text(encoding="utf-8"))
SLACK = 2


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_ref_native.pkl").open("rb") as f:
        rregions = pickle.load(f)
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        oregions = pickle.load(f)
    rm = {r.idx: r for r in rregions}
    om = {r.idx: r for r in oregions}

    print(f"{'idx':>3} | {'sisi':>4} {'cap':>3} {'feas':>4} {'sel':>4} pada MASK "
          f"REFERENSI | {'sisi':>4} {'cap':>3} {'feas':>4} {'sel':>4} pada mask kita")
    bad_ref, bad_ours = [], []
    for k in sorted(REF, key=int):
        i = int(k)
        t = REF[k]
        row = []
        for tag, mp in (("ref", rm), ("ours", om)):
            r = mp.get(i)
            if r is None:
                row.append((0, 0, 0, 0))
                continue
            m = typeset._region_box_mask(r)[1]
            cap = typeset.region_font_cap(m)
            feas = typeset._max_feasible(t, m, fp)
            row.append((int(min(m.shape[:2])), cap, feas, feas - cap))
            if feas < cap - SLACK or feas < SETTINGS.min_font_size:
                (bad_ref if tag == "ref" else bad_ours).append(i)
        a, b = row
        print(f"{i:>3} | {a[0]:>4} {a[1]:>3} {a[2]:>4} {a[3]:>+4} "
              f"            | {b[0]:>4} {b[1]:>3} {b[2]:>4} {b[3]:>+4}")

    print(f"\nwording referensi GAGAL kriteria di mask referensi : {bad_ref}")
    print(f"wording referensi GAGAL kriteria di mask kita       : {bad_ours}")
    print("\nKesimpulan:")
    if not bad_ref and bad_ours:
        print("  Kriterianya sehat — wording referensi lolos di balon aslinya.")
        print("  Yang berbeda kapasitas balon KITA (probe_scale.py: median -8.6 px).")
        print("  Jadi anggaran karakter WAJIB dihitung dari mask kita sendiri,")
        print("  dan wording referensi bukan target panjang yang bisa dicapai.")
    elif bad_ref:
        print(f"  Kriteria menolak referensi di balon aslinya juga ({bad_ref}) ->")
        print("  kriterianya yang keliru, bukan kapasitas balon.")
    else:
        print("  Keduanya lolos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
