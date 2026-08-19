#!/usr/bin/env python3
"""Apakah balon KITA sama besar dengan balon REFERENSI, setelah skala disamakan?

Ini pertanyaan yang muncul dari kegagalan kalibrasi: validator anggaran menolak
wording referensi di r10 ("A SUMMARY OF EVERYTHING WE'VE DONE SINCE THIS
SPRING.", 53 karakter) karena di mask r10 KITA teks itu tidak muat utuh pada
ukuran mana pun — 'EVERYTHING' (10 huruf) lebih lebar daripada baris terlebar
yang tersedia. Padahal di CONTOH/2.webp kalimat itu jelas tercetak rapi.

Dua kemungkinan, dan akibatnya berbeda sama sekali:
    (a) Halaman referensi memang LEBIH BESAR (1812 px vs 1577 px, 1.149x), jadi
        kapasitas karakternya sama saja setelah dinormalkan. Kalau begitu yang
        salah cuma cara membandingkan, bukan mask kita.
    (b) Interior r10 kita memang lebih KECIL daripada balon aslinya — misalnya
        karena disjoin_overlapping_interiors memotong piksel yang diperebutkan
        terlalu banyak. Kalau begitu ini cacat kita sendiri, dan wording sependek
        apa pun tidak akan memperbaikinya.

Yang diukur: sisi terpendek interior tiap region di kedua halaman, dinormalkan
ke tinggi halamannya masing-masing. Kalau rasionya sepadan -> (a). Kalau balon
kita sistematis lebih kecil -> (b).
"""

from __future__ import annotations

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

import imgio    # noqa: E402
import typeset  # noqa: E402


def _sides(pkl: Path, page_h: int) -> dict[int, tuple[int, int]]:
    with pkl.open("rb") as f:
        regions = pickle.load(f)
    out = {}
    for r in regions:
        m = typeset._region_box_mask(r)[1]
        h, w = m.shape[:2]
        out[r.idx] = (min(h, w), int((m > 0).sum()))
    return out


def main() -> int:
    ours_img = imgio.load_any(ROOT / "jepang_002.webp")
    ref_img = imgio.load_any(ROOT / "CONTOH" / "2.webp")
    oh, ow = ours_img.shape[:2]
    rh, rw = ref_img.shape[:2]
    k = oh / rh
    print(f"kita {ow}x{oh}   referensi {rw}x{rh}   skala kita/ref = {k:.3f}")

    ours = _sides(ROOT / ".probe_cache.pkl", oh)
    ref = _sides(ROOT / ".probe_ref_native.pkl", rh)
    print(f"region: kita {len(ours)}  referensi {len(ref)}")

    # Pencocokan berdasar URUTAN BACA, bukan idx mentah: kedua halaman dideteksi
    # terpisah sehingga penomorannya belum tentu sejajar. Kalau jumlahnya sama,
    # urutan baca adalah pencocokan paling masuk akal yang tersedia tanpa
    # menandai manual.
    if len(ours) != len(ref):
        print("jumlah region beda -> pencocokan per idx tidak sah, cetak apa adanya")
    print(f"\n{'idx':>3} {'sisi_kita':>9} {'sisi_ref':>8} {'ref*skala':>9} "
          f"{'selisih':>7}")
    diffs = []
    for i in sorted(set(ours) & set(ref)):
        a = ours[i][0]
        b = ref[i][0]
        bs = b * k
        d = a - bs
        diffs.append(d)
        print(f"{i:>3} {a:>9} {b:>8} {bs:>9.1f} {d:>+7.1f}")
    if diffs:
        print(f"\nselisih sisi terpendek (kita - ref*skala): "
              f"median {np.median(diffs):+.1f} px  min {min(diffs):+.1f}  "
              f"max {max(diffs):+.1f}")
        print("Negatif berarti balon kita lebih kecil daripada balon referensi "
              "pada skala yang sama.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
