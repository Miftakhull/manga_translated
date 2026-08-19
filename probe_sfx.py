#!/usr/bin/env python3
"""Apakah `sfx_idx: []` itu aman atau cacat? Diukur, bukan ditebak.

run_page.py menandai `sfx_idx kosong` sebagai GAGAL, dan itu benar sebagai
kecurigaan: halaman ini JELAS punya SFX — ノノノ di panel tengah (garis gerak) dan
三 kecil di panel kiri bawah. Kalau keduanya diklasifikasi DIALOGUE, mereka akan
diterjemahkan dan dihapus, dan itu melanggar syarat plan.txt ("NO TRANSLATE SFX").

Tapi debug/jepang_002/03_boxes.png memperlihatkan hal lain: detektor tidak
mengeluarkan kotak APA PUN di sana. 13 kotak, semuanya DIAL. Jadi ada dua
kemungkinan yang berbeda akibatnya:
    (a) SFX tidak terdeteksi -> tidak masuk erase_mask -> tetap utuh. AMAN.
        `sfx_idx: []` cuma berarti 'tidak ada yang perlu dijaga eksplisit'.
    (b) SFX kebetulan berada di dalam mask region tetangga -> ikut terhapus.
        CACAT sungguhan, dan assert_sfx_intact() tidak akan menangkapnya karena
        protected_mask-nya kosong (baris 79-80 verify.py langsung return True).

Yang membedakan (a) dari (b) cuma satu: apakah piksel SFX berubah antara input
dan hasil akhir. Itu yang diukur di sini, per kotak SFX, dalam piksel.
"""

from __future__ import annotations

import os
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
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))

import imgio  # noqa: E402

# Kotak SFX dibaca manual dari 03_boxes.png pada halaman 1134x1577. Sengaja
# manual: kalau diambil dari detektor, probe ini akan selalu kosong — justru
# ketiadaan deteksi itulah yang sedang diperiksa.
SFX_BOX = {
    "nonono (panel tengah)": (566, 487, 628, 553),
    "san (panel kiri bawah)": (300, 1178, 344, 1232),
}


def main() -> int:
    src = imgio.load_any(ROOT / "jepang_002.webp")
    cleaned = imgio.load_any(ROOT / "debug" / "jepang_002" / "09_cleaned.png")
    final = imgio.load_any(ROOT / "output" / "jepang_002.png")
    print(f"input {src.shape[1::-1]}  cleaned {cleaned.shape[1::-1]}  "
          f"final {final.shape[1::-1]}")

    worst = 0
    for name, (x1, y1, x2, y2) in SFX_BOX.items():
        a = src[y1:y2, x1:x2].astype(np.int16)
        b = cleaned[y1:y2, x1:x2].astype(np.int16)
        c = final[y1:y2, x1:x2].astype(np.int16)
        # Tinta SFX = piksel gelap di sumbernya. Yang dinilai bukan 'apakah kotak
        # ini berubah' melainkan 'apakah TINTA-nya hilang': erase mengubah gelap
        # jadi terang, jadi hitung piksel gelap yang menjadi terang.
        dark = a.mean(2) < 128
        gone_clean = int((dark & (b.mean(2) >= 128)).sum())
        gone_final = int((dark & (c.mean(2) >= 128)).sum())
        d_clean = int((np.abs(a - b).sum(2) > 90).sum())
        d_final = int((np.abs(a - c).sum(2) > 90).sum())
        worst = max(worst, gone_final)
        print(f"\n{name}  kotak {x2 - x1}x{y2 - y1}")
        print(f"  piksel tinta di sumber      : {int(dark.sum())}")
        print(f"  tinta hilang di 09_cleaned  : {gone_clean}")
        print(f"  tinta hilang di hasil akhir : {gone_final}")
        print(f"  piksel berubah (cleaned)    : {d_clean}")
        print(f"  piksel berubah (akhir)      : {d_final}")

    print("\n=== KESIMPULAN ===")
    if worst == 0:
        print("  SFX UTUH — tidak satu piksel tinta pun hilang.")
        print("  Jadi `sfx_idx: []` di halaman ini berarti 'tidak ada region SFX")
        print("  yang PERLU dijaga', bukan 'SFX terlanjur diterjemahkan'.")
        print("  Kriteria run_page.py yang menuntut sfx_idx tidak kosong salah")
        print("  sasaran: yang harus dijamin adalah SFX tidak berubah.")
    else:
        print(f"  CACAT: {worst} px tinta SFX hilang di hasil akhir.")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
