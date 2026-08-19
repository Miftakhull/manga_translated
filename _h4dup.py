"""Uji drop_nested_duplicates() terhadap bbox SUNGGUHAN, bukan sintetis.

Empat skenario, dan yang ketiga-keempat yang menentukan apakah perbaikannya
aman — bukan yang pertama:

  1. hasilnew4 (16 region, r0 bersarang 0.974 di r1)  -> harus buang TEPAT 1
  2. halaman bersih debug/*.json                       -> harus buang 0
  3. balon ganda sungguhan: satu kotak besar memuat DUA kotak kecil
                                                       -> harus buang 0
  4. r0/r1 hasilnew4 saja                              -> survivor bbox harus
                                                          jadi GABUNGAN keduanya

Tidak mengubah apa pun di luar .stage. Probe murni.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / ".stage"
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))

# Stage ulang: file _nbsrc baru saja diedit, .stage bisa basi.
STAGE.mkdir(exist_ok=True)
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
for p in sorted((ROOT / "_nbsrc").glob("*.py")):
    txt = p.read_text(encoding="utf-8")
    (STAGE / p.name).write_text(_MAGIC.sub("", txt, count=1), encoding="utf-8")
sys.path.insert(0, str(STAGE))

from config import Region                     # noqa: E402
from detect import _DUP_CONTAIN, drop_nested_duplicates  # noqa: E402

fail = 0


def mk(boxes: list[tuple]) -> list[Region]:
    return [Region(idx=i, bbox=b, det_class="text_bubble", det_conf=0.9)
            for i, b in enumerate(boxes)]


def check(name: str, ok: bool, detail: str = "") -> None:
    global fail
    print(f"  [{'OK ' if ok else 'GAGAL'}] {name}{('  ' + detail) if detail else ''}",
          flush=True)
    if not ok:
        fail += 1


print(f"_DUP_CONTAIN = {_DUP_CONTAIN}\n", flush=True)

# --- 1 + 4: halaman hasilnew4 sungguhan -------------------------------------
rep = json.loads((ROOT / "hasilnew4" / "hitomi_3740721_015.json")
                 .read_text(encoding="utf-8"))
h4 = [tuple(r["bbox"]) for r in rep["regions"]]
regs = mk(h4)
n = drop_nested_duplicates(regs)
print("1) hasilnew4, 16 region sungguhan", flush=True)
check("buang tepat 1", n == 1, f"dibuang={n} sisa={len(regs)}")
survivor = [r for r in regs if r.bbox[0] == 832]
check("r1 bertahan", len(survivor) == 1)
if survivor:
    got = survivor[0].bbox
    want = (832, 130, 1027, 405)   # gabungan (944,130,1024,321) & (832,135,1027,405)
    check("bbox survivor = gabungan", got == want, f"got={got} want={want}")
kept = {r.bbox for r in regs}
check("r0 (944,130,1024,321) hilang", (944, 130, 1024, 321) not in kept)
check("region lain utuh", len(regs) == 15)

# --- 2: halaman bersih ------------------------------------------------------
print("\n2) halaman bersih (harus 0 dibuang)", flush=True)
clean = []
for d in ("debug", "output", "hasilnew"):
    clean += sorted((ROOT / d).glob("*.json")) if (ROOT / d).is_dir() else []
if not clean:
    check("ada halaman bersih untuk diuji", False, "tidak ada *.json ditemukan")
for p in clean:
    try:
        r = json.loads(p.read_text(encoding="utf-8"))
        boxes = [tuple(x["bbox"]) for x in r.get("regions", [])]
    except Exception as e:                      # laporan rusak: lewati, catat
        print(f"  [skip ] {p.name}: {e}", flush=True)
        continue
    if len(boxes) < 2:
        continue
    rr = mk(boxes)
    k = drop_nested_duplicates(rr)
    check(f"{p.parent.name}/{p.name} ({len(boxes)} region)", k == 0,
          f"dibuang={k}")

# --- 3: balon ganda sungguhan: satu kotak besar memuat DUA kotak kecil ------
print("\n3) balon ganda (kotak besar memuat 2 kotak bersarang)", flush=True)
db = mk([
    (100, 100, 300, 400),   # kotak besar menutup kedua lobus
    (110, 110, 190, 390),   # lobus kiri, containment ~1.0
    (210, 110, 290, 390),   # lobus kanan, containment ~1.0
])
k = drop_nested_duplicates(db)
check("tidak ada yang dibuang", k == 0, f"dibuang={k}")
check("bbox besar tidak dilebarkan", db[0].bbox == (100, 100, 300, 400))

# lobus BERJAJAR tanpa kotak induk: containment rendah, jangan disentuh
print("\n   lobus berjajar tanpa induk", flush=True)
side = mk([(100, 100, 205, 400), (195, 100, 300, 400)])
k = drop_nested_duplicates(side)
check("tidak ada yang dibuang", k == 0, f"dibuang={k}")

# --- 5: kasus batas: tepat DI BAWAH ambang tidak boleh dibuang -------------
print("\n4) kasus batas ambang", flush=True)
# kotak kecil 100x100 = 10000 px; irisan 79x100 = 7900 -> 0.79 < 0.80
below = mk([(0, 0, 100, 100), (21, 0, 400, 100)])
k = drop_nested_duplicates(below)
check("containment 0.79 -> tidak dibuang", k == 0, f"dibuang={k}")
# irisan 81x100 = 8100 -> 0.81 >= 0.80
above = mk([(0, 0, 100, 100), (19, 0, 400, 100)])
k = drop_nested_duplicates(above)
check("containment 0.81 -> dibuang", k == 1, f"dibuang={k}")

print(f"\n{'SEMUA LOLOS' if fail == 0 else f'{fail} GAGAL'}", flush=True)
sys.exit(1 if fail else 0)
