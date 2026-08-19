#!/usr/bin/env python3
"""Anggaran karakter per balon — jumlah karakter yang SUNGGUH muat, bukan taksiran.

Kenapa ada: penerjemah LLM bisa disuruh apa saja, tapi "buatlah pendek" itu
bukan perintah, itu selera. Yang bisa dipatuhi mesin adalah ANGKA. Probe ini
menghitung angkanya dari geometri balon lewat `typeset.layout()` yang sama
dengan yang dipakai render — jadi anggaran ini bukan model terpisah yang bisa
melenceng dari kenyataan, melainkan hasil pengukuran mesin tata letak itu
sendiri.

Dua angka per balon, dan keduanya perlu:
    soft = muat pada region_font_cap()  -> ukuran yang DIINGINKAN (proporsional
           ke besar balon, rasio 0.117 hasil ukuran CONTOH/2.webp)
    hard = muat pada SETTINGS.min_font_size -> batas mutlak; lewat dari ini
           fit() jatuh ke jalur darurat _MIN_FONT_FLOOR dan hasilnya tak terbaca

Cuma soft saja terlalu ketat: wording referensi sendiri melewatinya di balon
padat (r9 referensi 61 karakter), jadi menjadikan soft sebagai batas keras akan
menolak typeset profesional yang justru jadi target. Cuma hard saja terlalu
longgar: teks jadi muat tapi selalu di ukuran minimum. Jadi soft = target,
hard = batas.

Diukur dengan teks pengisi berfrekuensi huruf wajar, bukan 'AAAA' (A itu glyph
lebar; anggarannya akan terlalu pesimistis) dan bukan 'IIII' (terlalu optimistis).
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / ".stage"
CACHE = ROOT / ".probe_cache.pkl"
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

import numpy as np    # noqa: E402
import typeset        # noqa: E402
from config import SETTINGS  # noqa: E402

# Teks pengisi untuk mengukur anggaran. Dua hal yang penting dan keduanya
# ditemukan lewat kegagalan, bukan dipikirkan lebih dulu:
#
# 1. GRANULARITAS. layout() bekerja per kata, jadi anggaran hanya bisa melompat
#    sebesar kata berikutnya. Pengisi versi pertama dimulai "THE PREZ WAS
#    PUTTING TOGETHER ..." — lompatan 20 -> 29 karakter, dan tujuh balon
#    berbeda semuanya melaporkan soft=20 karena kebetulan mentok di kata
#    'TOGETHER' yang sama. Angka itu bukan sifat balonnya, itu sifat pengisinya.
#    Kata pengisi sekarang 2-6 huruf berputar, jadi lompatannya ~4-5 karakter.
#
# 2. LEBAR GLYPH. Anime Ace tidak monospace; 'W' hampir dua kali 'I'. Pengisi
#    harus mendekati frekuensi huruf Inggris, kalau tidak anggarannya bias.
#    Kalimat di bawah memakai huruf umum (E T A O N I S R H) pada proporsi
#    wajar dan menghindari deretan W/M yang membuat anggaran terlalu pesimistis.
_FILLER = (
    "SO THE PREZ HAS ALL THE NOTES AND I SEE THEM HERE ON HER DESK "
    "AT ONE SIDE OF THE ROOM IT IS SO NICE AND I DO LIKE IT A LOT "
    "LET ME TAKE A LOOK AT THIS ONE FOR JUST A BIT MORE OK THANKS "
) * 8
_WORDS = _FILLER.split()


def _fits(text: str, mask: np.ndarray, size: int, font_path: str) -> bool:
    """Muat utuh tanpa penggalan pada ukuran ini? Persis kriteria fit()."""
    return typeset.layout(text, mask, size, font_path, hyphenate=False)[0]


def char_budget(mask: np.ndarray, size: int, font_path: str) -> int:
    """Karakter terbanyak (batas kata) yang masih muat utuh pada `size`.

    Dicari lewat jumlah KATA, bukan potongan karakter sembarang: layout bekerja
    per kata, jadi memotong di tengah kata memberi jawaban yang tidak pernah
    bisa dicapai teks sungguhan.
    """
    if size <= 0:
        return 0
    lo, hi = 0, len(_WORDS)
    if _fits(" ".join(_WORDS), mask, size, font_path):
        return len(" ".join(_WORDS))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _fits(" ".join(_WORDS[:mid]), mask, size, font_path):
            lo = mid
        else:
            hi = mid
    return len(" ".join(_WORDS[:lo]))


def max_word_len(mask: np.ndarray, size: int, font_path: str) -> int:
    """Kata TERPANJANG (satu kata, tanpa spasi) yang masih muat satu baris.

    Angka kedua ini wajib ada, dan itu ketemu dari percontohan yang gagal:
    r6 anggaran totalnya 39 karakter, tapi 'MY APOLOGIES' yang cuma 12 karakter
    tetap menghasilkan tanda hubung di balon itu. Yang menjepit BUKAN panjang
    kalimat melainkan 'APOLOGIES' — satu kata 9 huruf tidak muat di lebar 68 px,
    dan begitu satu kata tidak muat, layout() hanya punya dua pilihan: penggal
    atau gagal. Referensi memilih kata lain sama sekali ('SORRY.'), dan itulah
    keputusan yang perlu disampaikan ke penerjemah.

    Diukur dengan huruf ber-lebar sedang berulang, lalu diverifikasi lewat
    layout() supaya angkanya tunduk pada mesin tata letak yang sama.
    """
    if size <= 0:
        return 0
    best = 0
    for n in range(2, 25):
        # 'ORDINANCES'-style: konsonan/vokal bergantian, lebar rata-rata wajar.
        probe = ("RONALDESTI" * 3)[:n]
        if typeset.layout(probe, mask, size, font_path, hyphenate=False)[0]:
            best = n
        else:
            break
    return best


def budgets(regions, font_path: str) -> dict[int, dict[str, int]]:
    """{idx: {...}} anggaran per region: total karakter + panjang kata maksimum."""
    out: dict[int, dict[str, int]] = {}
    for r in regions:
        mask = typeset._region_box_mask(r)[1]
        cap = typeset.region_font_cap(mask)
        out[r.idx] = {
            "min_side": int(min(mask.shape[:2])),
            "cap": cap,
            "soft": char_budget(mask, cap, font_path),
            "hard": char_budget(mask, SETTINGS.min_font_size, font_path),
            "word_soft": max_word_len(mask, cap, font_path),
            "word_hard": max_word_len(mask, SETTINGS.min_font_size, font_path),
        }
    return out


def main() -> int:
    import json

    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with CACHE.open("rb") as f:
        regions = pickle.load(f)
    b = budgets(regions, fp)

    # Wording yang sudah diketahui, untuk menguji anggarannya masuk akal.
    ref = json.loads((ROOT / "probe_ref_texts.json").read_text(encoding="utf-8")) \
        if (ROOT / "probe_ref_texts.json").exists() else {}
    got = json.loads((ROOT / "probe_font_texts.json").read_text(encoding="utf-8"))

    print(f"min_font_size={SETTINGS.min_font_size} pad_ratio={SETTINGS.pad_ratio} "
          f"line_spacing={SETTINGS.line_spacing}")
    print(f"{'idx':>3} {'sisi':>4} {'plafon':>6} {'soft':>4} {'hard':>4} "
          f"{'kata':>4} {'DeepL':>5} {'ref':>4}  panjang kata terpanjang")
    over_hard, over_soft, over_word = [], [], []
    for i in sorted(b):
        d = b[i]
        s_d, s_r = str(got.get(str(i), "")), str(ref.get(str(i), ""))
        ln_d, ln_r = len(s_d), len(s_r)
        # Kata terpanjang dihitung tanpa punctuation ekor: 'SORRY.' yang bikin
        # sesak itu huruf-hurufnya, dan titik jauh lebih sempit daripada huruf.
        def _lw(s: str) -> int:
            w = [re.sub(r"[^A-Z']", "", x) for x in s.upper().split()]
            return max((len(x) for x in w), default=0)
        lw_d, lw_r = _lw(s_d), _lw(s_r)
        flag = ""
        if ln_r and ln_r > d["hard"]:
            flag += " REF>hard"
            over_hard.append(i)
        elif ln_r and ln_r > d["soft"]:
            flag += " ref>soft"
            over_soft.append(i)
        if lw_d > d["word_soft"]:
            flag += f" DeepL kata {lw_d}>{d['word_soft']}"
            over_word.append(i)
        print(f"{i:>3} {d['min_side']:>4} {d['cap']:>6} {d['soft']:>4} "
              f"{d['hard']:>4} {d['word_soft']:>4} {ln_d:>5} {ln_r:>4}  "
              f"DeepL={lw_d} ref={lw_r}{flag}")
    (ROOT / "probe_budget.json").write_text(
        json.dumps({str(k): v for k, v in sorted(b.items())}, indent=1),
        encoding="utf-8")
    print(f"\nwording referensi melewati hard: {over_hard}")
    print(f"wording referensi melewati soft: {over_soft}")
    print(f"DeepL punya kata lebih panjang dari yang muat: {over_word}")
    print("-> probe_budget.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
