"""Kandidat gerbang label cacat #2, diukur dua arah sebelum dipasang.

_label2.py mengukur cabang di DALAM balon `elif n == 3 and has_small: is_sfx =
True`: 20 dari 24 seruan dialog nyata jatuh ke SFX (ええっ はぁっ うわっ うんっ
やだっ だめっ いやっ まてっ うそっ なにっ ちょっ あのっ そこっ はいっ ねえっ
もうっ やめっ あーっ ふぇっ ええっ！？), dan SFX berarti translation=None +
PROTECTED, yaitu balon Jepang tercetak tanpa satu pun pesan error. Sekalian
terungkap arah sebaliknya: ハッ (katakana, SFX sejati) justru DIALOGUE.

Kandidatnya tiga bagian:
  N  normalisasi katakana->hiragana khusus untuk PENCARIAN KAMUS dialog, supaya
     エッ / ダメ / ハイ / ウン (dialog yang ditulis katakana) tidak lolos jadi SFX.
  K  di dalam balon, kana 2-3 huruf ber-っ yang SELURUHNYA katakana = SFX. Ini
     konvensi manga: bunyi ditulis katakana, ucapan hiragana. Yang ditulis
     katakana tapi memang dialog sudah diselamatkan (N) lebih dulu.
  S  di dalam balon, yang hiragana hanya SFX kalau BATANGNYA (core minus っ/ー
     di ujung) terbukti onomatope: batang-ganda ada di _SFX_DICT (どきっ ->
     どきどき, ごくっ -> ごくごく). Beban buktinya sengaja dibalik: salah menuduh
     dialog = balon Jepang tercetak, salah melepas SFX = SFX ikut diterjemah —
     dan yang kedua masih tertahan kontrak keras assert_sfx_intact + kamus.

Diukur dua arah sekaligus, karena memperbaiki satu arah dengan merusak arah
lain bukan perbaikan: DIALOGUE yang harus tetap dialog, dan SFX yang harus
tetap SFX.
"""
import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))
import translate as tr
from config import Region

DIALOG = [
    "えっ", "えっ！？", "えっ！？♥", "ええっ", "ええっ！？", "うわっ",
    "あっ！", "うんっ", "やだっ", "だめっ", "いやっ", "まてっ", "うそっ",
    "なにっ", "ちょっ", "あのっ", "そこっ", "はいっ", "ねえっ", "もうっ",
    "やめっ", "あーっ", "ふぇっ", "うう", "ええ", "はい", "うん", "だめ",
    # dialog yang ditulis KATAKANA — inilah yang butuh normalisasi
    "エッ", "エッ！？", "ダメッ", "ハイッ", "ウンッ", "アッ", "ムリッ",
    "オイッ", "ヤメッ", "マテッ", "ナニッ", "ウソッ", "イヤッ", "ヤダッ",
    "ゴメンッ", "ソウダッ", "ヤメテッ",
]
# はぁっ TIDAK ada di daftar dialog, dan itu keputusan sadar: ia embusan napas,
# bukan ucapan, dan _SFX_DICT sudah memuat はぁはぁ/ハァハァ. Membiarkannya SFX
# berarti tintanya utuh — sesuai kontrak 'SFX dan simbol tetap ada' — bukan
# cacat #2. Yang cacat adalah seruan yang JELAS ucapan (ええっ うんっ だめっ)
# ikut terkunci hanya karena panjangnya 3 dan ada っ.
SFX = [
    "ドキッ", "どきっ", "ハッ", "ゴクッ", "ドキドキ", "どきどき", "ばたばた",
    "はぁはぁ", "ごくん", "ぴくっ", "ごくっ", "がくっ", "ふわっ", "ぽんっ",
    "ズドンッ", "バキッ", "ガシャッ", "ぐちゅぐちゅ", "ドサッ", "パチンッ",
    "ズルッ", "ギュッ", "ドンッ", "ビクッ", "はぁっ", "ぞくっ", "びくっ",
]

# Kata dialog yang di manga sering ditulis KATAKANA dan belum ada padanan
# hiragana-nya di _KA_DIALOGUE. Ditambahkan dalam bentuk hiragana saja karena
# pencarian kamus kandidat menormalkan katakana -> hiragana lebih dulu, jadi
# satu entri menutup kedua tulisan.
TAMBAH_DIALOG = frozenset({
    "おい", "まて", "うそ", "ちょ", "ふぇ", "やだ", "そこ", "うわ", "むり",
    "ごめん", "そうだ", "やめて", "はいっ",
})

_SMALL, _LONG = tr._SMALL, tr._LONG


def kata2hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def all_kata(core: str) -> bool:
    ada = any("ァ" <= c <= "ヶ" for c in core)
    hira = any("ぁ" <= c <= "ゖ" for c in core)
    return ada and not hira


def stem(core: str) -> str:
    i = len(core)
    while i and (core[i - 1] in _SMALL or core[i - 1] == _LONG):
        i -= 1
    return core[:i]


def kandidat(t: str, in_bubble: bool = True) -> str:
    if tr._has_kanji(t):
        return "DIALOGUE"
    core = tr._sfx_core(t)
    if not core:
        return "DIALOGUE"
    hira = kata2hira(core)
    st = kata2hira(stem(core))
    kamus = frozenset(tr._KA_DIALOGUE) | TAMBAH_DIALOG
    # bagian N: kamus dialog dicari pada bentuk hiragana, DAN pada batangnya
    # (core minus っ/ー di ujung) — 'ダメッ' harus ketemu lewat 'だめ'.
    if core in kamus or hira in kamus or (st and st in kamus):
        return "DIALOGUE"
    n = len(core)
    has_small = any(c in _SMALL for c in core)
    has_long = _LONG in core
    if not in_bubble:
        if n <= 3:
            return "SFX"
        if has_small or has_long:
            return "SFX" if n <= 8 else "DIALOGUE"
        if tr._sfx_pattern(core) or core in tr._SFX_DICT or hira in tr._SFX_DICT:
            return "SFX"
        return "DIALOGUE"
    if tr._sfx_pattern(core):
        return "SFX"
    if core in tr._SFX_DICT or hira in tr._SFX_DICT:
        return "SFX"
    # bagian K: di dalam balon, kana ber-っ/ー yang SELURUHNYA katakana = bunyi.
    # Konvensi manga: bunyi katakana, ucapan hiragana. Yang ditulis katakana
    # tapi memang ucapan sudah lolos lewat kamus di atas.
    if n <= 6 and all_kata(core) and (has_small or has_long):
        return "SFX"
    # bagian S: yang hiragana hanya SFX kalau BATANGNYA terbukti onomatope,
    # yaitu bentuk gandanya ada di _SFX_DICT (どきっ -> どきどき, はぁっ ->
    # はぁはぁ). Beban buktinya sengaja dibalik dari yang lama: dulu cukup
    # 'n==3 dan ada っ', sekarang harus ada catatan onomatope-nya.
    if st and st not in kamus and (st + st) in tr._SFX_DICT:
        return "SFX"
    return "DIALOGUE"


def sekarang(t: str) -> str:
    r = Region(idx=0, bbox=(0, 0, 60, 40), det_class="text_bubble",
               bubble_bbox=(0, 0, 60, 40))
    r.src_text, r.translation = t, "X"
    tr._label_region(r)
    return r.label


for nama, kasus, benar in (("DIALOG", DIALOG, "DIALOGUE"), ("SFX", SFX, "SFX")):
    sn = sum(sekarang(t) != benar for t in kasus)
    kn = sum(kandidat(t) != benar for t in kasus)
    print(f"== {nama} ({len(kasus)} kasus) salah: sekarang={sn} kandidat={kn}",
          flush=True)
    for t in kasus:
        a, b = sekarang(t), kandidat(t)
        if a != benar or b != benar:
            print(f"   {t:<10s} sekarang={a:<9s} kandidat={b:<9s} "
                  f"core={tr._sfx_core(t)!r} batang={stem(tr._sfx_core(t))!r} "
                  f"ganda_di_dict="
                  f"{(kata2hira(stem(tr._sfx_core(t))) * 2) in tr._SFX_DICT}"
                  f"{'' if b == benar else '  <-- KANDIDAT MASIH SALAH'}", flush=True)

# jaga arah luar-balon tidak berubah tanpa sengaja
beda = [t for t in DIALOG + SFX
        if kandidat(t, False) != sekarang(t) and True]
print(f"\n luar balon: kandidat != sekarang pada {len(beda)} kasus (in_bubble=False "
      f"dibandingkan label in-bubble sekarang; hanya informatif)", flush=True)
