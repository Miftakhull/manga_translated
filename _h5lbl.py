"""Ukur kandidat perbaikan aturan K untuk r8 'ヒ．．．ッ！？'.

Cacatnya: katakana pendek ber-ッ DI DALAM balon bicara dikunci SFX oleh aturan
K di _sfx_in_bubble, jadi PROTECTED, jadi translate_page melewatinya dan balon
Jepangnya tercetak. Yang menyulitkan: selftest MEWAJIBKAN ("ハッ", True, "SFX")
— strukturnya identik dengan ヒッ (satu mora katakana + ッ, di dalam balon).
Jadi panjang batang TIDAK bisa membedakan keduanya; harus ada sinyal lain.

Probe ini menulis ulang HANYA aturan K, memakai helper _nbsrc yang sungguhan
untuk semua aturan lain (_sfx_core, _kata2hira, _sfx_stem, _all_katakana,
_sfx_pattern, _SFX_DICT, _KA_DIALOGUE), lalu menyapu 5 varian di atas 60 kasus
selftest yang ada + kasus baru. Varian A harus mereproduksi _label_region asli
PERSIS — kalau tidak, harnessnya sendiri yang salah dan angkanya tidak berarti.

Probe murni: tidak menulis apa pun ke _nbsrc/ maupun notebook.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / ".stage"
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))
STAGE.mkdir(exist_ok=True)
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
for p in sorted((ROOT / "_nbsrc").glob("*.py")):
    (STAGE / p.name).write_text(
        _MAGIC.sub("", p.read_text(encoding="utf-8"), count=1), encoding="utf-8")
sys.path.insert(0, str(STAGE))

import translate as tl                                          # noqa: E402
from config import Region                                       # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------- sinyal bicara

_ASK = frozenset("？?")
_BANG = frozenset("！!")
# Tanda JEDA. Sengaja tidak memuat ・ (U+30FB) dan ー (U+30FC): dua-duanya di
# blok kana, jadi _KANA cocok dan keduanya bukan jeda ucapan.
_PAUSE = frozenset("．.。、,，…‥")


def _asks(raw: str) -> bool:
    return any(c in _ASK for c in raw)


def _bangs(raw: str) -> bool:
    return any(c in _BANG for c in raw)


def _broken(raw: str) -> bool:
    """True kalau ada JEDA di ANTARA dua kana, bukan cuma di ujung.

    'ヒ．．．ッ' -> True  (jeda memutus mora: napas tertahan = suara)
    'フー．．．'  -> False (jeda di ujung: bunyi yang memanjang)
    """
    idx = [i for i, ch in enumerate(raw) if tl._KANA.match(ch)]
    if len(idx) < 2:
        return False
    return any(raw[i] in _PAUSE for i in range(idx[0] + 1, idx[-1]))


# ---------------------------------------------------------------- varian

VARIANTS = {
    "A": "baseline (kode sekarang)",
    "B": "K diblokir kalau teks asli ber-？",
    "C": "K diblokir kalau ber-？ ATAU jeda di ANTARA kana",
    "D": "K diblokir kalau ber-？ ATAU ber-！",
    "E": "K menuntut bukti kamus kalau batangnya <= 1 kana",
}


def _in_bubble_sfx(core: str, n: int, raw: str, v: str) -> bool:
    hira = tl._kata2hira(core)
    stem = tl._kata2hira(tl._sfx_stem(core))
    if hira in tl._KA_DIALOGUE or (stem and stem in tl._KA_DIALOGUE):
        return False
    if tl._sfx_pattern(core):
        return True
    if core in tl._SFX_DICT or hira in tl._SFX_DICT:
        return True
    if n <= 6 and tl._all_katakana(core) and core[-1] in tl._SMALL:
        blocked = False
        if v == "B":
            blocked = _asks(raw)
        elif v == "C":
            blocked = _asks(raw) or _broken(raw)
        elif v == "D":
            blocked = _asks(raw) or _bangs(raw)
        elif v == "E":
            blocked = len(tl._sfx_stem(core)) <= 1
        if not blocked:
            return True
    if stem and (stem + stem) in tl._SFX_DICT:
        return True
    return False


def label_of(text: str, in_bubble: bool, v: str) -> str:
    t = text.strip()
    if not t:
        return "DIALOGUE"
    if tl._has_kanji(t):
        return "DIALOGUE"
    core = tl._sfx_core(t)
    if not core:
        return "DIALOGUE"
    if core in tl._KA_DIALOGUE:
        return "DIALOGUE"
    n = len(core)
    has_small = any(c in tl._SMALL for c in core)
    has_long = tl._LONG in core
    if not in_bubble:
        if n <= 3:
            is_sfx = True
        elif has_small or has_long:
            is_sfx = n <= 8
        elif tl._sfx_pattern(core):
            is_sfx = True
        elif core in tl._SFX_DICT:
            is_sfx = True
        else:
            is_sfx = False
    else:
        is_sfx = _in_bubble_sfx(core, n, t, v)
    return "SFX" if is_sfx else "DIALOGUE"


def real_label(text: str, in_bubble: bool) -> str:
    r = Region(idx=99, bbox=(0, 0, 10, 10),
               bubble_bbox=(0, 0, 10, 10) if in_bubble else None)
    r.src_text = text
    tl._label_region(r)
    return r.label


# ---------------------------------------------------------------- kasus

LAMA = [
    ("フー．．．", False, "SFX"), ("ピクッ", False, "SFX"),
    ("ーー", False, "SFX"),
    ("ドキドキドキ", False, "SFX"), ("ぴくぴくっ", False, "SFX"),
    ("ガッタンゴットン", False, "SFX"), ("ドキッ", True, "SFX"),
    ("はぁっ", True, "SFX"),
    ("それは．．．", False, "DIALOGUE"), ("ちょっと", False, "DIALOGUE"),
    ("サッカー", True, "DIALOGUE"), ("こんにちは", True, "DIALOGUE"),
    ("うう．．．", True, "DIALOGUE"), ("んっ", True, "DIALOGUE"),
    ("ええっ", True, "DIALOGUE"), ("ええっ！？", True, "DIALOGUE"),
    ("うんっ", True, "DIALOGUE"), ("だめっ", True, "DIALOGUE"),
    ("いやっ", True, "DIALOGUE"), ("まてっ", True, "DIALOGUE"),
    ("うそっ", True, "DIALOGUE"), ("なにっ", True, "DIALOGUE"),
    ("ちょっ", True, "DIALOGUE"), ("そこっ", True, "DIALOGUE"),
    ("はいっ", True, "DIALOGUE"), ("ねえっ", True, "DIALOGUE"),
    ("もうっ", True, "DIALOGUE"), ("やめっ", True, "DIALOGUE"),
    ("うわっ", True, "DIALOGUE"), ("やだっ", True, "DIALOGUE"),
    ("あーっ", True, "DIALOGUE"), ("ふぇっ", True, "DIALOGUE"),
    ("ダメッ", True, "DIALOGUE"), ("ハイッ", True, "DIALOGUE"),
    ("ウンッ", True, "DIALOGUE"), ("ムリッ", True, "DIALOGUE"),
    ("オイッ", True, "DIALOGUE"), ("マテッ", True, "DIALOGUE"),
    ("ナニッ", True, "DIALOGUE"), ("ウソッ", True, "DIALOGUE"),
    ("イヤッ", True, "DIALOGUE"), ("ヤダッ", True, "DIALOGUE"),
    ("ヤメッ", True, "DIALOGUE"),
    ("ハッ", True, "SFX"), ("ズドンッ", True, "SFX"),
    ("ガシャッ", True, "SFX"), ("パチンッ", True, "SFX"),
    ("ゴクッ", True, "SFX"), ("どきっ", True, "SFX"),
    ("びくっ", True, "SFX"), ("ぴくっ", True, "SFX"),
    ("ごくっ", True, "SFX"), ("がくっ", True, "SFX"),
    ("ふわっ", True, "SFX"), ("どきどき", True, "SFX"),
    ("はぁはぁ", True, "SFX"), ("ぐちゅぐちゅ", True, "SFX"),
]

# Kasus BARU. Arah dialog = suara tokoh di dalam balon bicara (cacat r8).
# Arah SFX = yang tidak boleh ikut longgar, terutama teks yang SAMA di LUAR
# balon (bunyi latar) dan bunyi berkamus yang kebetulan ber-！ atau berjeda ujung.
BARU = [
    ("ヒ．．．ッ！？", True, "DIALOGUE"),      # <-- r8 hasilnew5, cacat sungguhan
    ("ヒッ！？", True, "DIALOGUE"),
    ("ヒ．．．ッ", True, "DIALOGUE"),
    ("ア．．．ッ", True, "DIALOGUE"),
    ("ウ．．．ッ！？", True, "DIALOGUE"),
    ("キャ．．．ッ", True, "DIALOGUE"),
    ("ヒ．．．ッ！？", False, "SFX"),           # di LUAR balon = bunyi latar
    ("ヒッ", True, "SFX"),                     # tanpa tanda bicara: tetap bunyi
    ("ドキッ！", True, "SFX"),
    ("ドキッ！？", True, "SFX"),                # diselamatkan bukti kamus (S)
    ("ゴクッ．．．", True, "SFX"),               # jeda di UJUNG, bukan di antara
    ("ズドンッ！", True, "SFX"),
    ("パチンッ！", True, "SFX"),
    ("ガシャッ．．．", True, "SFX"),
]

print("1) validitas harness: varian A harus == _label_region asli", flush=True)
bad = [f"{t!r} ib={ib} A={label_of(t, ib, 'A')} asli={real_label(t, ib)}"
       for t, ib, _ in LAMA + BARU
       if label_of(t, ib, "A") != real_label(t, ib)]
print(f"   beda = {len(bad)}", flush=True)
for b in bad:
    print("   ", b, flush=True)
if bad:
    sys.exit("harness tidak mereproduksi kode asli -> angka di bawah tak berarti")

print("\n2) sapuan varian\n", flush=True)
hdr = f"{'var':4} {'regresi 60 lama':>15} {'gagal 14 baru':>14}  keterangan"
print(hdr, flush=True)
print("-" * len(hdr), flush=True)
detail: dict[str, tuple[list[str], list[str]]] = {}
for v, ket in VARIANTS.items():
    reg = [f"{t!r}->{label_of(t, ib, v)} (ingin {w})"
           for t, ib, w in LAMA if label_of(t, ib, v) != w]
    new = [f"{t!r} ib={ib} ->{label_of(t, ib, v)} (ingin {w})"
           for t, ib, w in BARU if label_of(t, ib, v) != w]
    detail[v] = (reg, new)
    print(f"{v:4} {len(reg):>15} {len(new):>14}  {ket}", flush=True)

print("", flush=True)
for v in VARIANTS:
    reg, new = detail[v]
    if not reg and not new:
        continue
    print(f"  {v}: ", flush=True)
    for x in reg:
        print(f"     REGRESI  {x}", flush=True)
    for x in new:
        print(f"     BARU     {x}", flush=True)

lolos = [v for v in VARIANTS if not detail[v][0] and not detail[v][1]]
print(f"\n3) varian tanpa satu pun kesalahan: {lolos}", flush=True)
if lolos:
    print("   -> pilih yang TERLEBAR di antara yang aman (paling banyak "
          "menangkap suara di balon)", flush=True)
