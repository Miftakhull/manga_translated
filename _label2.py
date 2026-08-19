"""Cacat #2: balon pendek tidak diterjemah — mana dari dua penyebabnya yang nyata.

Dua tersangka yang tersisa setelah gerbang ink-ratio gugur (13 region halaman
referensi semuanya 0.2332-0.8533 vs min_ink_ratio 0.015):

  T1  translate._label_region: cabang di DALAM balon `elif n == 3 and has_small:
      is_sfx = True` mengunci seruan 3-huruf ber-っ sebagai SFX. SFX berarti
      translation=None + PROTECTED, jadi translate_page melewatinya TANPA SUARA
      dan render mencetak balon kosong/Jepang. Yang diuji: mana saja seruan
      dialog nyata yang jatuh ke cabang itu.

  T2  kelengkapan JSON: `missing_repair_rounds = 1`. Muatan 15-19 balon jauh
      lebih mudah kehilangan kunci pendek daripada 4 balon. Diuji terpisah.

Di sini T1 yang diukur, karena bisa diukur tanpa jaringan sama sekali. Daftar
ujinya: seruan yang PASTI dialog (harus DIALOGUE), SFX sejati di dalam balon
(harus tetap SFX), dan simbol emosi yang tidak boleh mengubah keputusan.
"""
import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MANGATL_WORK", ROOT)
os.environ.setdefault("MANGATL_ROOT", os.path.join(ROOT, ".stage"))
sys.path.insert(0, os.path.join(ROOT, '.stage'))
import translate as tr
from config import Region

# (teks, label yang BENAR menurut mata pembaca)
KASUS = [
    # seruan/dialog pendek — semua ini diucapkan orang, bukan bunyi
    ("えっ",       "DIALOGUE"), ("えっ！？",   "DIALOGUE"),
    ("えっ！？♥",   "DIALOGUE"), ("ええっ",     "DIALOGUE"),
    ("ええっ！？",  "DIALOGUE"), ("はぁっ",     "DIALOGUE"),
    ("うわっ",     "DIALOGUE"), ("あっ！",     "DIALOGUE"),
    ("うんっ",     "DIALOGUE"), ("やだっ",     "DIALOGUE"),
    ("だめっ",     "DIALOGUE"), ("いやっ",     "DIALOGUE"),
    ("まてっ",     "DIALOGUE"), ("うそっ",     "DIALOGUE"),
    ("なにっ",     "DIALOGUE"), ("ちょっ",     "DIALOGUE"),
    ("あのっ",     "DIALOGUE"), ("そこっ",     "DIALOGUE"),
    ("はいっ",     "DIALOGUE"), ("ねえっ",     "DIALOGUE"),
    ("もうっ",     "DIALOGUE"), ("やめっ",     "DIALOGUE"),
    ("あーっ",     "DIALOGUE"), ("ふぇっ",     "DIALOGUE"),
    # SFX sejati di dalam balon — harus TETAP SFX
    ("ドキッ",     "SFX"),      ("どきっ",     "SFX"),
    ("ハッ",       "SFX"),      ("ゴクッ",     "SFX"),
    ("どきどき",   "SFX"),      ("ばたばた",   "SFX"),
    ("はぁはぁ",   "SFX"),      ("ごくん",     "SFX"),
]

print(" teks        benar     dapat     n core        catatan")
salah = []
for t, benar in KASUS:
    r = Region(idx=0, bbox=(0, 0, 60, 40), det_class="text_bubble",
               bubble_bbox=(0, 0, 60, 40))
    r.src_text = t
    r.translation = "X"
    tr._label_region(r)
    core = tr._sfx_core(t)
    n = len(core)
    tag = ""
    if r.label != benar:
        salah.append((t, benar, r.label, core, n))
        tag = "  <-- SALAH"
        if benar == "DIALOGUE":
            tag += (" (n==3 & っ)" if n == 3 and any(c in tr._SMALL for c in core)
                    else " (pola/dict)")
    print(f" {t:<11s} {benar:<9s} {r.label:<9s} {n} {core:<11s}{tag}")

print(f"\n SALAH: {len(salah)}/{len(KASUS)}")
for t, b, g, core, n in salah:
    print(f"   {t!r} core={core!r} n={n} -> {g}, seharusnya {b}")
