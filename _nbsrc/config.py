%%writefile /content/mangatl/config.py

"""Konfigurasi terpusat: path, konstanta threshold, dan dataclass antar-stage."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

# ---------------------------------------------------------------- paths

ROOT = Path(os.environ.get("MANGATL_ROOT", "/content/mangatl"))
WORK = Path(os.environ.get("MANGATL_WORK", "/content/work"))
WEIGHTS = WORK / "weights"
FONTS = WORK / "fonts"
OUTPUT = WORK / "output"
DEBUG_DIR = WORK / "debug"

for _d in (WORK, WEIGHTS, FONTS, OUTPUT, DEBUG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- weights

# Tiap weight punya rantai mirror, dicoba berurutan — sama seperti FONT_CHAIN.
# Mirror pertama yang memberi file utuh dipakai. Ini bukan paranoia: URL CTD yang
# beredar di banyak tutorial menunjuk ke repo AnimeMangaInpainting, dan file itu
# TIDAK pernah ada di sana (404) — repo tersebut cuma berisi lama_large_512px.ckpt.
WEIGHT_URLS: dict[str, tuple[str, ...]] = {
    # RT-DETR-v2, 3 kelas: bubble / text_bubble / text_free
    "detector.onnx": (
        "https://huggingface.co/ogkalu/comic-text-and-bubble-detector"
        "/resolve/main/detector.onnx",
    ),
    # comic-text-detector: block head + soft text mask + line head.
    # Rilis GitHub upstream = sumber kanonik (94,669,756 byte, terverifikasi).
    "comictextdetector.pt.onnx": (
        "https://github.com/zyddnys/manga-image-translator/releases/download"
        "/beta-0.3/comictextdetector.pt.onnx",
        "https://huggingface.co/bropines/ballon-translator-models"
        "/resolve/main/models/comictextdetector.pt.onnx",
    ),
    # LaMa large 512px, di-finetune untuk anime/manga
    "lama_large_512px.ckpt": (
        "https://huggingface.co/dreMaz/AnimeMangaInpainting"
        "/resolve/main/lama_large_512px.ckpt",
    ),
}

# ---------------------------------------------------------------- fonts
#
# Anime Ace = face di gambar referensi (ALL CAPS, oblique, huruf I ber-crossbar).
# Lisensi Blambot melarang REDISTRIBUSI, jadi notebook hanya mengunduh saat
# runtime dan tidak pernah membundel .ttf-nya. Kalau URL mati, chain turun
# otomatis ke Comic Neue (OFL) tanpa crash.

FONT_CHAIN: list[dict[str, str]] = [
    {
        "name": "AnimeAce",
        "file": "anime_ace.ttf",
        "url": (
            "https://raw.githubusercontent.com/zyddnys/manga-image-translator"
            "/main/fonts/anime_ace.ttf"
        ),
        "role": "primary",
    },
    {
        "name": "AnimeAce2-Bold",
        "file": "animeace2_bld.ttf",
        "url": (
            "https://static.wfonts.com/download/data/2014/06/26"
            "/anime-ace-2-0-bb/animeace2_bld.ttf"
        ),
        "role": "primary",
    },
    {
        "name": "ComicNeue-BoldItalic",
        "file": "ComicNeue-BoldItalic.ttf",
        "url": (
            "https://raw.githubusercontent.com/google/fonts"
            "/main/ofl/comicneue/ComicNeue-BoldItalic.ttf"
        ),
        "role": "primary",
        "license_url": (
            "https://raw.githubusercontent.com/google/fonts"
            "/main/ofl/comicneue/OFL.txt"
        ),
    },
]

FONT_SHOUT = {
    "name": "Bangers",
    "file": "Bangers-Regular.ttf",
    "url": (
        "https://raw.githubusercontent.com/google/fonts"
        "/main/ofl/bangers/Bangers-Regular.ttf"
    ),
    "license_url": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/bangers/OFL.txt"
    ),
}

# Anime Ace cuma ~159 glyph -> tidak punya U+2661 (heart) maupun U+FF5E.
# Tanpa fallback ini, "AH~<3 NO, STOP~<3" keluar jadi kotak tofu.
FONT_SYMBOL = {
    "name": "NotoSansSymbols2",
    "file": "NotoSansSymbols2-Regular.ttf",
    "url": (
        "https://raw.githubusercontent.com/google/fonts"
        "/main/ofl/notosanssymbols2/NotoSansSymbols2-Regular.ttf"
    ),
    "license_url": (
        "https://raw.githubusercontent.com/google/fonts"
        "/main/ofl/notosanssymbols2/OFL.txt"
    ),
}

# Fallback lebar: Latin-ext (aksen, â é ñ ...), Cyrillic, Greek, Vietnam.
FONT_FALLBACK = {
    "name": "NotoSans",
    "file": "NotoSans.ttf",
    "url": (
        "https://raw.githubusercontent.com/google/fonts"
        "/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf"
    ),
    "license_url": (
        "https://raw.githubusercontent.com/google/fonts"
        "/main/ofl/notosans/OFL.txt"
    ),
}

# Fallback CJK: kana, kanji, Hangul — untuk target bahasa Asia.
FONT_CJK = {
    "name": "NotoSansCJKjp",
    "file": "NotoSansCJKjp-Regular.otf",
    "url": (
        "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF"
        "/Japanese/NotoSansCJKjp-Regular.otf"
    ),
    "license_url": (
        "https://raw.githubusercontent.com/googlefonts/noto-cjk"
        "/main/LICENSE"
    ),
}

# Fallback skrip RTL: Arab.
FONT_ARABIC = {
    "name": "NotoNaskhArabic",
    "file": "NotoNaskhArabic.ttf",
    "url": (
        "https://raw.githubusercontent.com/google/fonts"
        "/main/ofl/notonaskharabic/NotoNaskhArabic%5Bwght%5D.ttf"
    ),
    "license_url": (
        "https://raw.githubusercontent.com/google/fonts"
        "/main/ofl/notonaskharabic/OFL.txt"
    ),
}

# Fallback skrip: Thai.
FONT_THAI = {
    "name": "NotoSansThai",
    "file": "NotoSansThai.ttf",
    "url": (
        "https://raw.githubusercontent.com/google/fonts"
        "/main/ofl/notosansthai/NotoSansThai%5Bwdth%2Cwght%5D.ttf"
    ),
    "license_url": (
        "https://raw.githubusercontent.com/google/fonts"
        "/main/ofl/notosansthai/OFL.txt"
    ),
}

# ---------------------------------------------------------------- DeepL
#
# DeepL = mesin terjemahan MURNI: TIDAK menyensor konten apa pun (cocok
# untuk terjemahan uncensored), tapi juga TIDAK bisa klasifikasi label/SFX
# — itu ditangani heuristik di translate.py.
#
# Key berakhiran ':fx' = akun Free -> endpoint api-free (1 juta karakter/
# bulan). Key berbayar memakai https://api.deepl.com/v2.
DEEPL_API_BASE = "https://api-free.deepl.com/v2"

# Bahasa yang didukung DeepL (kode target_lang API). Thai/Vietnam/Filipino
# TIDAK didukung DeepL, jadi tidak muncul di daftar LANGUAGES versi ini.
DEEPL_TARGET: dict[str, str] = {
    "English": "EN",
    "Indonesian": "ID",
    "Spanish": "ES",
    "French": "FR",
    "German": "DE",
    "Portuguese": "PT",
    "Italian": "IT",
    "Dutch": "NL",
    "Russian": "RU",
    "Chinese (Simplified)": "ZH",
    "Chinese (Traditional)": "ZH-HANT",
    "Korean": "KO",
    "Japanese": "JA",
    "Arabic": "AR",
    "Turkish": "TR",
}

# ---------------------------------------------------------------- bahasa
#
# Pilihan bahasa sasaran di UI. "English" memakai gaya ALL CAPS + font Anime
# Ace; bahasa lain pakai huruf normal + fallback font multi-script.
LANGUAGES: list[str] = list(DEEPL_TARGET.keys())

# ---------------------------------------------------------------- router LLM
#
# Penyedia kedua: router OpenAI-compatible (gorouter). Bedanya dengan DeepL bukan
# soal mutu bahasa saja — LLM bisa DIBERI TAHU besar balonnya, dan itulah yang
# tidak mungkin dilakukan ke DeepL. DeepL menerjemahkan kalimat lepas konteks,
# jadi panjang hasilnya kebetulan; kalau kepanjangan, satu-satunya jalan keluar
# adalah mengecilkan font atau memenggal kata — dua cacat pertama di plan.txt.
#
# Keduanya tetap ada dan bisa dipilih di UI: DeepL lebih cepat dan gratis
# 1 juta karakter/bulan, router lebih patuh pada batas balon.
PROVIDERS: list[str] = ["DeepL", "LLM (freetokenfaucet)", "Router LLM (gorouter)"]
PROVIDER_DEFAULT = "LLM (freetokenfaucet)"

# WAJIB untuk gorouter. Tanpa header User-Agent, SETIAP permintaan ke host ini
# dibalas 403 "error code 1010" oleh Cloudflare — terukur 17 Agu 2026, dua
# bentuk auth (x-api-key dan Bearer) sama-sama 403. Bukan soal kredensial: key
# yang SAMA dengan UA ini membalas 200 dalam 8.2 s. Jebakan ini tidak terbaca
# dari pesan errornya, jadi jangan hapus tanpa mengukur lagi.
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Base URL router. Bisa ditimpa env ROUTER_API_BASE — host tunnel ini bisa
# berubah, dan hardcode-nya di kode berarti notebook harus diedit.
ROUTER_API_BASE = os.environ.get(
    "ROUTER_API_BASE", "https://rsx5kfk.abc-tunnel.us/v1"
).rstrip("/")

# SATU model saja, sesuai ANTHROPIC_MODEL di gorouter.txt dan itulah yang
# terukur bekerja (200, 8.2 s, JSON utuh, SFX ドキッ -> *thump*). Rantai cadangan
# sengaja berisi model yang sama, BUKAN kosong: router membalas 502 untuk model
# yang jelas-jelas ada lalu berhasil di panggilan berikutnya, jadi kegagalan
# pertama harus dianggap SEMENTARA dan yang menolong adalah mencoba lagi — bukan
# pindah model. Nama-nama model router lama dihapus; host ini tidak melayaninya.
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "gorouter/claude-opus-5")
ROUTER_FALLBACK: tuple[str, ...] = (ROUTER_MODEL,)

# Percobaan per model dan jeda dasar (detik, dilipatkan per percobaan).
#
# ANGKA-ANGKA INI PERNAH SALAH DAN AKIBATNYA PARAH, jadi alasannya ditulis.
# Semula 900/4/8: satu halaman bisa menunggu 900 s x 4 percobaan x 3 model =
# 3 JAM, dan dengan 2 ronde revisi jadi 9 jam. Yang terlihat di UI cuma
# "klasifikasi SFX + terjemah" menggantung — seolah GPU-nya lambat, padahal
# prosesnya menunggu HTTP yang tidak akan pernah datang.
#
# Halaman 13 balon yang SEHAT dijawab dalam 12.9-31.8 s (terukur, n=3). 120 s
# sudah 4x kasus terburuk yang sehat; lebih dari itu bukan "lambat" melainkan
# menggantung. ROUTER_DEADLINE membatasi SELURUH rangkaian percobaan, jadi
# batas atas satu panggilan bisa dihitung, bukan hasil perkalian diam-diam.
ROUTER_RETRY = 2
ROUTER_BACKOFF = 4
ROUTER_TIMEOUT = 120
ROUTER_DEADLINE = 240

# ------------------------------------------------------- LLM freetokenfaucet
#
# Penyedia ketiga, OpenAI-compatible sama seperti router, jadi dipegang kelas
# client yang sama — yang beda cuma base URL, nama model, dan satu parameter
# body. Alasan dia ada: dia GRATIS, dan anggaran balon butuh LLM (bukan DeepL),
# jadi tanpa penyedia ini fitur itu cuma jalan dengan kredit berbayar.
#
# MODELNYA HARUS MODEL GRATIS. Terukur 17 Agu 2026: dari 19 model terdaftar,
# 16 membalas HTTP 402 INSUFFICIENT_BALANCE ("model berbayar, saldo akun 0") —
# termasuk model yang dulu jadi default di sini, dan itulah yang membuat tiga
# halaman Colab keluar TANPA terjemahan. 402 ditolak SEBELUM satu token pun
# dibuat, jadi ini bukan soal kuota atau max_tokens.
#
# Yang gratis cuma tiga, semuanya diuji dengan _system_prompt() asli atas 19
# baris Jepang dan ketiganya membalas 19/19 kunci JSON dengan ♥ dan … utuh:
#   mimo-v2.5-pro   4.5-7.8 s, out 131-203  <- DIPAKAI. 3/3 sukses, SFX aman.
#   mimo-v2.5       4.6-6.9 s, out 144-205  - sekali MENERJEMAHKAN SFX ドキッ
#                                             jadi "My heart's racing" (salah).
#   gpt-5.6-terra   6.7-10.1 s, out 151-310 - balas ALL CAPS sendiri, sekali 524.
# Dua yang terakhir bisa dipakai lewat env FAUCET_MODEL tanpa mengedit sel.
FAUCET_API_BASE = os.environ.get(
    "FAUCET_API_BASE", "https://freetokenfaucet.com/v1"
).rstrip("/")
FAUCET_MODEL = os.environ.get("FAUCET_MODEL", "mimo-v2.5-pro")

# Model ini model REASONING, dan defaultnya membakar token untuk berpikir dulu.
# Terukur pada 3 balon: default 111-259 token reasoning lalu jawaban; dengan
# max_tokens=200 jatahnya habis di reasoning dan content keluar KOSONG STRING —
# gagal tanpa error HTTP, yang jauh lebih berbahaya daripada gagal berisik.
# thinking.type=disabled menghilangkan reasoning sepenuhnya: 26 token keluaran,
# 2.8 s, jawaban tetap benar. Token faucet TERBATAS, jadi ini bukan sekadar
# soal cepat — reasoning yang dibuang itu 4-10x biaya jawabannya sendiri.
# reasoning_effort="none" DITOLAK server (400, hanya low..max), dan
# reasoning.enabled=False DIABAIKAN (208 token reasoning tetap keluar) — jadi
# hanya bentuk inilah yang bekerja, jangan diganti tanpa mengukur lagi.
FAUCET_EXTRA: dict = {"thinking": {"type": "disabled"}}
# Plafon keluaran. Halaman 19 balon terukur puncak 203 token (mimo-v2.5-pro,
# 17 Agu 2026); 1200 memberi ruang untuk halaman padat tanpa memberi ruang untuk
# mengarang.
FAUCET_MAX_TOKENS = 1200
FAUCET_FALLBACK: tuple[str, ...] = (FAUCET_MODEL,)
# Jauh lebih pendek dari router: yang sehat 3-4 s, jadi 45 s sudah 10x kasus
# terburuk. Tidak ada gunanya menunggu 120 s ke endpoint yang biasanya 3 s.
FAUCET_TIMEOUT = 45
FAUCET_DEADLINE = 120

# ---------------------------------------------------------------- gaya
#
# Gaya terjemahan scanlation. Kunci = label di UI; nilai = instruksi tambahan
# yang disisipkan ke system prompt LLM.
TRANSLATION_STYLES: dict[str, str] = {
    "Manga Natural": (
        "Manga scanlation style: casual, natural dialogue that reads like a real "
        "released manga. Use contractions and short, punchy sentences. Match the "
        "speaker's personality and register (kids talk like kids, elders politely). "
        "Preserve emotion marks (... ! ? ~ <3)."
    ),
    "Formal": (
        "Formal, polite register: proper grammar, respectful tone, minimal "
        "contractions. Suitable for formal scenes, narration, or polite characters."
    ),
    "Casual & Slang": (
        "Casual and slangy: very informal, heavy contractions, modern slang and "
        "colloquialisms, playful or rough tone matching the scene."
    ),
    "Literal": (
        "Literal: stay as close to the Japanese wording as possible, even if the "
        "result sounds stiff. Keep structure and terms faithful to the source."
    ),
    "Uncensored": (
        "Uncensored: keep sexual, vulgar, crude, and dirty-joke language raw and "
        "faithful — no softening, no euphemism, no censorship, whatever the genre "
        "(comedy with dirty jokes, mature drama, hentai, ...). Use direct, explicit "
        "wording for sex acts and body parts exactly as the source intends. Explicit "
        "content is expected in this adult 18+ work and must be rendered literally."
    ),
    "Fully Localized": (
        "Fully localized: adapt idioms, jokes, and cultural references so the result "
        "reads as if the manga was originally written in the target language. "
        "Translate honorifics and Japanese terms into natural equivalents."
    ),
}

# Honorifik Jepang (san/kun/chan/sama/senpai) dipertahankan atau dilokalkan.
# Contoh: 'Yamada-kun' tetap 'Yamada-kun', bukan 'Tuan Yamada'.
KEEP_HONORIFICS_HELP = (
    "Pertahankan honorifik Jepang (san/kun/chan/senpai) — matikan untuk "
    "melokalkan jadi sapaan bahasa sasaran."
)

# ---------------------------------------------------------------- thresholds


@dataclass
class Settings:
    """Semua angka yang bisa disetel ada di sini, tidak tersebar di kode."""

    # deteksi
    det_size: int = 640
    det_conf: float = 0.30
    det_iou: float = 0.45
    # Balon ganda (連結吹き出し) = dua lobus menyatu yang kotaknya saling
    # tumpang tindih banyak. Pada 0.45 salah satu lobus disuppress, kedua
    # region teks jatuh ke satu kotak balon, dan dua terjemahan berakhir
    # bertumpuk. Ambang bubble sengaja lebih longgar; penjaga containment di
    # _nms() yang membedakan "dua lobus" dari "dua deteksi balon yang sama".
    det_iou_bubble: float = 0.65
    tiled_pass: bool = True          # 2x2 tile untuk menangkap teks kecil

    # mask
    seed_thresh: float = 0.50        # region yang pasti teks
    grow_thresh: float = 0.118       # melebar ke tepi anti-aliased
    dilate_ratio: float = 0.30       # kernel adaptif per tinggi glyph
    halo_deviation: int = 12         # |px - bg| minimal supaya dihitung halo
    min_cc_area: int = 50            # furigana terkecil yang masih diselamatkan

    # OCR
    min_ink_ratio: float = 0.015     # gate anti-halusinasi manga-ocr

    # erase
    flat_std_thresh: int = 10         # di bawah ini -> flat fill, tanpa GPU
    flat_std_thresh_noisy: int = 7
    # Isi PENUH interior balon saat menghapus, bukan cuma stroke glyph.
    #
    # Kenapa: mask stroke (ink_mask) dibangun dari ambang + dilasi adaptif, dan
    # ia sistematis melewatkan glyph yang tipis atau renggang. Terukur di
    # hasilnew/jp_6.JPG setelah hapusan: tanda dash panjang '——' di balon kanan
    # selamat utuh sebagai GARIS TIPIS, dan 'うう…' menyisakan dua coretan kecil.
    # Menaikkan dilasi tidak menyelesaikan ini — ia cuma menggeser cacatnya ke
    # garis balon yang ikut termakan. Interior balon tidak punya masalah itu:
    # batasnya bukan taksiran, itu garis balon sungguhan.
    #
    # Harganya jujur dan sudah disetujui: gradasi/screentone DI DALAM balon
    # hilang, diganti satu warna rata. Yang TIDAK ikut diputihkan: region tanpa
    # balon induk (bubble_bbox None -> art, bukan balon) dan region terlindungi
    # (SFX). Warna isian juga bukan putih buta — dipakai median latar balon itu
    # sendiri, jadi balon hitam tetap terisi hitam.
    bubble_fill: bool = True
    # Interior untuk ISIAN dikikis sekian x ketebalan garis balon. 1x, bukan 2x
    # seperti interior untuk tata letak: isian harus sampai mepet garis supaya
    # tidak ada pita tinta lama tertinggal di tepi, sementara tata letak justru
    # perlu jarak aman supaya glyph tidak menempel di garis.
    fill_erode_stroke: int = 1

    # typeset
    min_font_size: int = 11
    # Lebar halaman tempat min_font_size di atas dikalibrasi (CONTOH/2.webp
    # dipakai pada 1134 px). Lantai ukuran font TIDAK boleh angka mutlak: ia
    # dikalibrasi pada satu resolusi, dan halaman lain datang di resolusi lain.
    #
    # Terukur, bukan ditaksir. hasilnew/jp_6.JPG cuma 698 px lebar, dan pada
    # halaman itu lantai 11 px membuat anggaran balon jadi 2-39 karakter
    # sementara wording typeset referensi hasilnew/6.JPG untuk balon yang SAMA
    # 15-71 karakter (probe_floor6.py). Jadi model diperintah menulis jauh lebih
    # pendek daripada yang sebenarnya muat, dan hasilnya 'SO?' untuk balon yang
    # referensinya 'SO IN THE END, THE WHOLE CLASS GOT SO EXCITED...'.
    #
    # Huruf referensinya sendiri diukur 4-7 px tinggi (probe_refsize.py, modus 4,
    # median 5 pada 728 px) = ukuran font 5-8. Lantai 11 px memang di atas apa
    # yang dipakai typesetter manusia di resolusi ini.
    min_font_ref_width: int = 1134
    # Lantai mutlak: di bawah ini huruf tidak terbaca pada resolusi apa pun,
    # jadi penskalaan tidak boleh menembusnya.
    min_font_abs: int = 6
    max_font_size: int = 96
    # Diukur, bukan ditaksir. probe_lines.py mengambil profil baris tinta tiap
    # balon di CONTOH/2.webp: pitch baseline / cap_height = 1.36 (p25 1.33,
    # p75 1.37). Anime Ace punya asc+desc ~ 1.36x cap_height-nya sendiri, jadi
    # faktor yang menyamai referensi = 1.00 — bukan 1.28. Pada 1.28 tiap baris
    # menyisakan 4-12 px ruang kosong yang tidak dipakai ALL CAPS, dan ruang
    # itulah yang memaksa balon padat turun ke ukuran font minimum.
    line_spacing: float = 1.00
    # Margin dalam bubble. Dikurangi DUA KALI di layout.max_width_at (kiri dan
    # kanan), jadi 0.10 memakan 20% lebar tiap baris.
    #
    # Angka ini TIDAK dipilih dari margin nominalnya, tapi dari margin yang
    # BENAR-BENAR terukur. probe_tidy.py merender teks kita lalu mengukurnya
    # dengan kode yang sama seperti pengukuran referensi (cap_height dari
    # komponen terhubung, penyebut = kolom interior gabungan):
    #     pad   cap/min  sisi/min  isi      referensi: 0.117 / 0.165 / 70%
    #     0.04    0.116     0.096  80%
    #     0.06    0.115     0.100  80%
    #     0.10    0.113     0.144  69%
    # Naik dari 0.04 ke 0.06 nyaris tidak menambah margin nyata (0.096 -> 0.100)
    # tapi membuat DUA balon tersempit halaman (r6, r10) turun ke 10 px, di bawah
    # min_font_size, dan memperbesar galat ke ukuran referensi 3.05 -> 3.36 px
    # (probe_final.py, probe_cal.py). Jadi 0.04: nol region di bawah minimum,
    # galat terkecil, tanda hubung tetap satu.
    #
    # Sisa jarak ke sisi/min 0.165 milik referensi TIDAK dibayar dengan pad. Pada
    # pad 0.10 margin memang naik ke 0.144, tapi harganya ukuran font (galat 3.82)
    # dan tanda hubung tambahan — dua cacat yang eksplisit di plan.txt.
    # probe_wording.py memisahkan sebabnya: dengan wording referensi yang lebih
    # pendek, pad yang sama memberi isi 84% dan galat lebih kecil. Selisihnya
    # berasal dari panjang wording DeepL, dan itu tahap tersendiri.
    pad_ratio: float = 0.04
    force_upper: bool = True         # ALL CAPS hanya untuk target English
    target_lang: str = "English"     # di-set UI: Jepang -> bahasa ini
    translation_style: str = "Manga Natural"  # gaya scanlation (TRANSLATION_STYLES)
    keep_honorifics: bool = True     # san/kun/chan dipertahankan

    # terjemahan
    provider: str = "LLM (freetokenfaucet)"  # lihat PROVIDERS
    # Anggaran balon: kirim batas karakter per balon ke LLM, lalu ukur ulang
    # jawabannya dengan mesin tata letak sungguhan dan minta perbaikan untuk baris
    # yang melanggar. Hanya berlaku untuk provider LLM (faucet/router) — DeepL
    # tidak bisa diberi tahu apa pun. Biayanya ~0.8 detik CPU per balon (terukur: 13 balon =
    # 10 detik, 485 panggilan layout(), nol GPU) di atas tunggu jaringan ~20 detik.
    # Itu harga yang dibayar untuk 'NO KELUAR BUBBLE': tanpa anggaran, model yang
    # sama mengembalikan 'SORRY TO BARGE IN.' (18 karakter) untuk balon yang cuma
    # memuat 6 — bukan karena membangkang, tapi karena ia tidak melihat balonnya.
    balloon_budget: bool = True
    # Ronde perbaikan maksimum. Tiap ronde hanya mengirim ulang baris yang MASIH
    # melanggar, jadi ronde kedua jauh lebih murah dari yang pertama. Dibatasi 2
    # karena pengamatan: baris yang tidak membaik di ronde 2 juga tidak membaik di
    # ronde 5 — sisanya diserahkan ke fit() yang mengecilkan font.
    budget_repair_rounds: int = 2
    # Ronde permintaan ulang untuk balon yang kuncinya TIDAK dijawab model.
    # Beda urusan dengan budget_repair_rounds: yang itu menjaga jawaban tetap
    # muat, yang ini menjaga jawabannya ADA. Terukur di hasilnew/13.JPG: balon
    # 'えっ！？' terkirim tapi kuncinya hilang dari JSON balasan, jadi balon itu
    # tercetak berbahasa Jepang. Satu ronde sudah cukup untuk kasus itu (model
    # menjawab begitu diberi tahu balonnya nyata); angka 1 juga menjaga biaya
    # token faucet tetap kecil — yang dikirim ulang cuma idx yang kosong.
    missing_repair_rounds: int = 1
    oblique: float = 0.12            # shear sintetis; Anime Ace regular tegak,
                                     # referensi miring. 0 = matikan.
    # Rapatkan huruf secara horizontal (1.00 = matikan). BUKAN selera — ini
    # menutup selisih kerapatan font yang TERUKUR antara Anime Ace dan font
    # typeset referensi.
    #
    # probe_reffont2.py mengukur 10 baris CONTOH/6.JPG yang teksnya terbaca mata:
    # baris dicari dari komponen glyph (bukan kotak tangan), tinggi kapital =
    # median tinggi komponen tertinggi baris itu, lalu lebar baris dibanding
    # lebar string YANG SAMA di Anime Ace pada ukuran ber-cap-height sama:
    #     EMBARASSING       97 px vs 147 px   0.660
    #     DESCRIBED         77      113       0.681
    #     I'M PRAISING      88      130       0.677
    #     NEVER SEEN        95      136       0.699
    #     TO BE             42       58       0.724
    #     MOSTLY CAME BY   116      157       0.739
    #     BOTTOM OF         83      112       0.741
    #     TOO EXCITED AND  122      162       0.753
    #   median 0.690  -> referensi memuat ~1.45x karakter per baris pada tinggi
    #   huruf yang sama. Ukuran fontnya sendiri TIDAK salah: cap referensi 14.0 px
    #   pada 1357 px = 7.2 px pada halaman kita (698 px) = Anime Ace ukuran 6, dan
    #   kita memang merender 6-8. Yang beda cuma kerapatannya.
    #
    # Angkanya 0.85, bukan 0.690. probe_cond.py menyapu faktor pada mask jp_6
    # yang sungguhan dan menghitung tanda hubung yang tersisa:
    #     cond   wording kita        wording referensi
    #     1.00   3 hyphen  size 6-8  5 hyphen  1 luber
    #     0.88   2 hyphen  size 6-8  3 hyphen  0 luber
    #     0.85   2 hyphen  size 6-9  1 hyphen  0 luber
    #     0.72   0 hyphen  size 6-10 1 hyphen  0 luber
    # 0.85 memberi hampir seluruh perbaikannya (referensi 5->1 tanda hubung, luber
    # 1->0, plafon ukuran 9->10) sementara hurufnya masih berbentuk huruf. 0.690
    # dan 0.72 memang menghapus satu tanda hubung lagi, tapi pada cap 7 px itu
    # meremas batang huruf sampai di bawah satu piksel dan hasilnya kabur — mahal
    # untuk satu tanda hubung, dan tanda hubung sisanya (r3) sebabnya lain:
    # interiornya cuma menyisakan 26 px kolom bebas dari mask 46x87.
    condense: float = 0.85

    # inpaint
    lama_size: int = 512
    use_amp: bool = False            # torch.fft selalu fp32; fp16 nihil manfaat

    # verifikasi
    residue_deviation: int = 20
    # Komponen sisa TERBESAR yang masih dianggap wajar, dalam px — gerbang kedua
    # di verify.find_residue(), di samping gerbang jumlah. Gerbang jumlah
    # `max(30, 0.002*w*h)` berskala AREA balon sementara satu titik kotor tidak:
    # di balon 400x500 ambangnya jadi 400 px dan titik 60 px yang jelas terlihat
    # lolos tanpa satu ronde eskalasi. 12 dipilih karena komponen tinta yang
    # sah pun jauh lebih besar — komponen ink_mask terkecil di halaman referensi
    # yang bukan derau = 150 px — jadi gerbang ini tidak bisa salah menuduh satu
    # stroke utuh. Enam ambang (12/16/20/24/30/40) diuji berdampingan di
    # _residue5.py: semuanya menandai region yang sama, jadi 12 dipilih sebagai
    # yang paling ketat tanpa satu pun false positive. Yang menahan alarm palsu
    # bukan angka ini melainkan LINGKUP-nya — lihat verify._SCOPE_NEAR; dengan
    # lingkup yang benar halaman referensi bersih di semua enam ambang.
    residue_blob_max: int = 12
    max_escalation: int = 2

    # output
    output_format: str = "both"   # pilihan UI: "png" | "jpg" | "both"
    jpg_quality: int = 92
    debug: bool = False


SETTINGS = Settings()

# ---------------------------------------------------------------- ONNX Runtime

# Provider yang BENAR-BENAR dipakai tiap sesi. Sel warm-up membacanya untuk
# memutuskan berhenti atau lanjut.
ORT_REPORT: dict[str, str] = {}

# ---------------------------------------------------------------- catatan jalan
#
# Jalur terdegradasi pipeline ini SENGAJA tidak melempar: satu balon gagal
# diterjemah tidak boleh membunuh halaman, dan itu keputusan yang benar. Tapi
# konsekuensinya terukur pahit — satu run Colab keluar dengan `diterjemah 0` di
# tiga halaman (semua balon tetap berbahasa Jepang) dan user melihat NOL pesan,
# karena satu-satunya jejak tiap kegagalan adalah print() di dalam thread
# handler Gradio, yang di Colab nyangkut di sel yang sudah discroll.
#
# Jadi kegagalan dicatat, bukan cuma dicetak. Bentuknya daftar tuple dan bukan
# logging.Logger: pembacanya (app.py) perlu MEMUTUSKAN berdasarkan isinya —
# "ada error level apa saja di halaman ini" — dan itu jauh lebih murah atas
# daftar terstruktur daripada atas teks yang harus diurai ulang dengan regex.
# Pola yang sama dengan ORT_REPORT di atas: modul menulis, pembaca menyimpulkan.
RUN_NOTES: list[tuple[str, str, str]] = []

# Awalan yang dipakai note(). Dipilih supaya bisa dipindai mata DAN grep di log
# mentah: "[!!]" tidak pernah muncul di keluaran library pihak ketiga mana pun
# yang dipakai pipeline ini, jadi `grep '\[!!\]' run.log` = daftar error murni.
_NOTE_MARK: dict[str, str] = {"error": "[!!]", "warn": "[!]", "info": "[i]"}


def note(level: str, tag: str, msg: str) -> None:
    """Catat jalur terdegradasi SEKALIGUS cetak. level: "error"|"warn"|"info".

    Mencetak juga, bukan hanya menyimpan: sel notebook (warm-up, probe, sel 25)
    memanggil modul di luar UI, dan di sana stdout memang terlihat. Yang
    disimpan dipakai app.py untuk menyusun banner.

    RAHASIA TIDAK BOLEH MASUK `msg`. Pemanggilnya yang menjaga itu — sama
    seperti print() sebelumnya — karena di sinilah satu-satunya tempat pesan
    bisa ikut tertulis ke file .log yang lalu diunduh user.
    """
    RUN_NOTES.append((level, tag, msg))
    print(f"{_NOTE_MARK.get(level, '[i]')} [{tag}] {msg}")


def notes_since(mark: int) -> list[tuple[str, str, str]]:
    """Catatan yang muncul sejak `len(RUN_NOTES)` bernilai `mark`.

    Dipakai supaya catatan menempel ke HALAMAN yang bersangkutan, bukan ke
    seluruh batch: pemanggil mencatat panjangnya sebelum halaman dimulai lalu
    memanggil ini sesudahnya. Tanpa itu, halaman ke-3 mewarisi error halaman
    ke-1 dan tabel UI menuduh halaman yang benar.
    """
    return RUN_NOTES[max(mark, 0):]

# EXHAUSTIVE (default ORT) mengukur ulang tiap algoritma konvolusi untuk setiap
# bentuk input baru. Halaman manga ukurannya beda-beda, jadi biaya itu dibayar
# terus dan tidak pernah teramortisasi. HEURISTIC memilih langsung.
_CUDA_OPTS = {
    "device_id": 0,
    "cudnn_conv_algo_search": "HEURISTIC",
    "arena_extend_strategy": "kSameAsRequested",
    "do_copy_in_default_stream": True,
}


def ort_session(path, tag: str):
    """InferenceSession di CUDA. Kalau jatuh ke CPU, jatuhnya BERISIK.

    Fallback diam adalah sebab pipeline pernah makan 100 detik per halaman tanpa
    satu pun pesan: onnxruntime-gpu gagal memuat CUDA EP, ORT diam saja, semua
    inference pindah ke CPU. Sekarang tiap kegagalan dicatat di ORT_REPORT.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 2  # warning: ORT menyebut .so yang hilang

    # Sel 5 sudah menguji CUDA di subproses. Kalau di sana ia abort, mencobanya
    # lagi di sini membunuh kernel Colab tanpa traceback - jadi jangan dicoba.
    probe = os.environ.get("MANGATL_ORT_CUDA", "ok")
    if probe not in ("ok", ""):
        note("warn", "ort", f"{tag}: sel 5 menandai CUDA tidak aman ({probe}) -> CPU")
    elif "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            sess = ort.InferenceSession(
                str(path), opts,
                providers=[("CUDAExecutionProvider", _CUDA_OPTS), "CPUExecutionProvider"],
            )
            ORT_REPORT[tag] = sess.get_providers()[0]
            if "CUDAExecutionProvider" not in sess.get_providers():
                note("warn", "ort", f"{tag}: CUDA ditolak saat membuat sesi -> CPU (lambat)")
            return sess
        except Exception as exc:  # noqa: BLE001 - ORT melempar tipe khusus per build
            note("warn", "ort", f"{tag}: CUDA gagal ({str(exc)[:140]}) -> CPU (lambat)")
    else:
        note("warn", "ort", f"{tag}: CUDAExecutionProvider tidak tersedia -> CPU (lambat)")

    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    ORT_REPORT[tag] = "CPUExecutionProvider"
    return sess

DetClass = Literal["bubble", "text_bubble", "text_free"]
RegionLabel = Literal["DIALOGUE", "THOUGHT", "NARRATION", "SIGN", "SFX", "UNREADABLE"]
Route = Literal["flat", "lama", "skip"]

ID2LABEL: dict[int, str] = {0: "bubble", 1: "text_bubble", 2: "text_free"}

# Label yang TIDAK BOLEH dihapus dari halaman, apa pun yang terjadi.
PROTECTED_LABELS: frozenset[str] = frozenset({"SFX", "UNREADABLE"})


@dataclass
class Region:
    """Satu blok teks. Diteruskan lewat semua stage, diisi bertahap."""

    idx: int
    bbox: tuple[int, int, int, int]                    # xyxy, koordinat halaman
    det_class: DetClass = "text_bubble"
    det_conf: float = 0.0
    quad: np.ndarray | None = None                     # 4 titik, untuk rotasi
    bubble_bbox: tuple[int, int, int, int] | None = None
    # Kotak balon SEBELUM dibelah, diisi hanya kalau balon ini dipakai bersama
    # region lain (balon ganda). textmask.partition_shared_interiors() butuh
    # kotak asli untuk flood fill sekali di seluruh balon, bukan per belahan.
    shared_bubble_bbox: tuple[int, int, int, int] | None = None
    bubble_mask: np.ndarray | None = None              # interior bubble (lokal)
    # Mask ISIAN untuk erase: SELURUH interior balon, bukan cuma stroke glyph.
    # Dipisah dari bubble_mask karena dua-duanya punya tugas berbeda dan salah
    # satunya dirusak demi yang lain: bubble_mask dipangkas
    # disjoin_overlapping_interiors() supaya tata letak dua balon bertetangga
    # tidak beririsan, dan kalau mask yang sudah dipangkas itu dipakai mengisi,
    # sliver yang dipangkas tadi tidak pernah tersentuh dan tinta Jepang di
    # sana selamat. fill_mask direkam SEBELUM pemangkasan, jadi isian selalu
    # menutup interior penuh. fill_bbox ikut disimpan karena bubble_bbox juga
    # digeser oleh pemangkasan itu.
    fill_bbox: tuple[int, int, int, int] | None = None
    fill_mask: np.ndarray | None = None
    ink_mask: np.ndarray | None = None                 # stroke teks (lokal)
    est_font_size: float = 0.0
    angle: float = 0.0
    is_vertical: bool = False

    # OCR
    src_text: str = ""
    ink_ratio: float = 0.0

    # LLM
    label: RegionLabel = "DIALOGUE"
    label_conf: float = 0.0
    translation: str | None = None

    # erase
    route: Route = "flat"
    bg_color: tuple[int, int, int] | None = None

    # typeset
    final_font_size: int = 0
    lines: list[str] = field(default_factory=list)
    overflowed: bool = False

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def is_protected(self) -> bool:
        """SFX dan teks tak terbaca tidak pernah dihapus dari halaman."""
        return self.label in PROTECTED_LABELS

    def to_dict(self) -> dict:
        """Ringkasan untuk report.json — tanpa array numpy."""
        return {
            "idx": self.idx,
            "bbox": list(self.bbox),
            "det_class": self.det_class,
            "det_conf": round(self.det_conf, 3),
            "label": self.label,
            "label_conf": round(self.label_conf, 3),
            "src_text": self.src_text,
            "translation": self.translation,
            "route": self.route,
            "est_font_size": round(self.est_font_size, 1),
            "final_font_size": self.final_font_size,
            "lines": self.lines,
            "ink_ratio": round(self.ink_ratio, 4),
            "overflowed": self.overflowed,
            "protected": self.is_protected,
        }


SUPPORTED_EXT: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp", ".dib",
    ".tif", ".tiff", ".gif", ".ppm", ".pgm", ".pbm", ".avif", ".heic", ".heif",
})

