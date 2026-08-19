%%writefile /content/mangatl/typeset.py

"""Typeset: unduh font, binary-search ukuran, centroid-outward line growing.

Target visual = gambar referensi: ALL CAPS, oblique, center dua sumbu, hitam
murni tanpa stroke di dalam bubble, bubble sempit pecah satu kata per baris.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (FONT_ARABIC, FONT_CHAIN, FONT_CJK, FONT_FALLBACK, FONT_SHOUT,
                    FONT_SYMBOL, FONT_THAI, FONTS, SETTINGS, Region, note)

FONT_USED: str = ""
_SYMBOL_PATH: Path | None = None
_FALLBACK_PATH: Path | None = None
_CJK_PATH: Path | None = None
_ARABIC_PATH: Path | None = None
_THAI_PATH: Path | None = None
_CMAP_CACHE: dict[str, frozenset[int]] = {}

# Batas minimum pecahan kata saat dipenggal — lihat _split_word(). Aturan
# tipografi klasik: minimal 2 huruf ditinggal di baris ini, 3 huruf dibawa ke
# baris berikutnya. Menuntut 3 di kepala menolak awalan yang sah ('RE-QUESTED',
# 'DIS-POSING') dan itu menjepit ukuran font: 'REQUESTED.' jadi tidak bisa
# dipenggal sama sekali, dan balonnya tertahan di font 14.
_MIN_HEAD = 2
_MIN_TAIL = 3
# Penggalan kata baru dipakai kalau versi utuh membuat font turun di bawah
# fraksi ini dari versi ber-hyphen. Tanpa ambang, max() memilih versi ber-hyphen
# begitu ia lebih besar SATU poin pun, dan hasilnya penuh 'FI-NALLY', 'RE-ALLY',
# 'EXECU-TIVE' — sementara typeset referensi tidak punya satu pun tanda hubung.
# 0.75 = tukar tanda hubung hanya kalau ia membeli >= 33% ukuran font.
_HYPHEN_MIN_GAIN = 0.75
# Lantai keras ukuran font, dipakai HANYA di jalur darurat fit() ketika teks tidak
# muat di min_font_size sekalipun. Sebelumnya jalur itu memakai
# `min_font_size // 2` (= 5), yang membuat balon sempit dirender 6 px tanpa satu
# pun peringatan — di manga ukuran cetak itu tidak terbaca, dan 'berhasil tapi tak
# terbaca' lebih buruk daripada gagal yang kelihatan.
#
# Pada konfigurasi terkalibrasi (line_spacing 1.00, pad_ratio 0.04) halaman
# referensi TIDAK menyentuh lantai ini sama sekali — kedua balon tersempit (r6
# 68 px, r10 102 px) dirender 11 px (probe_final.py). Jadi ini murni jaring
# pengaman untuk halaman lain, bukan jalur yang dipakai rutin. Balon yang
# menabraknya menandakan wording-nya terlalu panjang untuk balonnya — itu urusan
# tahap wording, bukan ukuran font.
_MIN_FONT_FLOOR = 9
# Lebar halaman yang sedang dikerjakan, di-set sekali per halaman oleh
# pipeline.process_page() dan render_page(). Nol = belum di-set, dan min_font()
# lalu memakai lebar kalibrasi — jadi pemanggil lama (probe, selftest) berperilaku
# persis seperti sebelum lantai ini berskala resolusi.
_PAGE_W: int = 0
# Berapa px ukuran font boleh diturunkan demi blok yang seimbang atas-bawah —
# lihat _rebalance(). Ukuran terbesar yang muat tidak selalu bisa ditata rapi:
# di r12 halaman ini teks 'IS IT? LEMME SEE, C'MON~!' pada ukuran 15 mustahil
# lebih baik dari ketimpangan 43 px pada margin manapun (probe_r12_exhaust.py
# memindai SETIAP pemecahan baris dan SETIAP y yang legal), sementara ukuran 14
# turun ke 1 px. Jadi satu-dua px ukuran ditukar dengan blok yang benar-benar
# terpusat. Batasnya sengaja kecil: melonggarkannya berarti teks diam-diam
# mengecil, dan ukuran yang mengecil adalah cacat #3 di plan.txt.
_BAL_MAX_DROP = 3
# Berapa banyak ukuran font seorang PELEPAS boleh menyusut supaya tetangganya
# yang tercekik bisa membuang satu tanda hubung — lihat reclaim_unused_interiors().
# Fraksi, bukan angka px, dengan lantai 1 px: 3 px di ukuran 40 cuma 7% dan tidak
# terlihat, sedangkan 3 px di ukuran 9 adalah sepertiga tinggi huruf. Ambang px
# tetap karena itu salah di salah satu ujung — versi pertama memakai 1 px dan
# menolak pertukaran yang jelas menguntungkan di halaman uji balon-bertetangga
# (pengklaim 2 tanda hubung -> 1, pelepas 40 -> 37 pada balon 335x291 yang masih
# lapang), sementara di jp_6 pertukaran yang benar memang hanya 1 px (r2 9 -> 8).
# 0.10 melewatkan keduanya dan tetap menolak balon kecil digunduli.
_RECLAIM_LOSS = 0.10
_VOWELS = frozenset("AEIOUY")
_BREAK_CACHE: dict[str, set[int]] = {}
# Fraksi piksel band yang harus berada di dalam balon — lihat _row_free().
_ROW_COVER = 0.985
# Fraksi pita blok yang harus interior supaya satu baris dihitung "masih di
# dalam rongga" saat MENGUKUR batas atas/bawah — lihat block_slack().
# Sengaja jauh lebih longgar dari _ROW_COVER: yang di atas adalah izin "baris
# ini muat" (melanggarnya = huruf keluar balon), yang ini pengukuran letak
# batas. Menuntut 0.985 di sini mengembalikan cacat teks tidak terpusat.
# 0.20 dipilih supaya ekor balon (10-14 px) tidak ikut menggeser batas bawah,
# sementara 0.10 masih mengakuinya; lima ambang diukur di _fix12.py.
_SLACK_COVER = 0.20
# Simbol emosi yang TIDAK BOLEH diambil dari font utama, walau font utama
# mengaku punya codepoint-nya. Ini bukan kehati-hatian berlebih — ini terukur:
# anime_ace.ttf (259 glyph, font display Latin) MEMETAKAN U+2665 ke glyph
# bernama `yat`, yaitu huruf Cyrillic Ѣ, bukan hati. Jadi rantai fallback di
# _char_font() — yang hanya menyala kalau font utama TIDAK punya codepoint-nya —
# tidak pernah menyala, dan pembaca melihat huruf aneh di ujung balon. Itu
# persis cacat yang dilaporkan pada hasilnew/6.JPG dibanding jp_6.JPG.
#
# Terukur dari fonts/ di repo ini (fontTools getBestCmap):
#   anime_ace.ttf        U+2665 -> yat        (SALAH BENTUK), ♡ ♪ ☆ ★ 〜 tidak ada
#   NotoSansSymbols2     U+2665 -> heart, ♡ ❤ ☆ ★ ada; ♪ ♫ ♬ 〜 TIDAK ada
#   NotoSansCJKjp        ♥ ♡ ♪ ♫ ♬ ☆ ★ 〜 ～ ada semua
# Karena itu urutan rantai untuk karakter ini dibalik: simbol dulu (bentuk paling
# pas dan advance-nya tidak full-width), lalu CJK yang melengkapi not musik dan
# 〜, baru NotoSans. plan.txt mewajibkan simbol ini bertahan apa adanya, jadi
# menggambarnya dengan bentuk yang salah sama buruknya dengan menghapusnya.
_FORCE_SYMBOL = frozenset(ord(c) for c in "♥♡❤♪♫♬☆★〜～")
_PYPHEN: object | None = None
_PYPHEN_TRIED = False


def _pyphen():
    """Kamus pola penggalan. None kalau pyphen tidak terpasang — bukan error."""
    global _PYPHEN, _PYPHEN_TRIED
    if _PYPHEN_TRIED:
        return _PYPHEN
    _PYPHEN_TRIED = True
    try:
        import pyphen

        _PYPHEN = pyphen.Pyphen(lang="en_US")
    except (ImportError, OSError, KeyError):
        _PYPHEN = None
    return _PYPHEN


# ---------------------------------------------------------------- font setup


def _download(url: str, dest: Path, timeout: int = 60) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 1024:
            return False
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return False


def _usable(path: Path) -> bool:
    """Font baru dianggap valid kalau Pillow benar-benar bisa membukanya."""
    try:
        ImageFont.truetype(str(path), 20)
        return True
    except (OSError, ValueError):
        return False


def setup_fonts(verbose: bool = True) -> str:
    """Unduh FONT_CHAIN berurutan, pakai yang pertama berhasil.

    Lisensi Blambot melarang redistribusi, jadi .ttf hanya diunduh saat runtime
    dan tidak pernah dibundel. Kalau semua gagal, chain turun ke Comic Neue (OFL)
    tanpa crash.
    """
    global FONT_USED, _SYMBOL_PATH, _FALLBACK_PATH, _CJK_PATH
    global _ARABIC_PATH, _THAI_PATH
    FONTS.mkdir(parents=True, exist_ok=True)

    for entry in FONT_CHAIN:
        dest = FONTS / entry["file"]
        if _download(entry["url"], dest) and _usable(dest):
            FONT_USED = str(dest)
            if entry.get("license_url"):
                _download(entry["license_url"], dest.with_suffix(".OFL.txt"))
            if verbose:
                print(f"[font] pakai {entry['name']} -> {dest.name}")
            break
        if verbose:
            print(f"[font] {entry['name']} gagal, lanjut ke kandidat berikutnya")

    if not FONT_USED:
        FONT_USED = _system_fallback()
        # note() bukan print(): ini satu-satunya cabang di sini yang mengubah
        # HASIL, bukan cuma jalannya. Font sistem bukan Anime Ace, jadi seluruh
        # halaman keluar dengan huruf yang salah — dan itu harus terlihat di UI
        # walau verbose=False (app.py memanggil setup_fonts(verbose=False)).
        note("warn", "font",
             f"semua kandidat gagal diunduh, pakai font sistem: {FONT_USED} — "
             "hurufnya BUKAN Anime Ace")

    sym = FONTS / FONT_SYMBOL["file"]
    if _download(FONT_SYMBOL["url"], sym) and _usable(sym):
        _SYMBOL_PATH = sym
        _download(FONT_SYMBOL["license_url"], sym.with_suffix(".OFL.txt"))

    shout = FONTS / FONT_SHOUT["file"]
    if _download(FONT_SHOUT["url"], shout):
        _download(FONT_SHOUT["license_url"], shout.with_suffix(".OFL.txt"))

    # Fallback multi-script: Latin-ext/Cyrillic/Greek, CJK, Arab, Thai.
    fb = FONTS / FONT_FALLBACK["file"]
    if _download(FONT_FALLBACK["url"], fb) and _usable(fb):
        _FALLBACK_PATH = fb
        _download(FONT_FALLBACK["license_url"], fb.with_suffix(".OFL.txt"))
    for key, attr in ((FONT_CJK, "_CJK_PATH"), (FONT_ARABIC, "_ARABIC_PATH"),
                      (FONT_THAI, "_THAI_PATH")):
        p = FONTS / key["file"]
        if _download(key["url"], p) and _usable(p):
            globals()[attr] = p
            _download(key["license_url"], p.with_suffix(".OFL.txt"))

    return FONT_USED


def _system_fallback() -> str:
    """Font terakhir yang pasti ada di hampir semua sistem."""
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ):
        if Path(cand).exists():
            return cand
    return str(Path(ImageFont.__file__).parent / "fonts")


def set_user_font(path: str | Path) -> str:
    """Slot upload font sendiri — jalur paling aman secara lisensi."""
    global FONT_USED
    p = Path(path)
    if p.exists() and _usable(p):
        FONT_USED = str(p)
        _font.cache_clear()
    return FONT_USED


@lru_cache(maxsize=64)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _cmap(path: str) -> frozenset[int]:
    """Set codepoint yang dipunya font. Anime Ace cuma ~159 glyph."""
    if path in _CMAP_CACHE:
        return _CMAP_CACHE[path]
    try:
        from fontTools.ttLib import TTFont

        cm = frozenset(TTFont(path, fontNumber=0, lazy=True).getBestCmap().keys())
    except Exception:  # noqa: BLE001 - fontTools opsional
        cm = frozenset()
    _CMAP_CACHE[path] = cm
    return cm


# ---------------------------------------------------------------- layout


def _cond() -> float:
    """Faktor rapat horizontal yang berlaku sekarang; 1.0 = mati.

    Satu pintu untuk SELURUH file. Kerapatan tidak boleh dipasang di dua tempat
    terpisah: kalau JALUR UKUR dan JALUR GAMBAR memakai angka berbeda, baris
    dinilai muat lalu tergambar lebih lebar dan menembus garis balon — persis
    cacat 'keluar bubble' yang dilarang plan.txt. Di sini hanya ada satu
    pemakainya (_line_width, yang dipakai _measure DAN render_region), jadi
    keduanya tidak mungkin pecah.

    Alasan angkanya ada di SETTINGS.condense (config.py): selisih kerapatan
    terukur 0.690 antara Anime Ace dan font typeset CONTOH/6.JPG, dan sapuan
    probe_cond.py yang memilih 0.85.

    Nilai di luar [0.3, 1.0] diabaikan (jatuh ke 1.0), bukan di-clamp: angka
    seperti itu selalu salah tulis, dan meremas huruf 5x lebih baik gagal
    kelihatan daripada diam-diam merender teks yang tidak terbaca.
    """
    c = float(getattr(SETTINGS, "condense", 1.0) or 1.0)
    return c if 0.3 <= c <= 1.0 else 1.0


def _measure(text: str, font: ImageFont.FreeTypeFont) -> float:
    """Lebar advance baris, sudah memperhitungkan glyph fallback.

    getlength() = advance width (float). Jangan campur dengan getbbox().

    Sudah termasuk faktor rapat horizontal (_cond) karena _line_width memakainya
    — jadi seluruh mesin tata letak mengukur lebar yang BENAR-BENAR tergambar.

    Dulu fungsi ini memanggil font.getlength() mentah, jadi JALUR UKUR (fit,
    penataan baris) dan JALUR GAMBAR (_draw_line, yang jatuh ke font lain per
    karakter) bisa memakai font berbeda untuk karakter yang sama — dan lebarnya
    memang berbeda. Terukur pada size 20: ☆ 14.0 px di anime_ace tapi 21.0 px di
    NotoSansSymbols2, ♥ 18.0 vs 15.0. Ukur-kurang seperti ☆ itu yang berbahaya:
    fit() menyangka baris muat, lalu penggambaran melebarkannya 7 px keluar
    balon — cacat "keluar bubble" yang justru dilarang plan.txt.
    """
    cmap = _cmap(getattr(font, "path", "") or "")
    return _line_width(text, font, cmap, int(getattr(font, "size", 0) or 0))


def _line_height(font: ImageFont.FreeTypeFont) -> int:
    asc, desc = font.getmetrics()
    return int((asc + desc) * SETTINGS.line_spacing)


@lru_cache(maxsize=128)
def _ink_band(path: str, size: int) -> tuple[int, int]:
    """Offset atas & bawah tinta di dalam kotak baris, relatif ke anchor 'la'.

    Kotak baris `(asc + desc) * line_spacing` jauh lebih tinggi daripada tinta
    yang benar-benar tergambar: teks ALL CAPS tidak memakai ruang descender sama
    sekali. Kalau probe balon memakai kotak penuh, baris dianggap menembus garis
    balon padahal yang menembus cuma ruang kosong — dan fit() lalu mengecilkan
    font tanpa alasan (balon 'DID THE CONTRACTOR BAIL?' turun dari 21 ke 16 dan
    kata terpanjangnya pecah tiga kali). Yang harus muat adalah tintanya.
    """
    f = _font(path, size)
    _, y0, _, y1 = f.getbbox("AHJQ,;()")
    return int(y0), int(y1)


def _row_free(mask: np.ndarray, y0: int, y1: int, x0: float, x1: float) -> bool:
    """Probe: apakah rentang ini masih di dalam interior bubble?"""
    h, w = mask.shape[:2]
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    x0i, x1i = max(0, int(x0)), min(w, int(x1))
    if y1 <= y0 or x1i <= x0i:
        return False
    # Diuji lewat CAKUPAN, bukan rata-rata. Rata-rata tidak bisa membedakan
    # "seluruhnya di dalam balon" dari "sebagian keluar tapi sisanya putih
    # pekat": band setinggi satu baris yang 14% ujungnya sudah di luar oval
    # tetap bernilai 239 dan lolos ambang 220 — itulah yang membuat baris
    # pertama dan terakhir menembus garis balon walau tiap baris lolos probe.
    band = mask[y0:y1, x0i:x1i]
    return float((band >= 200).mean()) >= _ROW_COVER


def _centroid(mask: np.ndarray) -> tuple[int, int]:
    m = cv2.moments((mask > 0).astype(np.uint8))
    if m["m00"] == 0:
        h, w = mask.shape[:2]
        return w // 2, h // 2
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def free_run(flags: np.ndarray, a: int, b: int) -> tuple[int, int] | None:
    """Rentang baris bebas yang MENYAMBUNG dan paling menaungi [a, b].

    Harus menyambung, bukan sekadar "baris bebas paling atas dan paling bawah".
    Interior balon yang sudah dipartisi dari tetangganya kerap terpecah: pita
    bebas di atas dinding partisi tidak bisa dicapai teks di bawahnya. Memakai
    flatnonzero(...)[0] apa adanya melaporkan ruang yang sebenarnya terhalang —
    r6 halaman referensi dilaporkan punya 94 px ruang di atas padahal pitanya
    ada di seberang dinding, dan penyeimbang mana pun lalu mengejar angka semu.
    """
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return None
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], cuts + 1))
    ends = np.concatenate((cuts, [idx.size - 1]))
    best: tuple[int, tuple[int, int]] | None = None
    for s, e in zip(starts, ends):
        lo, hi = int(idx[s]), int(idx[e])
        ov = min(hi, b) - max(lo, a)  # >0 tumpang tindih tinta, <0 jaraknya
        if best is None or ov > best[0]:
            best = (ov, (lo, hi))
    return None if best is None else best[1]


def _free_flags(mask: np.ndarray, cx: int, width: float) -> np.ndarray:
    """Baris mana yang bebas seluruhnya pada pita selebar `width` di sekitar cx."""
    x1 = int(max(cx - width / 2, 0))
    x2 = int(min(cx + width / 2, mask.shape[1]))
    col = (mask[:, x1:x2] > 0) if x2 > x1 else np.zeros((mask.shape[0], 1), bool)
    flags = col.all(1)
    return flags if flags.sum() >= 2 else col.any(1)


def _cover_flags(mask: np.ndarray, cx: int, width: float, cover: float) -> np.ndarray:
    """Baris mana yang masih DI DALAM rongga pada pita selebar `width` di cx.

    Bedanya dengan _free_flags: di sini satu baris cukup bercakupan `cover`,
    tidak harus bebas seluruh pita. Dipakai block_slack() untuk mengukur LETAK
    batas atas/bawah rongga — bukan untuk memutuskan satu baris muat (itu
    _row_free, dan ambangnya memang ketat). Lihat block_slack() untuk angka
    perbandingan lima ambang.
    """
    x1 = int(max(cx - width / 2, 0))
    x2 = int(min(cx + width / 2, mask.shape[1]))
    if x2 <= x1:
        return np.zeros(mask.shape[0], bool)
    return (mask[:, x1:x2] > 0).mean(1) >= cover


def _band_run(mask: np.ndarray, y0: int, y1: int) -> tuple[int, int] | None:
    """Rentang kolom TERLEBAR yang bebas di SELURUH band y0..y1, atau None.

    Bedanya dengan ekspansi simetris di max_width_at(): rongga tidak dipaksa
    simetris terhadap satu x. Pada interior yang sudah dipotong tetangganya
    rongga memang tidak simetris, dan memaksanya simetris membuang separuh ruang
    yang ada. r6 halaman ini di ketinggian tengah bebas di x=14..67 (54 px),
    tapi ekspansi simetris di centroid x=36 cuma mengakui 2*(36-14)=44 px;
    dikurangi pad*2 jadi 40 px, sementara 'SORRY.' butuh 49 px. Akibatnya
    SETIAP y terpusat ditolak dan satu-satunya y yang lolos ada di 119..140 —
    39 px di bawah tengah, dan itu yang terlihat sebagai teks melorot
    (probe_fitwin.py mencetak ketiga aturan lebar berdampingan).

    Ambangnya sama dengan _row_free (>=200), hanya lebih ketat: per kolom
    dituntut bebas SELURUH tinggi band, bukan 98.5% dari kotak. Jadi lebar yang
    dilaporkan di sini tidak pernah melebihi yang diizinkan _row_free.
    """
    mh = mask.shape[0]
    a, b = max(int(y0), 0), min(int(y1), mh)
    if b <= a:
        return None
    idx = np.flatnonzero((mask[a:b] >= 200).all(0))
    if idx.size == 0:
        return None
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], cuts + 1))
    ends = np.concatenate((cuts, [idx.size - 1]))
    s, e = max(zip(starts, ends), key=lambda p: idx[p[1]] - idx[p[0]])
    return int(idx[s]), int(idx[e])


def block_axis(mask: np.ndarray, lines: list[str], top: int, lh: int,
               ink_top: int, ink_bot: int, font: ImageFont.FreeTypeFont,
               fallback: int) -> int:
    """Satu sumbu x untuk SELURUH blok: pusat rongga yang sah untuk semua baris.

    Satu sumbu, bukan satu per baris. Memusatkan tiap baris di rongganya sendiri
    memang menurunkan ketimpangan, tapi blok jadi bergerigi sampai 15 px di r7
    halaman ini — rapi menurut angka, kacau menurut mata (probe_axis.py varian
    V1 vs V2). Yang dipakai: IRISAN rongga semua baris, supaya sumbunya sah
    bahkan untuk baris terlebar, lalu titik tengah irisan itu.

    Kalau irisannya lebih sempit dari baris terlebar, sumbu blok tidak ada dan
    fungsi ini jatuh ke `fallback` (centroid) — _verify() yang lalu menolak
    kandidat itu, sama seperti sebelumnya.
    """
    if not lines:
        return fallback
    lo, hi = 0, mask.shape[1] - 1
    for k in range(len(lines)):
        run = _band_run(mask, top + k * lh + ink_top, top + k * lh + ink_bot)
        if run is None:
            return fallback
        lo, hi = max(lo, run[0]), min(hi, run[1])
    if hi - lo + 1 < max(_measure(ln, font) for ln in lines):
        return fallback
    return (lo + hi) // 2


def line_axis(mask: np.ndarray, lines: list[str], start_y: int, size: int,
              font_path: str) -> int:
    """Sumbu x yang dipakai layout() untuk blok ini — dihitung ulang, bukan disimpan.

    render_region() harus menggambar di sumbu yang SAMA dengan yang dipakai
    layout() saat memutuskan blok ini muat. block_axis() murni fungsi dari
    (mask, lines, start_y, ukuran), jadi menghitungnya ulang di sini selalu
    memberi angka yang sama tanpa menambah nilai balik layout() — yang dipakai
    belasan probe.
    """
    font = _font(font_path, size)
    ink_top, ink_bot = _ink_band(font_path, size)
    return block_axis(mask, lines, start_y, _line_height(font),
                      ink_top, ink_bot, font, _centroid(mask)[0])


def block_slack(mask: np.ndarray, cx: int, pad: int, w_first: float, w_last: float,
                ink_a: int, ink_b: int) -> tuple[int, int]:
    """Sisa ruang di ATAS tinta baris pertama dan di BAWAH tinta baris terakhir.

    SATU pita acuan selebar baris TERLEBAR, dan satu run untuk kedua ujung.

    Versi sebelumnya memakai dua lebar berbeda — pita selebar baris PERTAMA
    untuk ujung atas, selebar baris TERAKHIR untuk ujung bawah. Itu terdengar
    benar tapi tidak bisa dipakai untuk menilai keseimbangan: di balon oval
    kedua pita menyempit pada laju yang berbeda, jadi dua angka yang
    dibandingkan `abs(up - dn)` diukur terhadap dua batas yang berbeda. Blok
    yang mata lihat melenceng 14-72 px tetap melaporkan bal=0, lalu MENANG di
    pemindaian n lewat `if bal <= tol: break`, mendapat nol iterasi _polish
    (`abs(dn-up)//2 == 0`) dan juga melewati _rebalance (yang menilai dari
    fungsi ini juga). Itu cacat "teks tidak di tengah antara atas dan bawah".
    Terukur di _fix12.py: 13 region halaman referensi mean 25.4 max 72 px
    ketimpangan nyata, sintetis mean 18.8 max 46.

    Barisnya dihitung "masih interior" lewat CAKUPAN, bukan semua-atau-tidak.
    _free_flags menuntut SELURUH pita bebas dalam satu baris (`col.all(1)`);
    dipakai pada pita selebar baris terlebar itu memotong baris yang sebenarnya
    masih di dalam oval, dan hasilnya memburuk di halaman asli (r5 23->43,
    r7 8->48, r12 59->77 di _center5.py). Cakupan >= _SLACK_COVER melihat apa
    yang dilihat mata: batas atas dan bawah rongga di kolom tempat blok berada.

    Ambangnya BUKAN _ROW_COVER (0.985). Itu ambang "baris ini muat", dituntut
    ketat karena melanggarnya berarti huruf keluar balon; ini pengukuran
    LETAK batas, dan menuntut 98.5% di sini mengembalikan cacatnya. 0.20 dipilih
    karena ekor balon (10-14 px) tidak pernah mencapai 20% lebar blok, jadi ekor
    tidak ikut menggeser batas bawah — sementara 0.10 masih mengakuinya.
    Terukur berdampingan (0.10/0.20/0.30/0.40/0.50) di _fix12.py: halaman
    mean 2.3/2.5/3.1/3.5/4.8, sintetis mean 1.4/1.4/1.7/2.4/3.2, over=0 semua.

    Dinding partisi (interior terbelah tetangga) tetap dihormati: barisnya
    bercakupan 0, jadi `free_run` — yang mengambil run TERSAMBUNG — berhenti di
    dinding dan tidak pernah menaungi lobus tetangga. Inilah yang menjaga
    syarat "jangan sampai ada huruf yang termakan mask lain kalau double bubble
    merge": sisa ruang diukur di dalam lobus milik region ini saja.

    Kalau tidak ada run sama sekali (mis. pita acuan lebih lebar dari rongga
    mana pun), fungsi ini JATUH ke perhitungan dua-lebar yang lama, bukan ke
    `return 0, 0` — 0,0 berarti "seimbang" dan itu justru menutupi kegagalan.
    """
    mh = mask.shape[0]
    w_ref = max(float(w_first), float(w_last), 1.0)
    run = free_run(_cover_flags(mask, cx, w_ref, _SLACK_COVER), ink_a, ink_b)
    if run is None:
        top = free_run(_free_flags(mask, cx, w_first), ink_a, ink_a)
        bot = free_run(_free_flags(mask, cx, w_last), ink_b, ink_b)
        if top is None or bot is None:
            return 0, 0
        return ink_a - max(top[0], pad), min(bot[1], mh - pad) - ink_b
    return ink_a - max(run[0], pad), min(run[1], mh - pad) - ink_b


def _break_points(word: str) -> set[int]:
    """Indeks tempat kata boleh dipenggal, dari kamus pola kalau tersedia."""
    key = word.upper()
    if key in _BREAK_CACHE:
        return _BREAK_CACHE[key]

    dic = _pyphen()
    pts: set[int] = set()
    if dic is not None:
        # pyphen bekerja pada huruf saja; tanda baca di ujung digeser manual
        # supaya indeksnya tetap merujuk ke posisi di kata aslinya.
        head_pad = len(word) - len(word.lstrip("\"'([“‘"))
        core = word[head_pad:].rstrip(".,!?…\"')]”’")
        if core:
            for pos in dic.positions(core.lower()):
                pts.add(head_pad + pos)
    else:
        # Cadangan HANYA saat kamusnya tidak terpasang. Set kosong dari pyphen
        # adalah jawaban yang sah — 'STRANGE' memang tidak punya batas suku kata,
        # dan menimpanya dengan heuristik justru menghasilkan 'STRA-NGE'.
        pts = {
            n for n in range(1, len(word))
            if word[n - 1].isalpha() and word[n].isalpha()
            and word[n - 1].upper() in _VOWELS and word[n].upper() not in _VOWELS
        }

    _BREAK_CACHE[key] = pts
    return pts


def _cjk_break(line: str, font: ImageFont.FreeTypeFont, avail: float) -> tuple[str, str]:
    """Pecah teks CJK per karakter — bahasa Asia tidak pakai spasi antar kata.

    Returns:
        (kepala_yang_muat, sisa) atau ("", line) kalau tidak bisa dipecah.
    """
    n = 0
    w = 0.0
    for ch in line:
        # Dikali _cond() supaya sebanding dengan `avail`, yang datang dari
        # max_width_at() lewat _measure() dan sudah dalam satuan rapat.
        w += font.getlength(ch) * _cond()
        if w <= avail and n < len(line) - 1:
            n += 1
        else:
            break
    if n <= 0 or n >= len(line):
        return "", line
    return line[:n], line[n:]


def _has_cjk(text: str) -> bool:
    return any(_is_cjk(c) for c in text)


def _split_word(word: str, font: ImageFont.FreeTypeFont, avail: float) -> tuple[str, str]:
    """Penggal kata yang lebih lebar dari kolomnya: 'CONTRACTORS' -> 'CONTRAC-', 'TORS'.

    Dipakai sebagai bagian dari pencarian ukuran font, bukan cuma penyelamat saat
    gagal: tanpa penggalan, ukuran font dijepit oleh kata terpanjang dan balon
    lega ikut mengecil mengikutinya.

    Penggalan bebas menghasilkan 'UG-H', 'ENOUG-H', 'WH-AT' — muat, tapi tidak
    layak cetak. Titik penggalannya diambil dari kamus pola pyphen (Knuth-Liang),
    yang tahu 'CON-TRAC-TOR' dan 'WASH-ING'; heuristik vokal->konsonan cuma
    cadangan kalau modulnya tidak terpasang, dan itu memang meleset ('CONTR-ACTOR').
    Dua penjaga tetap berlaku di kedua jalur: kata pendek tidak pernah dipenggal,
    dan tiap pecahan minimal 3 HURUF — dihitung per huruf supaya 'HEARD.' tidak
    lolos lewat ekor 'RD.' yang panjangnya 3 karakter tapi cuma 2 huruf.

    Returns:
        (kepala_dengan_tanda_hubung, sisa) atau ("", word) kalau tetap tidak muat.
    """
    letters = sum(c.isalpha() for c in word)
    if letters < _MIN_HEAD + _MIN_TAIL:
        return "", word

    def accept(n: int) -> tuple[str, str] | None:
        if sum(c.isalpha() for c in word[:n]) < _MIN_HEAD:
            return None
        if sum(c.isalpha() for c in word[n:]) < _MIN_TAIL:
            return None
        head = f"{word[:n]}-"
        # _cond(): `avail` berasal dari _measure(), jadi kepala penggalan harus
        # diukur di satuan yang sama. Tanpa itu penggalan dinilai dengan lebar
        # renggang sementara barisnya digambar rapat — kata dipenggal padahal
        # utuhnya muat, dan tanda hubung itu tepat yang sedang dihilangkan.
        return (head, word[n:]) if font.getlength(head) * _cond() <= avail else None

    # Terpanjang dulu: makin banyak yang muat di baris ini, makin sedikit baris.
    for n in sorted(_break_points(word), reverse=True):
        hit = accept(n)
        if hit:
            return hit
    return "", word


def layout(
    text: str, mask: np.ndarray, size: int, font_path: str,
    allow_overflow: bool = False, hyphenate: bool = False,
    from_top: bool = False,
) -> tuple[bool, list[str], int]:
    """Centroid-outward line growing.

    Bentuk oval muncul sendiri: baris dekat lengkung atas/bawah gagal probe
    lebih awal jadi lebih pendek. Lebih bagus daripada inscribed-rectangle.

    Returns:
        (fits, lines, start_y)
    """
    font = _font(font_path, size)
    words = text.split()
    if not words:
        return True, [], 0

    lh = _line_height(font)
    ink_top, ink_bot = _ink_band(font_path, size)
    cx, cy = _centroid(mask)
    mh, mw = mask.shape[:2]
    pad = int(min(mh, mw) * SETTINGS.pad_ratio)

    def max_width_at(y_top: int) -> float:
        """Lebar tersedia pada baris ini = rongga bebas TERLEBAR di band-nya.

        Bukan ekspansi simetris di sekitar satu x. Lihat _band_run() untuk
        alasannya: rongga interior yang sudah dipotong tetangganya tidak
        simetris terhadap centroid, dan memaksanya simetris membuang separuh
        ruang yang ada — itulah yang membuat 'SORRY.' (r6) tidak pernah muat di
        ketinggian tengah dan terpaksa melorot 39 px ke bawah.
        """
        run = _band_run(mask, y_top + ink_top, y_top + ink_bot)
        if run is None:
            return 0.0
        return max((run[1] - run[0] + 1) - pad * 2, 0.0)

    def axis_of(lines: list[str], top: int) -> int:
        """Sumbu x blok ini; centroid kalau tidak ada irisan rongga yang sah."""
        return block_axis(mask, lines, top, lh, ink_top, ink_bot, font, cx)

    def _center_y(n_lines: int) -> int:
        """y anchor baris pertama supaya TINTA blok terpusat di centroid."""
        ink_h = (n_lines - 1) * lh + (ink_bot - ink_top)
        return cy - ink_h // 2 - ink_top

    def _tops(n_lines: int) -> list[int]:
        """Kandidat y baris pertama: centroid dulu, lalu digeser ke atas-bawah.

        Centroid saja tidak cukup. Interior balon yang sudah dipotong tetangganya
        sering paling lebar di pita yang TIDAK melewati centroid: pada balon
        r12 halaman referensi baris terlebar ada di y=118..150 (93 px bebas)
        sementara centroid memaksa baris pertama ke y=102 yang cuma 44 px —
        dan 'WHAT?' butuh 45 px. Tanpa pencarian ini region itu gagal di SEMUA
        ukuran font, lalu fit() jatuh ke jalur di bawah min_font_size dan
        merendernya 6 px.

        Centroid tetap dicoba pertama supaya balon normal tetap terpusat seperti
        sebelumnya; geseran hanya dipakai kalau yang terpusat memang gagal.
        """
        ink_h = (n_lines - 1) * lh + (ink_bot - ink_top)
        lo, hi = pad - ink_top, mh - pad - ink_h - ink_top
        if hi < lo:
            return [_center_y(n_lines)]
        cands = [int(np.clip(_center_y(n_lines), lo, hi))]
        # Langkah setengah baris: cukup halus untuk menemukan pita lebar, cukup
        # kasar supaya jumlah percobaan tetap belasan, bukan ratusan.
        step = max(lh // 2, 2)
        cands += [t for t in range(lo, hi + 1, step) if t != cands[0]]
        return cands

    def build(top: int) -> tuple[list[str], bool]:
        """Tumbuhkan baris ke bawah mulai dari `top`. bool = semua kata termuat."""
        lines: list[str] = []
        queue = list(words)  # bisa tumbuh saat kata dipenggal
        i = 0
        for _ in range(64):  # batas keras, jangan sampai loop selamanya
            if i >= len(queue):
                return lines, True
            avail = max_width_at(top + len(lines) * lh)
            if avail < size * 0.9 and not allow_overflow:
                # Baris pertama harus muat sesuatu, tapi JANGAN diberi seluruh
                # lebar kotak: di balon oval, lebar kotak jauh melewati garis
                # balon pada baris teratas, dan itu membuat baris pertama
                # menembus keluar. Ukuran ini gagal — fit() yang mengecilkan.
                return lines, False

            line = queue[i]
            j = i + 1
            while j < len(queue) and _measure(f"{line} {queue[j]}", font) <= avail:
                line = f"{line} {queue[j]}"
                j += 1

            # Satu kata pun tidak muat.
            if j == i + 1 and _measure(line, font) > avail:
                if _has_cjk(line):
                    head, tail = _cjk_break(line, font, avail)
                else:
                    head, tail = _split_word(line, font, avail) if hyphenate else ("", line)
                if head:
                    queue[i : i + 1] = [head, tail]  # sisanya ke baris berikutnya
                    line, j = head, i + 1
                elif not allow_overflow:
                    return lines, False  # ukuran font ini gagal

            lines.append(line)
            i = j
        return lines, i >= len(queue)

    def _verify(lines: list[str], start_y: int) -> bool:
        """Tiap baris benar-benar di dalam interior, pada lebar baris itu sendiri.

        Menuntut seluruh lebar mask bebas berarti menuntut sudut-sudut persegi
        dari mask bubble oval — yang tidak akan pernah bebas berapa pun ukuran
        fontnya, jadi tiap region selalu dilaporkan overflow walau tiap barisnya
        sudah lolos probe. Yang benar: cek kotak nyata tiap baris di sekitar
        sumbu blok — sumbu yang sama yang dipakai render_region() menggambar
        (lihat line_axis()), bukan centroid, kalau tidak yang diverifikasi bukan
        tempat tintanya benar-benar jatuh.
        """
        ax = axis_of(lines, start_y)
        for k, line in enumerate(lines):
            lw = _measure(line, font)
            y_top = start_y + k * lh
            if not _row_free(
                mask, y_top + ink_top, y_top + ink_bot, ax - lw / 2, ax + lw / 2
            ):
                return False
        # Blok teks diukur dari tinta baris pertama sampai tinta baris terakhir,
        # bukan dari tepi kotak — alasannya sama seperti di _ink_band().
        return not (start_y + ink_top < pad
                    or start_y + (len(lines) - 1) * lh + ink_bot > mh - pad)

    def _slack(lines: list[str], top: int) -> tuple[int, int]:
        """Sisa ruang atas/bawah blok ini — lihat block_slack() untuk alasannya.

        Diukur di sumbu blok, bukan di centroid: pada interior yang terpotong
        keduanya bisa berjarak belasan px, dan mengukur sisa ruang di kolom yang
        TIDAK dilewati tinta melaporkan ruang semu.
        """
        ax = axis_of(lines, top)
        return block_slack(
            mask, ax, pad,
            _measure(lines[0], font) if lines else 1.0,
            _measure(lines[-1], font) if lines else 1.0,
            top + ink_top,
            top + (len(lines) - 1) * lh + ink_bot,
        )

    def _polish(lines: list[str], top: int) -> tuple[list[str], int]:
        """Geser blok 1 px demi 1 px ke arah yang menyeimbangkan sisa atas/bawah.

        Sapuan _tops() berlangkah setengah baris, jadi kandidat paling seimbang
        pun masih bisa timpang setengah langkah — belasan px di halaman ini,
        cukup untuk terlihat menempel ke satu sisi.

        Pemecahan barisnya DIPERTAHANKAN, tidak dibangun ulang. Versi pertama
        memanggil build(top) di tiap langkah dan berhenti di langkah pertama
        pada keempat region yang tersisa: menggeser blok ke tengah oval
        melebarkan baris yang tersedia, build lalu memuat lebih banyak kata dan
        jumlah barisnya turun, jadi penjaga "jumlah baris harus sama" langsung
        menolaknya. Pemecahan yang sudah dipilih memang sah — yang perlu
        dipastikan cuma masih muat di y baru, dan itu tepat yang diuji _verify
        (tiap baris pada lebarnya sendiri, plus batas pad).
        """
        up, dn = _slack(lines, top)
        best_top, best_bal = top, abs(up - dn)
        step = 1 if dn > up else -1
        for _ in range(abs(dn - up) // 2):
            t = best_top + step
            if not _verify(lines, t):
                break
            u2, d2 = _slack(lines, t)
            if abs(u2 - d2) >= best_bal:
                break
            best_top, best_bal = t, abs(u2 - d2)
        return lines, best_top

    # Jumlah baris menentukan start_y, tapi start_y juga menentukan lebar tiap
    # baris — jadi tebakan awal harus dikoreksi, bukan dipakai apa adanya. Tanpa
    # koreksi ini blok mulai terlalu tinggi, baris-baris bawahnya jatuh di luar
    # balon, dan ukuran yang sebenarnya muat ditolak: 'DID THE CONTRACTOR BAIL?'
    # tertahan di font 16 padahal 21 muat.
    n0 = max(1, int(np.ceil(_measure(text, font) / max(mw - pad * 2, 1))))
    lines, done, start_y = [], False, _center_y(n0)
    # Ambang "sudah cukup terpusat" untuk berhenti menyapu. Sisanya diserahkan
    # ke _polish, yang bisa turun ke ketimpangan ~0 — jadi ambang sekasar
    # setengah baris tidak mengorbankan kerapian, hanya menghemat percobaan.
    tol = max(2, lh // 2)
    # Jumlah baris DIPINDAI, tidak dikoreksi dari satu percobaan. Versi
    # sebelumnya memakai `nxt = len(lines) + 1` dari build di y terpusat, dan
    # koreksi itu MELOMPATI jumlah baris yang benar: di r12 halaman ini n0=3,
    # build di y terpusat memuat 4 baris tapi gagal (ok=False), jadi n naik
    # langsung ke 5 dan n=4 tidak pernah dicoba sama sekali. Hasilnya 5 baris
    # yang mentok di ketimpangan 41 px, padahal 4 baris muat dengan
    # ketimpangan 1 px pada margin yang sama persis. Memindai n0..n0+4 dan
    # memilih yang paling seimbang menghapus lompatan itu tanpa melonggarkan
    # batas apa pun — margin build tetap pad*2 seperti sebelumnya.
    hit: tuple[list[str], int, int] | None = None
    for n in range(n0, n0 + 5):
        tops = _tops(n)
        # Kandidat yang muat DIPILIH yang paling seimbang, bukan yang pertama.
        # Versi sebelumnya `break` pada fit pertama, dan karena sapuan berjalan
        # dari tepi atas balon ke bawah, yang pertama muat sering jauh dari
        # tengah: pada halaman referensi start_y yang diterima duduk di
        # peringkat kandidat sampai ke-21, dan 10 dari 13 balon meleset dari
        # tengah (terburuk 40 px). Sapuannya sendiri tetap wajib ada — lihat
        # _tops() — jadi yang diubah cuma kriteria pemilihannya.
        for top in tops:
            cand, ok_all = build(top)
            if top == tops[0]:
                lines, done, start_y = cand, ok_all, top
            if not (ok_all and len(cand) == n and _verify(cand, top)):
                continue
            up, dn = _slack(cand, top)
            bal = abs(up - dn)
            if hit is None or bal < hit[2]:
                hit = (cand, top, bal)
            if bal <= tol:
                break
        if hit is not None and hit[2] <= tol:
            break  # sudah terpusat; jumlah baris yang lebih besar tak perlu
    if hit is not None:
        lines, start_y = _polish(hit[0], hit[1])
        done = True

    if from_top:
        # Teks terlalu panjang untuk ukuran berapa pun: susun dari ATAS balon
        # ke bawah supaya AWAL kalimat yang terlihat (bukan potongan tengah
        # yang tidak terbaca); kelebihan dipotong di tepi bawah oleh klip.
        lines, done = build(pad)
        return (done or allow_overflow), lines, pad

    if not done and not allow_overflow:
        return False, [], 0

    return (_verify(lines, start_y) or allow_overflow), lines, start_y



def _search(
    text: str, mask: np.ndarray, lo: int, hi: int, font_path: str, hyphenate: bool
) -> tuple[int, list[str], int] | None:
    """Ukuran terbesar yang masih muat pada [lo, hi], atau None kalau tidak ada."""
    ok, lines, y = layout(text, mask, hi, font_path, hyphenate=hyphenate)
    if ok:
        return hi, lines, y  # jalur cepat: ukuran asli sudah muat

    best: tuple[int, list[str], int] | None = None
    a, b = lo, hi
    while b - a > 1:
        mid = (a + b) // 2
        ok, lines, y = layout(text, mask, mid, font_path, hyphenate=hyphenate)
        if ok:
            best, a = (mid, lines, y), mid
        else:
            b = mid

    # Loop di atas hanya menguji titik tengah, jadi lo — batas bawahnya sendiri —
    # tidak pernah dicoba. Kalau semua titik tengah gagal, best masih None padahal
    # ukuran minimum belum tentu gagal; tanpa cek ini region dilaporkan overflow
    # tanpa pernah diuji pada ukuran yang justru paling mungkin muat.
    if best is None:
        ok, lines, y = layout(text, mask, lo, font_path, hyphenate=hyphenate)
        if ok:
            best = (lo, lines, y)
    return best


def _block_bal(lines: list[str], y: int, mask: np.ndarray, size: int,
               font_path: str) -> int:
    """Ketimpangan sisa ruang atas vs bawah blok, dalam px. 0 = terpusat."""
    if not lines:
        return 0
    font = _font(font_path, size)
    lh = _line_height(font)
    ink_top, ink_bot = _ink_band(font_path, size)
    mh, mw = mask.shape[:2]
    pad = int(min(mh, mw) * SETTINGS.pad_ratio)
    # Sumbu blok, bukan centroid — sama seperti _slack() di dalam layout().
    ax = line_axis(mask, lines, y, size, font_path)
    up, dn = block_slack(
        mask, ax, pad, _measure(lines[0], font), _measure(lines[-1], font),
        y + ink_top, y + (len(lines) - 1) * lh + ink_bot,
    )
    return abs(up - dn)


def _bal_tol(size: int, font_path: str) -> int:
    """Ambang 'sudah terpusat' — setengah tinggi baris, sama seperti layout()."""
    return max(2, _line_height(_font(font_path, size)) // 2)


def _rebalance(
    text: str, mask: np.ndarray, font_path: str,
    cand: tuple[int, list[str], int], hyphenate: bool,
) -> tuple[int, list[str], int]:
    """Turun ukuran sedikit kalau ukuran terpilih mustahil ditata seimbang.

    Ukuran terbesar yang MUAT tidak selalu ukuran yang bisa dirapikan. Pada r12
    halaman referensi 'IS IT? LEMME SEE, C'MON~!' di ukuran 15 mentok pada
    ketimpangan 43 px — bukan karena pencariannya kurang teliti, tapi karena
    tidak ada satu pun kombinasi pemecahan baris dan y legal yang lebih baik
    (probe_r12_exhaust.py memindai semuanya). Di ukuran 14 blok yang sama turun
    ke 1 px. Tanpa langkah ini teks menempel ke satu sisi balon sejauh 21 px
    dari tengah, dan itu terlihat langsung.

    Yang TIDAK dilakukan di sini: melonggarkan margin build. Varian itu diukur
    (probe_margin.py) dan memang menyembuhkan r12, tapi median jarak tinta ke
    garis balon seluruh halaman jatuh dari 3 px ke 0 px — menukar satu balon
    timpang dengan tinta menempel garis di enam balon lain.

    Turunnya dibatasi _BAL_MAX_DROP dan region yang sudah seimbang di ukuran
    terpilih tidak pernah masuk loop, jadi ukuran font halaman tidak ikut
    mengecil: pada halaman referensi hanya r12 yang turun (15 -> 14).
    """
    size, lines, y = cand
    if _block_bal(lines, y, mask, size, font_path) <= _bal_tol(size, font_path):
        return cand
    floor = max(size - _BAL_MAX_DROP, min_font())
    for s in range(size - 1, floor - 1, -1):
        ok, ls, yy = layout(text, mask, s, font_path, hyphenate=hyphenate)
        if not ok or not ls:
            continue
        if _block_bal(ls, yy, mask, s, font_path) <= _bal_tol(s, font_path):
            return s, ls, yy
    return cand


def set_page_width(w: int) -> None:
    """Catat lebar halaman yang sedang dikerjakan (untuk min_font())."""
    global _PAGE_W
    _PAGE_W = int(w) if w and w > 0 else 0


def min_font(page_w: int | None = None) -> int:
    """Lantai ukuran font untuk halaman selebar `page_w`, BUKAN angka mutlak.

    Kenapa berskala: SETTINGS.min_font_size dikalibrasi pada satu resolusi
    (CONTOH/2.webp, 1134 px). Halaman lain datang di resolusi lain, dan lantai
    yang tidak ikut menyusut berubah jadi PLAFON di halaman kecil — bukan lantai.
    Terukur pada hasilnew/jp_6.JPG (698 px, probe_floor6.py): lantai 11 px
    membuat region_font_cap() semua balon mentok di 11, anggaran yang dikirim ke
    model jadi 2-39 karakter, dan _max_feasible() atas wording typeset referensi
    mengembalikan 0 di 7 dari 8 balon — artinya pipeline meminta model menulis
    lebih pendek daripada yang sebenarnya muat, lalu tetap menolak wording yang
    muat. Itulah sebab 'translate-nya sedikit banget' dibanding hasilnew/6.JPG.

    Huruf referensi di halaman itu diukur 4-7 px tinggi (probe_refsize.py, modus
    4, median 5 pada 728 px) = ukuran font 5-8. Jadi typesetter manusia memang
    turun di bawah 11 px pada resolusi ini; lantai 11 bukan batas keterbacaan,
    melainkan batas keterbacaan DI 1134 px.

    Skalanya linear terhadap lebar halaman karena keterbacaan bergantung pada
    ukuran RELATIF terhadap halaman (dan balonnya, yang juga ikut mengecil),
    bukan pada jumlah piksel. Dibatasi dua arah: tidak pernah melebihi
    min_font_size (halaman besar tetap memakai angka terkalibrasi, tidak
    diperbesar diam-diam) dan tidak pernah di bawah min_font_abs.
    """
    w = _PAGE_W if page_w is None else int(page_w)
    ref = max(int(SETTINGS.min_font_ref_width), 1)
    base = int(SETTINGS.min_font_size)
    if w <= 0 or w >= ref:
        return base
    scaled = int(round(base * w / ref))
    return int(np.clip(scaled, int(SETTINGS.min_font_abs), base))


def emergency_floor() -> int:
    """Lantai jalur darurat fit(), ikut berskala bersama min_font().

    _MIN_FONT_FLOOR (9) berjarak 2 px di bawah min_font_size (11) yang menjadi
    kalibrasinya. Jarak itulah yang dipertahankan, bukan angka 9-nya: kalau
    lantai normal turun ke 7 px di halaman 698 px sementara lantai darurat tetap
    9, jalur darurat berada DI ATAS lantai normal, penjaga `lo - 1 >= floor`
    tidak pernah benar, dan balon yang cuma butuh 1 px lagi langsung dipotong di
    tepi bawah alih-alih dikecilkan sedikit.
    """
    gap = max(int(SETTINGS.min_font_size) - _MIN_FONT_FLOOR, 0)
    return int(max(min_font() - gap, int(SETTINGS.min_font_abs)))


def renders_ok(text: str, mask: np.ndarray, font_path: str) -> tuple[bool, int]:
    """(muat?, ukuran) untuk teks INI di balon INI, diukur seperti render nyata.

    Bedanya dengan _max_feasible(): fungsi ini memakai fit() apa adanya, jadi
    penggalan kata dan lantai darurat IKUT dihitung. _max_feasible() melarang
    penggalan karena tugasnya lain — ia mengukur plafon 'muat utuh' untuk
    kalibrasi ukuran, bukan menjawab 'apakah kalimat ini bisa dicetak'.

    Memakai _max_feasible() sebagai kriteria lulus terukur salah arah. Pada
    hasilnew/jp_6.JPG (698 px) wording typeset referensi untuk r3
    ("CAN'T HELP IT ♥", interior 46x87) mengembalikan feasible 0 — tidak muat
    utuh pada ukuran mana pun di atas lantai — padahal fit() merendernya bersih
    di 6 px dengan tiga baris, dan 6 px persis yang dipakai typesetter manusia di
    resolusi ini (probe_r34.py, probe_refsize.py). Jadi validator lama menolak
    justru wording yang sedang ditiru, lalu menyuruh model menulis lebih pendek —
    itu sebab langsung 'translate-nya sedikit banget'.

    Yang tersisa sebagai cacat sungguhan cuma satu: fit() melaporkan luber, yang
    berarti barisnya benar-benar dipotong di tepi bawah balon.
    """
    size, _lines, _y, over = fit(text, mask, region_font_cap(mask), font_path)
    return (not over), int(size)


def fit(text: str, mask: np.ndarray, est_font: float, font_path: str) -> tuple[int, list[str], int, bool]:
    """Binary search pada [MIN_FONT, est_font]. Jangan pernah di atas est.

    Fit-search yang memaksimalkan ukuran akan overshoot dan hasilnya lebih
    besar dari teks aslinya.
    """
    hi = int(np.clip(round(est_font), min_font(), SETTINGS.max_font_size))
    lo = min_font()

    plain = _search(text, mask, lo, hi, font_path, hyphenate=False)
    if plain is not None and plain[0] >= hi:
        # Jalur cepat tetap lewat _rebalance: plafon yang muat belum tentu
        # plafon yang bisa ditata terpusat — lihat docstring _rebalance().
        s, ls, y = _rebalance(text, mask, font_path, plain, hyphenate=False)
        return s, ls, y, False

    # Penggalan kata adalah BAGIAN dari pencarian ukuran, bukan jalan terakhir.
    # Tanpa baris ini ukuran font dijepit oleh kata terpanjang: 'CONTRACTORS'
    # cuma muat utuh di font 12 pada balon selebar 118 px, jadi balon setinggi
    # 218 px diisi tiga baris kecil dan sisanya kosong — jauh dari gambar
    # referensi yang tiap balonnya terisi penuh. Dengan penggalan, kata panjang
    # pecah jadi dua baris dan font bisa naik ke belasan-duapuluhan.
    hyph = _search(text, mask, lo, hi, font_path, hyphenate=True)

    cands = [c for c in (plain, hyph) if c is not None]
    if not cands:
        # Degradasi bertingkat untuk balon yang teksnya tidak muat di
        # min_font_size pun. Urutannya sengaja: teks UTUH yang sedikit lebih
        # kecil jauh lebih baik dibaca daripada teks terpotong.
        #
        # Batas bawahnya emergency_floor() — lantai bernama dengan alasan
        # tertulis, bukan `min_font_size // 2` yang diam-diam mengizinkan 5-6 px,
        # dan ikut berskala resolusi bersama lo supaya di halaman kecil ia tetap
        # BERADA DI BAWAH lo. Dijaga `lo - 1 >= floor` supaya kalau lantai normal
        # sudah menyentuh lantai darurat, _search tidak dipanggil dengan hi < lo
        # (jalur cepatnya akan mengembalikan ukuran di bawah lantai tanpa pernah
        # mengujinya).
        #
        # UTUH DIUJI DULU, dan itu bukan kosmetik. Sebelumnya jalur ini hanya
        # memanggil _search(hyphenate=True), jadi di seluruh jalur darurat tidak
        # pernah ada kandidat utuh yang bisa menang — tanda hubung menang tanpa
        # lawan, dan _HYPHEN_MIN_GAIN di atas tidak berlaku sama sekali di sini.
        # Itulah sebab langsung 'WON-/DER' di r3 halaman jp_6: teksnya cuma
        # 'NO WONDER ♥!', kolomnya sempit, jadi ia jatuh ke jalur ini dan langsung
        # dipenggal. Ambangnya sama dengan jalur normal supaya aturannya satu:
        # tanda hubung hanya kalau ia membeli >= 33% ukuran font.
        floor = emergency_floor()
        if lo - 1 >= floor:
            lp = _search(text, mask, floor, lo - 1, font_path, hyphenate=False)
            lh_ = _search(text, mask, floor, lo - 1, font_path, hyphenate=True)
            if lp is not None and (lh_ is None or lp[0] >= lh_[0] * _HYPHEN_MIN_GAIN):
                low = lp
            else:
                low = lh_
        else:
            low = None
        if low is not None:
            return low[0], low[1], low[2], False
        # Terlalu panjang untuk ukuran apa pun: susun dari atas balon
        # (awal kalimat terlihat), sisanya dipotong di tepi bawah.
        _, lines, y = layout(
            text, mask, lo, font_path, allow_overflow=True, hyphenate=True,
            from_top=True,
        )
        return lo, lines, y, True

    # Utuh menang kecuali harganya benar-benar mahal — lihat _HYPHEN_MIN_GAIN.
    if plain is None:
        best = hyph
        hy = True
    elif hyph is None or plain[0] >= hyph[0] * _HYPHEN_MIN_GAIN:
        best = plain
        hy = False
    else:
        best = hyph
        hy = True
    s, ls, y = _rebalance(text, mask, font_path, best, hyphenate=hy)
    return s, ls, y, False


# ---------------------------------------------------------------- render


def _is_cjk(ch: str) -> bool:
    """Kana, kanji, Hangul, dan ideograf CJK."""
    o = ord(ch)
    return (
        0x3040 <= o <= 0x30FF  # kana
        or 0x3400 <= o <= 0x4DBF  # CJK ext-A
        or 0x4E00 <= o <= 0x9FFF  # CJK unified
        or 0xF900 <= o <= 0xFAFF  # kompatibilitas
        or 0xAC00 <= o <= 0xD7AF  # Hangul
    )


def _is_arabic(ch: str) -> bool:
    o = ord(ch)
    return 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0xFB50 <= o <= 0xFDFF


def _is_thai(ch: str) -> bool:
    return 0x0E00 <= ord(ch) <= 0x0E7F


def _char_font(ch: str, main: ImageFont.FreeTypeFont, cmap: frozenset[int],
               size: int) -> ImageFont.FreeTypeFont:
    """Font per karakter: utama -> CJK -> Arab -> Thai -> NotoSans -> simbol.

    Anime Ace cuma ~159 glyph; terjemahan non-Inggris (aksen, Cyrillic, CJK,
    Arab, Thai, ...) butuh fallback ini supaya tidak jadi kotak tofu.

    Tiap kandidat DIPERIKSA punya glyph-nya, tidak cuma dicocokkan lewat rentang
    aksara. Versi sebelumnya memilih font dari rentang saja lalu jatuh ke
    NotoSans sebagai penampung terakhir, dan justru simbol yang wajib bertahan
    menurut plan.txt yang jadi korban: 〜 (U+301C) ada di blok CJK Symbols and
    Punctuation yang tidak dicakup _is_cjk(), jadi ia — bersama ♡ ♪ ♫ ☆ ★ —
    dirutekan ke NotoSans, satu-satunya font di rantai ini yang TIDAK punya satu
    pun dari simbol itu. Hasilnya kotak tofu di dalam balon, padahal
    NotoSansCJKjp punya 〜 ～ ♥ ♡ ♪ ☆ ★ dan NotoSansSymbols2 punya ♥ ♡ ☆ ★ ❤.
    Dengan pemeriksaan cmap, rantainya berhenti di font pertama yang benar-benar
    bisa menggambar karakter itu.

    Satu pengecualian, lihat _FORCE_SYMBOL: untuk simbol emosi, "punya glyph"
    tidak sama dengan "punya glyph yang BENAR". anime_ace memetakan U+2665 ke
    huruf Cyrillic `yat`, jadi cek cmap saja meloloskan bentuk yang salah.
    Simbol-simbol itu selalu diambil dari font simbol/CJK, bukan font utama.
    """
    o = ord(ch)
    if o in _FORCE_SYMBOL:
        for path in (_SYMBOL_PATH, _CJK_PATH, _FALLBACK_PATH):
            if path is not None and o in _cmap(str(path)):
                return _font(str(path), size)
        # Tidak ada font pengganti yang punya simbolnya: font utama tetap lebih
        # baik daripada kotak tofu, walau bentuknya tidak ideal.
        return main
    if ch.isspace() or (cmap and o in cmap):
        return main
    # Urutan preferensi tetap: aksara yang cocok dulu, lalu penampung umum.
    chain = []
    if _is_cjk(ch):
        chain.append(_CJK_PATH)
    if _is_arabic(ch):
        chain.append(_ARABIC_PATH)
    if _is_thai(ch):
        chain.append(_THAI_PATH)
    chain += [_FALLBACK_PATH, _CJK_PATH, _SYMBOL_PATH]
    fallback = None
    for path in chain:
        if path is None:
            continue
        p = str(path)
        cm = _cmap(p)
        if o in cm:
            return _font(p, size)
        # cmap kosong = fontTools tidak terpasang; font itu tetap dipakai sebagai
        # cadangan terakhir supaya perilakunya tidak lebih buruk dari sebelumnya.
        if not cm and fallback is None:
            fallback = p
    if fallback is not None:
        return _font(fallback, size)
    return main


def _needs_fallback(line: str, cmap: frozenset[int]) -> bool:
    """Baris ini butuh digambar per-karakter?

    Dua sebab, dan sebab kedua yang dulu terlewat: (1) ada karakter yang font
    utama TIDAK punya, dan (2) ada simbol emosi yang font utama punya tapi
    dengan bentuk yang salah (_FORCE_SYMBOL). Tanpa syarat kedua, baris seperti
    'I LOVE YOU ♥' lolos sebagai "semua ada" lalu digambar satu kali dengan font
    utama, jadi _char_font() tidak pernah dipanggil dan pembetulan simbolnya
    tidak berpengaruh sama sekali.
    """
    return any(
        ord(c) in _FORCE_SYMBOL or (cmap and ord(c) not in cmap)
        for c in line if not c.isspace()
    )


def _draw_line(
    draw: ImageDraw.ImageDraw, xy: tuple[float, float], line: str,
    font: ImageFont.FreeTypeFont, fill: tuple[int, int, int],
    cmap: frozenset[int], size: int, stroke: int = 0,
) -> None:
    """Gambar per-karakter dengan fallback multi-script (CJK/aksen/simbol)."""
    if not _needs_fallback(line, cmap):
        draw.text(
            xy, line, font=font, fill=fill, anchor="la",
            stroke_width=stroke, stroke_fill=(255, 255, 255) if stroke else None,
        )
        return

    x, y = xy
    for ch in line:
        f = _char_font(ch, font, cmap, size)
        draw.text(
            (x, y), ch, font=f, fill=fill, anchor="la",
            stroke_width=stroke, stroke_fill=(255, 255, 255) if stroke else None,
        )
        x += f.getlength(ch)


def _line_width(line: str, font: ImageFont.FreeTypeFont, cmap: frozenset[int], size: int) -> float:
    """Lebar sejati termasuk glyph fallback — kalau tidak, center-nya meleset.

    Harus memakai syarat yang SAMA dengan _draw_line(), termasuk _FORCE_SYMBOL:
    glyph hati dari font simbol lebarnya beda dari glyph `yat` anime_ace, jadi
    kalau pengukuran memakai font utama sementara penggambaran memakai font
    simbol, baris itu diukur salah dan center-nya bergeser.

    Sudah dikali _cond(): inilah lebar yang BENAR-BENAR tergambar, karena
    render_region() memampatkan tile-nya dengan faktor yang sama.
    """
    if not _needs_fallback(line, cmap):
        return font.getlength(line) * _cond()
    return sum(_char_font(c, font, cmap, size).getlength(c) for c in line) * _cond()



def _bg_luminance(img: np.ndarray, region: Region) -> float:
    """Median luminance interior bubble pada gambar (teks sudah terhapus).

    Dipakai memilih warna tinta: putih di interior gelap, hitam di terang.
    Pipeline memanggil typeset pada halaman bersih, jadi nilai ini = warna
    latar sungguhan. Kalau bubble_mask tidak tersedia, median seluruh crop
    cukup akurat karena interior selalu mayoritas piksel.
    """
    if region.bubble_bbox is None:
        return 255.0
    bx1, by1, bx2, by2 = region.bubble_bbox
    crop = img[by1:by2, bx1:bx2]
    if crop.size == 0:
        return 255.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    mask = region.bubble_mask
    if mask is None or mask.shape[:2] != gray.shape:
        return float(np.median(gray))
    vals = gray[mask > 0]
    return float(np.median(vals)) if vals.size else float(np.median(gray))


def _region_box_mask(region: Region) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Kotak render + mask interior region. Persegi 255 kalau mask tidak cocok."""
    box = region.bubble_bbox or region.bbox
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    mask = region.bubble_mask
    if mask is None or mask.shape[:2] != (bh, bw):
        mask = np.full((bh, bw), 255, np.uint8)
    return box, mask


def _max_feasible(text: str, mask: np.ndarray, font_path: str) -> int:
    """Ukuran terbesar yang muat di balon ini TANPA penggalan; 0 kalau nihil.

    Plafonnya dari GEOMETRI balon, bukan tinggi glyph Jepang. est_font_size
    diukur MELINTANG kolom vertikal Jepang (textmask._glyph_height) dan
    variansinya besar, jadi memakainya sebagai plafon per region membuat ukuran
    font beda sampai ~2x antar balon dalam satu panel yang sama.
    """
    mh, mw = mask.shape[:2]
    pad = int(min(mh, mw) * SETTINGS.pad_ratio)
    lo = min_font()
    hi = int(np.clip(mh - 2 * pad, lo, SETTINGS.max_font_size))
    best = _search(text, mask, lo, hi, font_path, hyphenate=False)
    return best[0] if best else 0


# Ukuran font = rasio tetap terhadap SISI TERPENDEK interior balon. Kedua angka
# diukur, bukan ditaksir:
#   0.117 = cap_height / min(sisi interior) di CONTOH/2.webp (probe_refnative.py,
#           13 balon; p25 0.108 p75 0.150) — jauh lebih stabil daripada
#           cap_height-nya sendiri, yang berkisar 13..27 px (sebaran 2.08x).
#   0.844 = cap_height / ukuran font Anime Ace, konstan pada ukuran 11..32
#           (probe_cap.py, dari render "HAMBURG").
#
# CATATAN: ini MENYIMPANG dari plan.txt langkah 4, yang menyuruh satu ukuran
# seragam untuk seluruh halaman ("Ini persis pola typesetter referensi"). Ukuran
# mengatakan sebaliknya — referensi justru MENSKALAKAN teks ke besar balon. Tiga
# model diuji terhadap ukuran referensi yang terukur (probe_model.py):
#   seragam-halaman (persentil 35)  galat rata-rata 4.31 px
#   proporsional balon              galat rata-rata 2.71 px  <- ini
#   proporsional per panel          galat rata-rata 4.19 px
# Kontingensi di plan ("kelompokkan region per panel") ternyata LEBIH BURUK
# daripada proporsional biasa di halaman ini, jadi tidak dipakai.
_REF_CAP_PER_MIN = 0.117
_CAP_PER_SIZE = 0.844


def region_font_cap(mask: np.ndarray) -> int:
    """Plafon ukuran font untuk satu balon, dari geometrinya sendiri.

    Plafon, bukan keputusan akhir: fit() masih menurunkannya kalau teksnya
    memang tidak muat. Yang penting plafon ini TIDAK berasal dari est_font_size
    — tinggi glyph Jepang diukur melintang kolom vertikal, variansinya besar,
    dan itulah sebab awal ukuran font beda ~2x antar balon satu panel.
    """
    mn = min(mask.shape[:2])
    size = int(round(mn * _REF_CAP_PER_MIN / _CAP_PER_SIZE))
    return int(np.clip(size, min_font(), SETTINGS.max_font_size))


# ---------------------------------------------------------------- anggaran balon
#
# Berapa karakter yang SUNGGUH muat di satu balon — dipakai jalur LLM untuk
# memberi tahu penerjemah harus sependek apa. Diletakkan di sini, bukan di
# translate.py, karena angkanya harus keluar dari mesin tata letak YANG SAMA
# dengan yang merender. Anggaran yang dihitung terpisah bisa melenceng dari
# kenyataan tanpa ada yang tahu.
#
# Kenapa perlu sama sekali: "buatlah pendek" bukan perintah, itu selera. Percobaan
# yang cuma menyuruh "PREFER THE SHORTER natural phrasing" tetap mengembalikan
# 'SORRY TO BARGE IN.' (18 karakter) untuk balon yang memuat 6 — dan model tidak
# melanggar apa pun, ia memang tidak PUNYA cara menaati perintah tanpa angka. Ia
# tidak melihat balonnya.
#
# Teks pengisi, dan dua sifatnya yang penting (keduanya ketemu dari kegagalan,
# bukan dipikirkan lebih dulu):
#
# 1. GRANULARITAS. layout() bekerja per kata, jadi anggaran hanya bisa melompat
#    sebesar kata berikutnya. Pengisi versi pertama dimulai "THE PREZ WAS PUTTING
#    TOGETHER ..." — lompatan 20 -> 29 karakter, dan tujuh balon berbeda semuanya
#    melaporkan soft=20 karena mentok di kata 'TOGETHER' yang sama. Angka itu
#    bukan sifat balonnya, itu sifat pengisinya. Kata di sini 2-6 huruf berputar.
#
# 2. LEBAR GLYPH. Anime Ace tidak monospace; 'W' hampir dua kali 'I'. Pengisi
#    harus mendekati frekuensi huruf Inggris — 'AAAA' membuat anggaran terlalu
#    pesimistis, 'IIII' terlalu optimistis.
_BUDGET_FILLER = (
    "SO THE PREZ HAS ALL THE NOTES AND I SEE THEM HERE ON HER DESK "
    "AT ONE SIDE OF THE ROOM IT IS SO NICE AND I DO LIKE IT A LOT "
    "LET ME TAKE A LOOK AT THIS ONE FOR JUST A BIT MORE OK THANKS "
) * 8
_BUDGET_WORDS = _BUDGET_FILLER.split()


def char_budget(mask: np.ndarray, size: int, font_path: str) -> int:
    """Karakter terbanyak (batas kata) yang masih muat UTUH pada `size`.

    Dicari lewat jumlah KATA, bukan potongan karakter sembarang: layout() bekerja
    per kata, jadi memotong di tengah kata memberi jawaban yang tidak pernah bisa
    dicapai teks sungguhan.
    """
    if size <= 0:
        return 0
    words = _BUDGET_WORDS
    whole = " ".join(words)
    if layout(whole, mask, size, font_path, hyphenate=False)[0]:
        return len(whole)
    lo, hi = 0, len(words)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if layout(" ".join(words[:mid]), mask, size, font_path, hyphenate=False)[0]:
            lo = mid
        else:
            hi = mid
    return len(" ".join(words[:lo]))


def max_word_len(mask: np.ndarray, size: int, font_path: str) -> int:
    """Kata TERPANJANG (tanpa spasi) yang masih muat satu baris pada `size`.

    Angka kedua ini wajib ada, dan itu ketemu dari percontohan yang gagal: satu
    balon anggaran totalnya 39 karakter, tapi 'MY APOLOGIES' yang cuma 12 karakter
    tetap menghasilkan tanda hubung. Yang menjepit BUKAN panjang kalimat melainkan
    'APOLOGIES' — satu kata 9 huruf tidak muat di lebar 68 px, dan begitu satu kata
    tidak muat, layout() hanya punya dua pilihan: penggal atau gagal. Typeset
    referensi memilih kata lain sama sekali ('SORRY.'), dan ITULAH keputusan yang
    perlu disampaikan ke penerjemah.
    """
    if size <= 0:
        return 0
    best = 0
    for n in range(2, 25):
        # Konsonan/vokal bergantian: lebar rata-rata wajar, bukan 'WWWW'/'IIII'.
        probe = ("RONALDESTI" * 3)[:n]
        if layout(probe, mask, size, font_path, hyphenate=False)[0]:
            best = n
        else:
            break
    return best


def region_budget(region: Region, font_path: str) -> dict[str, int]:
    """Anggaran satu balon: {cap, soft, hard, word_soft, word_hard}.

    soft = muat pada region_font_cap() -> ukuran yang DIINGINKAN (proporsional ke
           besar balon). hard = muat pada min_font_size -> batas mutlak; lewat dari
           ini fit() jatuh ke jalur darurat dan hasilnya tidak terbaca.

    Keduanya perlu. Cuma soft: terlalu ketat — wording typeset profesional sendiri
    melewatinya di balon padat (61 karakter di satu balon halaman referensi), jadi
    menjadikannya batas keras berarti menolak hasil yang justru ditiru. Cuma hard:
    terlalu longgar — teks jadi muat tapi selalu di ukuran minimum. Jadi soft =
    target, hard = batas.
    """
    mask = _region_box_mask(region)[1]
    cap = region_font_cap(mask)
    lo = min_font()
    return {
        "cap": cap,
        "soft": char_budget(mask, cap, font_path),
        "hard": char_budget(mask, lo, font_path),
        "word_soft": max_word_len(mask, cap, font_path),
        "word_hard": max_word_len(mask, lo, font_path),
    }


def render_region(img: np.ndarray, region: Region, font_path: str | None = None, own_map: np.ndarray | None = None, forb_map: np.ndarray | None = None, size_cap: int | None = None) -> np.ndarray:
    """Tulis terjemahan ke halaman. Center dua sumbu; ALL CAPS hanya English.

    Warna tinta menyesuaikan bubble: putih di interior gelap, hitam di terang.

    Tiap baris digambar ke overlay sendiri lalu di-shear di sekitar baseline-nya.
    Anime Ace versi regular tegak lurus sedangkan gambar referensi miring; shear
    per-baris inilah yang meniru italic sungguhan — men-shear seluruh blok
    sekaligus akan mendorong baris teratas keluar dari balon.

    `size_cap` = plafon ukuran font. Kalau None, diambil dari geometri balon
    lewat region_font_cap() — BUKAN dari est_font_size, yang berasal dari tinggi
    glyph Jepang dan variansinya besar.
    """
    if not region.translation or region.is_protected:
        return img
    font_path = font_path or FONT_USED
    if not font_path:
        return img

    box, mask = _region_box_mask(region)
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    if own_map is None:
        # Pemanggil langsung (tanpa render_page): bangun peta halaman sendiri.
        ih, iw = img.shape[:2]
        own_map = np.zeros((ih, iw), np.uint8)
        own_map[by1 : by1 + bh, bx1 : bx1 + bw] = mask

    text = region.translation.upper() if SETTINGS.force_upper else region.translation
    cap = size_cap if size_cap else region_font_cap(mask)
    size, lines, start_y, overflow = fit(text, mask, cap, font_path)
    region.final_font_size = size
    region.lines = lines
    region.overflowed = overflow
    if not lines:
        return img

    font = _font(font_path, size)
    cmap = _cmap(font_path)
    lh = _line_height(font)
    # Sumbu x = sumbu yang dipakai layout() menilai blok ini muat, bukan centroid.
    # Pada interior yang dipotong tetangganya keduanya bisa berjarak belasan px,
    # dan menggambar di centroid setelah memverifikasi di sumbu blok berarti
    # tinta jatuh di tempat yang tidak pernah diuji — bisa menembus garis balon.
    cx = line_axis(mask, lines, start_y, size, font_path)

    # Warna tinta mengikuti bubble: putih di interior gelap, hitam di terang.
    # Dulu selalu hitam murni - di bubble hitam terjemahan tidak terlihat.
    # Threshold 128 sengaja SAMA dengan _bubble_interior di textmask.py.
    fill = (255, 255, 255) if _bg_luminance(img, region) < 128 else (0, 0, 0)

    # Teks di dalam balon: tanpa stroke, sesuai referensi. Teks bebas di atas
    # art butuh stroke putih supaya tetap terbaca.
    stroke = 0 if region.bubble_bbox is not None else max(2, size // 9)
    k = SETTINGS.oblique
    pad = int(abs(k) * lh) + stroke + 4

    pil = Image.fromarray(img).convert("RGBA")
    cnd = _cond()
    for i, line in enumerate(lines):
        w = _line_width(line, font, cmap, size)   # lebar TERGAMBAR (sudah rapat)
        wn = w / cnd                              # lebar renggang, untuk kanvas
        tile = Image.new("RGBA", (int(wn) + pad * 2, lh + pad * 2), (0, 0, 0, 0))
        _draw_line(
            ImageDraw.Draw(tile), (pad, pad), line, font, fill, cmap, size, stroke
        )
        if k or cnd != 1.0:
            th = tile.height
            # Satu transform untuk DUA hal, bukan dua transform berurutan: tiap
            # resample memakan ketajaman, dan pada cap 6-8 px huruf kedua kalinya
            # sudah kabur. Koefisien AFFINE PIL adalah pemetaan BALIK
            # (in = a*out + b*out_y + c), jadi rapat x=cnd di sekitar x=pad
            # ditambah shear k memberi:
            #     in_x = pad + (out_x - pad + k*out_y - k*th/2) / cnd
            # Kanvas sengaja tetap selebar versi renggang: tintanya menyusut ke
            # [pad, pad+w], sisanya transparan dan tidak berbiaya apa pun.
            tile = tile.transform(
                tile.size, Image.AFFINE,
                (1 / cnd, k / cnd, pad - (pad + k * th / 2) / cnd, 0, 1, 0),
                resample=Image.BICUBIC,
            )
        tx = int(bx1 + cx - w / 2) - pad
        ty = by1 + start_y + i * lh - pad
        tile = _clip_to_mask(tile, tx, ty, own_map, forb_map)
        _paste(pil, tile, tx, ty)

    return np.asarray(pil.convert("RGB"), dtype=np.uint8)


def _paste(base: Image.Image, tile: Image.Image, x: int, y: int) -> None:
    """Composite dengan clipping — alpha_composite raise kalau tile lewat tepi."""
    x0, y0 = max(0, -x), max(0, -y)
    x1 = min(tile.width, base.width - x)
    y1 = min(tile.height, base.height - y)
    if x1 <= x0 or y1 <= y0:
        return
    if (x0, y0, x1, y1) != (0, 0, tile.width, tile.height):
        tile = tile.crop((x0, y0, x1, y1))
    base.alpha_composite(tile, dest=(x + x0, y + y0))


def _clip_to_mask(
    tile: Image.Image, tx: int, ty: int,
    own: np.ndarray, forb: np.ndarray | None,
) -> Image.Image:
    """Hapus alpha tile di luar balon sendiri / di dalam balon tetangga.

    Inilah jaminan 'tidak saling timpa': teks yang meluap dipotong di garis
    balon (own), dan teks tidak pernah ditulis di atas interior balon region
    lain (forb). Mask 255 penuh (region tanpa balon) = tidak memotong apa pun.
    """
    h, w = own.shape[:2]
    cx0, cy0 = max(tx, 0), max(ty, 0)
    cx1, cy1 = min(tx + tile.width, w), min(ty + tile.height, h)
    if cx1 <= cx0 or cy1 <= cy0:
        return tile
    keep = own[cy0:cy1, cx0:cx1]
    if forb is not None:
        keep = np.minimum(keep, 255 - forb[cy0:cy1, cx0:cx1])
    if not keep.size or int(keep.min()) >= 254:
        return tile  # seluruhnya di area yang boleh ditulis — tanpa biaya
    if int(keep.max()) < 128:
        return Image.new("RGBA", tile.size, (0, 0, 0, 0))  # seluruhnya terlarang
    alpha = np.asarray(keep, np.uint8)
    if int(alpha.min()) < int(alpha.max()):
        # Erode 1 px dulu supaya tepi feather tetap berada DI DALAM balon.
        alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8))
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    factor = alpha.astype(np.float32) / 255.0
    sub = tile.crop((cx0 - tx, cy0 - ty, cx1 - tx, cy1 - ty))
    a = np.asarray(sub.getchannel("A"), np.float32) * factor
    sub.putalpha(Image.fromarray(a.astype(np.uint8)))
    out = Image.new("RGBA", tile.size, (0, 0, 0, 0))
    out.alpha_composite(sub, dest=(cx0 - tx, cy0 - ty))
    return out


def _paste_mask(box: tuple[int, int, int, int], mask: np.ndarray,
                h: int, w: int) -> np.ndarray:
    """Mask lokal -> kanvas halaman h x w. Dipotong di tepi halaman."""
    out = np.zeros((h, w), np.uint8)
    x1, y1 = box[0], box[1]
    mh, mw = mask.shape[:2]
    sy1, sx1 = max(y1, 0), max(x1, 0)
    sy2, sx2 = min(y1 + mh, h), min(x1 + mw, w)
    if sy2 > sy1 and sx2 > sx1:
        out[sy1:sy2, sx1:sx2] = mask[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1]
    return out


def _line_bands(box: tuple[int, int, int, int], mask: np.ndarray, size: int,
                lines: list[str], start_y: int,
                font_path: str) -> list[tuple[int, int, int, int]]:
    """Kotak TINTA tiap baris di koordinat halaman: (y0, y1, x0, x1).

    Dihitung analitik dari sumbu blok + _line_width, bukan dengan merender:
    keduanya angka yang SAMA dengan yang dipakai render_region() menempel tile,
    jadi kotak ini benar-benar tempat tintanya jatuh — dan gratis.
    """
    if not lines:
        return []
    font = _font(font_path, size)
    cmap = _cmap(font_path)
    lh = _line_height(font)
    ink_top, ink_bot = _ink_band(font_path, size)
    ax = line_axis(mask, lines, start_y, size, font_path)
    out = []
    for k, line in enumerate(lines):
        w = _line_width(line, font, cmap, size)
        y = box[1] + start_y + k * lh
        out.append((y + ink_top, y + ink_bot + 1,
                    box[0] + int(ax - w / 2), box[0] + int(ax + w / 2) + 1))
    return out


def reclaim_unused_interiors(img: np.ndarray, regions: list[Region],
                             font_path: str | None = None) -> int:
    """Lebar yang diambil disjoin tapi TIDAK dipakai tetangga -> dikembalikan.

    disjoin_overlapping_interiors() menyelesaikan irisan interior secara
    Voronoi — per PIKSEL, tanpa tahu di baris mana teks tetangga benar-benar
    akan jatuh. Hasilnya lebar disandera di ketinggian yang tetangganya bahkan
    tidak sentuh. Terukur di hasilnew/jp_6.JPG (probe_row.py): r3 kehilangan
    20 px tetap di y=139..191, sementara tinta r2 berhenti di y=168 — jadi di
    lima baris terakhir r3 lebar itu hilang tanpa ada yang memakainya. Sisanya
    cuma 26 px dari mask 46x87, sedangkan 'WONDER' butuh 32 px pada size 6, dan
    itulah sebab langsung 'NO WON-/DER'.

    Aturannya, tiap syaratnya menutup satu cacat:

        kandidat_i = fill_mask_i          interior balon SENDIRI sebelum dipangkas
                     & interior region lain  hanya yang DIAMBIL, bukan tepi baru
                     - kotak tinta fase 1    yang benar-benar dipakai tetap milik dia

    Yang boleh mengklaim HANYA region yang benar-benar tercekik: hasil fase 1-nya
    ber-tanda-hubung atau luber. Region yang fontnya kecil tapi rapi tidak
    mengklaim apa pun — "boleh panjang, font mengecil" adalah kebijakan yang
    dipilih, jadi ukuran di bawah plafon bukan cacat dan tidak pantas dibayar
    dengan lebar tetangga. Ini bukan kehati-hatian, ini hasil ukuran: versi
    pertama membiarkan SEMUA region mengklaim, dan di jp_6 r2 (rapi, tanpa tanda
    hubung) mengambil 413 px dari r3 yang justru sedang tercekik — r3 turun 7->6
    dan tanda hubungnya TETAP ada, r2 turun 9->8. Reclaim yang membuat halaman
    lebih buruk daripada tidak dijalankan.

    Pikselnya DIPINDAH, bukan digandakan — dan itu bukan kerapian, itu syarat
    supaya langkah ini berguna sama sekali. render_page() menyusun forb_map dari
    interior region LAIN, jadi kalau piksel yang dikembalikan tetap tercatat
    sebagai interior tetangga, _clip_to_mask() menghapus tepat piksel itu dan
    hasilnya lebih buruk daripada tidak melakukan apa-apa: baris dinilai muat
    lalu tintanya dibuang. Jadi yang menerima menambah, yang melepas mengurangi,
    dalam satu operasi.

    Tiap klaim DIUJI dengan fit() lalu diterima atau DIBATALKAN, bukan dipercaya:
    luas yang bertambah ternyata bukan lebar yang bisa dipakai. Piksel rampasan
    disjoin berbentuk pita Voronoi yang bergerigi, jadi menambahkannya menaikkan
    luas tanpa menaikkan RUN bebas yang menyambung di band satu baris — terukur
    di r3, luas +365 px tapi run band-nya 24->20, 26->24, 26->25 (probe_reclaim3).
    Satu-satunya penilai yang jujur karena itu hasil fit() sesudahnya.

    Syarat terima: pengklaim kehilangan tanda hubung (atau, dengan jumlah tanda
    hubung sama, ukurannya naik), DAN tidak ada pelepas yang mendapat tanda
    hubung baru, luber, atau menyusut lebih dari _RECLAIM_LOSS. Pertukarannya
    searah: satu tanda hubung adalah cacat #4 yang disebut plan.txt, sedangkan
    menyusut sedikit adalah kebijakan yang sudah dipilih ("boleh panjang, font
    mengecil") — jadi tanda hubung boleh ditebus dengan ukuran, tidak pernah
    sebaliknya.

    Sumbernya fill_mask region itu sendiri (direkam build_fill_mask() SEBELUM
    pemangkasan), jadi langkah ini secara konstruksi tidak bisa memunculkan
    "teks keluar bubble": piksel yang dikembalikan selalu piksel yang dulu
    memang interior balon ini. Syarat "& interior region lain" perlu karena
    fill_mask dikikis lebih tipis daripada bubble_mask (fill_erode_stroke) —
    tanpa itu reclaim ikut memakan jarak aman ke garis balon.

    Dua fase, jadi fit() memang berjalan berkali-kali. Itu harganya, dan tidak
    bisa dihindari: kotak tinta tetangga baru diketahui SETELAH ditata, sementara
    lebar yang boleh diambil harus diketahui SEBELUM menata.

    Returns:
        Jumlah region yang interiornya berubah (melebar atau menyusut).
    """
    font_path = font_path or FONT_USED
    if not font_path or len(regions) < 2:
        return 0
    h, w = img.shape[:2]

    live = [r for r in regions if r.translation and not r.is_protected]
    if len(live) < 2:
        return 0
    by_idx = {r.idx: r for r in regions}

    maps = {r.idx: _paste_mask(*_region_box_mask(r), h, w) > 0 for r in regions}
    fills = {
        r.idx: (_paste_mask(r.fill_bbox, r.fill_mask, h, w) > 0
                if r.fill_mask is not None and r.fill_bbox is not None
                else np.zeros((h, w), bool))
        for r in regions
    }
    # Region terlindungi (SFX) tidak pernah jadi pelepas: interiornya ikut
    # menyusun forb_map, jadi merampasnya membuka jalan teks Inggris menimpa SFX
    # — dilarang plan.txt, dan dijaga assert_sfx_intact.
    keep_out = np.zeros((h, w), bool)
    for r in regions:
        if r not in live:
            keep_out |= maps[r.idx]

    def _lay(r: Region, mp: np.ndarray):
        """fit() pada peta halaman `mp`, bukan pada r.bubble_mask yang sekarang."""
        ys, xs = np.nonzero(mp)
        if ys.size == 0:
            return None
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        mask = np.where(mp[box[1]:box[3], box[0]:box[2]], 255, 0).astype(np.uint8)
        text = r.translation.upper() if SETTINGS.force_upper else r.translation
        size, lines, start_y, over = fit(text, mask, region_font_cap(mask), font_path)
        return box, mask, size, lines, start_y, over

    def _score(st) -> tuple[int, int, int]:
        """(tanda hubung, luber, ukuran); dua yang pertama makin kecil makin baik."""
        if st is None:
            return (99, 1, 0)
        _b, _m, size, lines, _sy, over = st
        return (sum(1 for x in lines if x.endswith("-")), int(bool(over)), size)

    # Fase 1: tata letak pada interior sekarang — hanya untuk tahu di mana
    # tintanya jatuh dan siapa yang tercekik. Tata letaknya sendiri dibuang;
    # render_region() menata ulang pada interior final.
    lay = {r.idx: _lay(r, maps[r.idx]) for r in live}

    def _ink_union() -> np.ndarray:
        """Kotak tinta semua region, dari tata letak terakhir yang sah."""
        m = np.zeros((h, w), bool)
        for q in live:
            st = lay[q.idx]
            if st is None:
                continue
            for y0, y1, x0, x1 in _line_bands(st[0], st[1], st[2], st[3],
                                              st[4], font_path):
                m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
        return m

    # Fase 2: satu pengklaim per putaran, diuji lalu diterima atau dibatalkan.
    # Yang paling tercekik jalan lebih dulu (tanda hubung terbanyak, lalu luber,
    # lalu ukuran terkecil) supaya lebar yang terbatas jatuh ke yang paling
    # butuh; idx sebagai pemutus seri terakhir supaya hasilnya deterministik.
    changed: set[int] = set()
    for r in sorted(live, key=lambda q: (-_score(lay[q.idx])[0],
                                         -_score(lay[q.idx])[1],
                                         _score(lay[q.idx])[2], q.idx)):
        base = _score(lay[r.idx])
        if base[0] == 0 and base[1] == 0:
            continue                     # rapi: tidak berhak merampas tetangga
        ink = _ink_union()
        others = np.zeros((h, w), bool)
        for q in regions:
            if q.idx != r.idx:
                others |= maps[q.idx]
        cand = fills[r.idx] & ~maps[r.idx] & others & ~ink & ~keep_out
        if not cand.any():
            continue

        trial = {r.idx: maps[r.idx] | cand}
        losers = [q for q in live
                  if q.idx != r.idx and bool((maps[q.idx] & cand).any())]
        for q in losers:
            trial[q.idx] = maps[q.idx] & ~cand
        newlay = {i: _lay(by_idx[i], m) for i, m in trial.items()}
        if newlay[r.idx] is None or any(newlay[i] is None for i in trial):
            continue                     # jangan pernah mengosongkan balon

        gain = _score(newlay[r.idx])
        better = gain[0] < base[0] or (gain[0] == base[0] and gain[1] < base[1]) \
            or (gain[:2] == base[:2] and gain[2] > base[2])
        if not better:
            continue
        harmed = False
        for q in losers:
            was, now = _score(lay[q.idx]), _score(newlay[q.idx])
            floor = was[2] - max(1, int(round(was[2] * _RECLAIM_LOSS)))
            if now[0] > was[0] or now[1] > was[1] or now[2] < floor:
                harmed = True
                break
        if harmed:
            continue

        for i, m in trial.items():       # diterima: pasang, dan ingat ulang
            maps[i] = m
            lay[i] = newlay[i]
            changed.add(i)

    # Baru sekarang bubble_mask/bubble_bbox ditulis, sekali per region: selama
    # pengujian di atas semuanya masih di `maps` supaya klaim yang ditolak tidak
    # meninggalkan bekas apa pun di Region.
    for i in sorted(changed):
        r = by_idx[i]
        ys, xs = np.nonzero(maps[i])
        bx1, by1 = int(xs.min()), int(ys.min())
        bx2, by2 = int(xs.max()) + 1, int(ys.max()) + 1
        r.bubble_bbox = (bx1, by1, bx2, by2)
        r.bubble_mask = np.where(maps[i][by1:by2, bx1:bx2], 255, 0).astype(np.uint8)
    return len(changed)


def render_page(img: np.ndarray, regions: list[Region]) -> np.ndarray:
    out = img
    h, w = img.shape[:2]
    # Lantai ukuran font berskala lebar halaman — lihat min_font(). Di-set di sini
    # supaya pemanggil render_page() langsung (probe, notebook) ikut benar tanpa
    # perlu mengingat memanggilnya sendiri.
    set_page_width(w)
    n = len(regions)
    if n == 0:
        return img

    # Lebar yang disandera disjoin tanpa dipakai dikembalikan DULU, sebelum peta
    # dan plafon dihitung — semuanya turunan dari bubble_mask.
    reclaim_unused_interiors(img, regions)

    def _page_map(r: Region) -> np.ndarray:
        """Interior balon (atau bbox, untuk region tanpa balon) di halaman.

        Wajib memakai kotak+mask yang SAMA dengan render_region. Kalau keduanya
        berbeda, own_map bisa kosong tepat di tempat teks digambar dan
        _clip_to_mask menghapus seluruh barisnya.
        """
        return _paste_mask(*_region_box_mask(r), h, w)

    # Peta "area terlarang" tiap region = gabungan interior balon region LAIN,
    # dihitung lewat prefix/suffix max supaya biayanya O(N) bukan O(N^2).
    # Klip inilah yang menjamin teks panjang tidak pernah saling timpa, apa pun
    # bentuk balonnya (double bubble menyatu, balon saling tumpang tindih, dll).
    maps = [_page_map(r) for r in regions]
    pref = [np.zeros((h, w), np.uint8)] * (n + 1)
    for i in range(n):
        pref[i + 1] = np.maximum(pref[i], maps[i])
    suff = [np.zeros((h, w), np.uint8)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suff[i] = np.maximum(suff[i + 1], maps[i])

    # Plafon dihitung per balon dari geometrinya (region_font_cap), bukan satu
    # angka halaman: referensi TERUKUR menskalakan teks ke besar balon, bukan
    # menyeragamkannya — lihat komentar di atas _REF_CAP_PER_MIN.
    for i, r in enumerate(regions):
        forb = np.maximum(pref[i], suff[i + 1])
        out = render_region(out, r, own_map=maps[i], forb_map=forb)
    return out

