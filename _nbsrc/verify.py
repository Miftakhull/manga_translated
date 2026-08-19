%%writefile /content/mangatl/verify.py

"""Verifikasi residu + escalation ladder.

Jalankan ulang DETECTOR pada halaman bersih — jangan OCR. manga-ocr
terdokumentasi berhalusinasi pada input kosong dan akan terus memberi
false positive.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import SETTINGS, Region
import detect
import erase


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(ua, 1)


# Berapa px interior balon dilebarkan sebelum dipakai MENGURUNG lingkup
# pemeriksaan sisa — lihat _residue_scope(). Bukan nol, karena tinta Jepang yang
# menempel garis balon memang duduk sedikit di luar mask interior dan justru
# itulah tinta yang paling sering tertinggal; bukan besar, karena di luar sana
# ada ART yang ikut terbawa ink_mask dan akan dituduh sisa. Empat ambang diukur
# berdampingan di _erasescope.py pada halaman referensi:
#
#   N     alarm palsu   pengawasan rata2 / terburuk
#   lama  r7, r8        100.0% / 100.0%
#   0     -              97.6% /  81.3%
#   2     -              98.2% /  83.5%
#   3     -              98.3% /  84.4%   <- dipilih
#   4     r7, r8         98.5% /  85.4%
#   8     r7, r8         98.9% /  88.8%
#
# 3 adalah yang TERLEBAR yang masih nol alarm palsu, jadi butanya paling kecil
# di antara yang aman. Di kedua ujung sapuan, sisa BUATAN tetap tertangkap —
# baik di tengah interior (153 px) maupun tepat di tepi garis balon (28 px),
# yang terakhir adalah mekanisme cacat #1 yang sesungguhnya.
_SCOPE_NEAR = 3


def _residue_scope(clean: np.ndarray, region: Region) -> tuple[np.ndarray, np.ndarray] | None:
    """(dev, scope) untuk region ini: piksel menyimpang, dan di mana dicari.

    Lingkupnya `(bekas stroke | cincin 3 px di sekitarnya)` DIKURUNG ke interior
    balon yang dilebarkan `_SCOPE_NEAR` px.

    Cincin itu ada karena `textmask.protect_bubble_outline()` MENGHAPUS piksel
    dari ink_mask: tinta Jepang yang menempel garis balon memang tidak boleh
    dicat, kalau tidak garis balonnya ikut hilang. Konsekuensinya piksel itu
    (a) tidak dicat jalur ink_mask dan (b) tidak pernah diperiksa, karena
    lingkup pemeriksaan justru `ink_mask > 0` — dan piksel itu baru saja dibuang
    dari sana. Selama fill_mask ada, isian interior menutupinya; begitu
    build_fill_mask menyerah, ia tertinggal di halaman dan definisi lama
    melaporkan NOL. Terukur di halaman referensi (est_font_size TERISI seperti
    produksi): protect_bubble_outline melepas 215 px.

    Kurungan interiornya yang menahan ALARM PALSU, dan ini yang terukur paling
    mahal. Tanpa kurungan, r7 (36 px) dan r8 (103 px) ditandai — padahal
    _erasewho.py mengukur bahwa piksel itu (a) sama sekali TIDAK diubah erase
    (utuh = sisa, jadi bukan bekas cat yang gagal), (b) NOL px-nya di dalam
    bubble_mask, (c) nol di bawah guard, dan (d) gray minimumnya 113/4 dengan
    median 184/190 — itu ART di luar balon yang ikut terbawa ink_mask. Kedua
    jawaban atas tuduhan itu merusak halaman:
      * mengecatnya (gabungan fill|ink di erase_flat) menaruh bg_color putih di
        atas art gray 12-18 — 7 bercak TERLIHAT terukur di _eraseblotch.py,
        yaitu cacat #3 lewat pintu belakang, di region yang bahkan tidak punya
        sisa untuk diperbaiki;
      * membiarkannya membuat find_residue memanggil escalate(), dan mask
        eskalasinya memakan 250 px `bubble_outline_guard` (_erasegate.py) —
        persis yang protect_bubble_outline ada untuk mencegah.
    Jadi yang salah bukan cat maupun eskalasinya, melainkan LINGKUP tuduhannya.

    Kenapa cincin 3 px dan bukan seluruh interior: sudut kotak balon berisi
    ART, dan sejak `textmask._keep_ink_lobes()` art itu memang sengaja tidak
    dicat. Lingkup seluruh interior akan melaporkan art sebagai sisa lalu
    mengeskalasi inpaint ke atasnya. Terukur: r9 menyimpan 198 px art di sudut
    kanan-bawah interiornya (halaman y1280-1303 x608-625, gray 218-234);
    lingkup ini TIDAK melihatnya, karena 3 px dari bekas tinta tidak menjangkau
    sudut kotak.

    Cincin `near` tetap digabung, bukan diganti: lingkup cincin-saja kehilangan
    bekas stroke yang justru sedang diperiksa. `gabungan >= lama` terukur di
    seluruh 13 region — pada halaman referensi keduanya sama besar, jadi cincin
    ini bermotif struktural (ia menutup lubang pengawasan di atas) dan belum
    punya contoh positif di halaman ini.
    """
    if region.ink_mask is None:
        return None
    x1, y1, x2, y2 = region.bbox
    crop = clean[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    mh, mw = region.ink_mask.shape[:2]
    sub = region.ink_mask[: min(mh, gray.shape[0]), : min(mw, gray.shape[1])]
    area = gray[: sub.shape[0], : sub.shape[1]]
    inside = sub > 0
    if not inside.any():
        return None
    scope = inside
    if region.bubble_mask is not None and region.bubble_bbox is not None:
        big = np.zeros(clean.shape[:2], np.uint8)
        bx1, by1 = region.bubble_bbox[0], region.bubble_bbox[1]
        bh, bw = region.bubble_mask.shape[:2]
        yy = min(by1 + bh, clean.shape[0])
        xx = min(bx1 + bw, clean.shape[1])
        big[by1:yy, bx1:xx] = region.bubble_mask[: yy - by1, : xx - bx1]
        if big.any():
            itr = big[y1 : y1 + sub.shape[0], x1 : x1 + sub.shape[1]] > 0
            near = cv2.dilate(
                sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
            wide = cv2.dilate(big, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * _SCOPE_NEAR + 1,) * 2))
            keep = wide[y1 : y1 + sub.shape[0], x1 : x1 + sub.shape[1]] > 0
            scope = (inside | (near & itr[: sub.shape[0], : sub.shape[1]])) & keep
    bg = float(np.median(area[~inside])) if (~inside).any() else 255.0
    dev = np.abs(area.astype(np.int16) - bg) > SETTINGS.residue_deviation
    return dev, scope


def pixel_residue(clean: np.ndarray, region: Region) -> int:
    """Hitung piksel yang masih menyimpang dari median background lokal.

    Non-nol di region flat-fill berarti mask atau fill-nya salah.
    Lingkupnya dijelaskan di _residue_scope().
    """
    got = _residue_scope(clean, region)
    if got is None:
        return 0
    dev, scope = got
    return int((dev & scope).sum())


def residue_blob(clean: np.ndarray, region: Region) -> int:
    """Komponen tersambung TERBESAR dari sisa — bukan jumlahnya.

    Gerbang jumlah `max(30, 0.002*w*h)` berskala AREA balon, sementara satu
    titik kotor tidak. Di balon 121x199 halaman ini ambangnya 48 px, tapi di
    balon 400x500 ia menjadi 400 px — dan titik 60 px yang jelas terlihat lolos
    utuh tanpa satu ronde eskalasi. Yang dilihat mata adalah satu titik, jadi
    yang harus dijaga ukuran titik terbesarnya, bukan totalnya.
    """
    got = _residue_scope(clean, region)
    if got is None:
        return 0
    dev, scope = got
    hit = dev & scope
    if not hit.any():
        return 0
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(hit.astype(np.uint8), 8)
    return 0 if n <= 1 else int(stats[1:, cv2.CC_STAT_AREA].max())


def find_residue(clean: np.ndarray, regions: list[Region]) -> list[Region]:
    """Region mana yang masih punya sisa teks setelah erase.

    Deteksi di luar region yang dibersihkan = SFX yang sengaja dijaga, abaikan.

    Region PROTECTED sengaja tetap dikecualikan: tintanya memang harus utuh, dan
    memeriksanya berarti mengeskalasi inpaint ke atas SFX — pelanggaran kontrak
    'SFX dan simbol tetap ada'. Yang diperiksa hanya route flat/lama.
    """
    erased = [r for r in regions if r.route in ("flat", "lama") and not r.is_protected]
    if not erased:
        return []

    try:
        new_regions, _ = detect.detect(clean, conf=max(SETTINGS.det_conf, 0.35))
    except (RuntimeError, FileNotFoundError):
        new_regions = []

    failed: list[Region] = []
    for r in erased:
        hit = any(_iou(r.bbox, nr.bbox) > 0.25 for nr in new_regions)
        # Dua gerbang, bukan satu: TOTAL berskala area balon (satu titik kotor
        # tidak), jadi ditambah gerbang komponen terbesar — lihat residue_blob().
        if (hit
                or pixel_residue(clean, r) > max(30, int(0.002 * r.width * r.height))
                or residue_blob(clean, r) > SETTINGS.residue_blob_max):
            failed.append(r)
    return failed


def assert_sfx_intact(erase_mask: np.ndarray, protected_mask: np.ndarray) -> bool:
    """Kontrak keras: mask hapus tidak boleh menyentuh satu piksel SFX pun."""
    if protected_mask.max() == 0:
        return True
    return not bool(cv2.bitwise_and(erase_mask, protected_mask).any())


def escalate(
    img: np.ndarray, clean: np.ndarray, failed: list[Region], device: str = "cuda"
) -> np.ndarray:
    """Perluas mask region yang gagal lalu inpaint ulang HANYA region itu.

    Ladder: kernel 3 -> 5 -> 7 mengikuti percobaan ke-berapa.

    Mask yang sudah dipekarkan DIKURUNG ke interior balon (dilebarkan
    `_SCOPE_NEAR` px, ambang yang sama dengan lingkup pemeriksaan). Tanpa
    kurungan itu, memekarkan ink_mask dengan kernel 5 memakan 250 px
    `bubble_outline_guard` di halaman referensi (terukur di _erasegate.py) —
    LaMa lalu melukis ulang garis balon yang `protect_bubble_outline` baru saja
    susah-susah selamatkan, dan hasilnya balon bergaris putus. Kurungannya tidak
    membuat eskalasi tumpul: sisa BUATAN di dalam interior tetap tertutup
    7727/7727 px dengan maupun tanpa kurungan. Region tanpa bubble_mask
    dibiarkan seperti dulu — tidak ada interior untuk dijadikan pagar.
    """
    if not failed:
        return clean
    out = clean
    for attempt in range(1, SETTINGS.max_escalation + 1):
        mask = np.zeros(img.shape[:2], np.uint8)
        k = 3 + 2 * attempt
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        for r in failed:
            if r.ink_mask is None:
                continue
            x1, y1, x2, y2 = r.bbox
            mh, mw = r.ink_mask.shape[:2]
            y2, x2 = min(y2, y1 + mh), min(x2, x1 + mw)
            grown = cv2.dilate(r.ink_mask[: y2 - y1, : x2 - x1], el, iterations=1)
            if r.bubble_mask is not None and r.bubble_bbox is not None:
                itr = np.zeros(img.shape[:2], np.uint8)
                bx1, by1 = r.bubble_bbox[0], r.bubble_bbox[1]
                bh, bw = r.bubble_mask.shape[:2]
                yy = min(by1 + bh, img.shape[0])
                xx = min(bx1 + bw, img.shape[1])
                itr[by1:yy, bx1:xx] = r.bubble_mask[: yy - by1, : xx - bx1]
                if itr.any():
                    wide = cv2.dilate(itr, cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (2 * _SCOPE_NEAR + 1,) * 2))
                    grown = cv2.bitwise_and(grown, wide[y1:y2, x1:x2])
            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], grown)

        if not mask.any():
            break
        out = erase.erase_neural(out, mask, device)
        failed = find_residue(out, failed)
        if not failed:
            break
    return out


def report(regions: list[Region], failed: list[Region], font_used: str,
           notes: list[tuple[str, str, str]] | None = None) -> dict:
    """Ringkasan untuk report.json + tabel UI.

    `notes` = catatan config.note() yang muncul SELAMA halaman ini diproses
    (lihat config.notes_since). Ikut di sini, bukan cuma di batch, karena
    pertanyaan yang ditanyakan pembaca selalu per halaman: "halaman ini kenapa
    kosong?". Kalau catatan hanya ada di tingkat batch, halaman ke-3 mewarisi
    tuduhan halaman ke-1.
    """
    # untranslated_idx: region yang boleh diterjemahkan (bukan SFX) tapi tidak
    # dapat terjemahan. Ini yang menangkap cacat hasilnew/13.JPG di sidecar:
    # translated_count saja tidak cukup karena angkanya harus dibandingkan
    # dengan jumlah region yang MEMANG perlu diterjemahkan — dan pembacanya
    # tidak punya angka itu. Kalau daftar ini tidak kosong, ada balon yang
    # tercetak berbahasa Jepang.
    untranslated = [r.idx for r in regions
                    if not r.is_protected and r.src_text and not r.translation]
    notes = list(notes or [])
    return {
        "region_count": len(regions),
        "residue_count": len(failed),
        "residue_idx": [r.idx for r in failed],
        "overflow_count": sum(1 for r in regions if r.overflowed),
        "protected_count": sum(1 for r in regions if r.is_protected),
        "sfx_idx": [r.idx for r in regions if r.label == "SFX"],
        "translated_count": sum(1 for r in regions if r.translation),
        "untranslated_count": len(untranslated),
        "untranslated_idx": untranslated,
        # translatable_count: pembanding yang hilang selama ini. region_count
        # memuat SFX dan UNREADABLE yang memang TIDAK boleh diterjemahkan, jadi
        # "translated 0 dari 15 region" bisa berarti dua hal yang jauh berbeda:
        # 15 balon gagal, atau 15 region itu semuanya SFX. Tanpa angka ini
        # pembaca tabel tidak bisa membedakannya.
        "translatable_count": sum(1 for r in regions
                                  if not r.is_protected and r.src_text),
        "notes": [{"level": lv, "tag": tg, "msg": ms} for lv, tg, ms in notes],
        "error_count": sum(1 for lv, _t, _m in notes if lv == "error"),
        "warn_count": sum(1 for lv, _t, _m in notes if lv == "warn"),
        "route_flat": sum(1 for r in regions if r.route == "flat"),
        "route_lama": sum(1 for r in regions if r.route == "lama"),
        "font_used": font_used,
        "regions": [r.to_dict() for r in regions],
    }

