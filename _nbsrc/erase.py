%%writefile /content/mangatl/erase.py

"""Routing erase: flat-fill cepat vs neural inpaint LaMa.

70-85% region bisa cukup flat-fill — nol GPU, nol residu secara konstruksi.
Sisanya baru masuk LaMa atau cv2.inpaint.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import SETTINGS, Region
import inpaint as inp
# textmask tidak mengimpor erase, jadi arah impor ini tidak melingkar.
import textmask as tm


def _fill_on_page(region: Region, shape: tuple[int, int]) -> np.ndarray | None:
    """fill_mask region di koordinat halaman, atau None kalau tidak ada."""
    if region.fill_mask is None or region.fill_bbox is None:
        return None
    h, w = shape
    x1, y1, x2, y2 = region.fill_bbox
    sx1, sy1 = max(x1, 0), max(y1, 0)
    mh, mw = region.fill_mask.shape[:2]
    sx2, sy2 = min(x2, x1 + mw, w), min(y2, y1 + mh, h)
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    out = np.zeros((h, w), np.uint8)
    out[sy1:sy2, sx1:sx2] = region.fill_mask[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1]
    return out


def fill_color(img: np.ndarray, region: Region) -> tuple[int, int, int] | None:
    """Warna isian interior balon: median piksel interior yang BUKAN tinta.

    Bukan putih tetap. Balon hitam (teks putih di atas hitam) memang ada di
    manga, dan mengisinya putih akan merusak halaman lebih parah daripada sisa
    tinta yang mau dihilangkan. Median interior-minus-tinta mengembalikan hitam
    untuk balon hitam dan putih untuk balon putih, tanpa cabang khusus.

    Tinta dikeluarkan dari perhitungan lewat ink_mask yang didilatasi: kalau
    tidak, glyph ikut menarik median ke arah warna tinta dan isian jadi kelabu.
    """
    m = _fill_on_page(region, img.shape[:2])
    if m is None:
        return None
    inside = m > 0
    if not inside.any():
        return None
    ink = np.zeros(img.shape[:2], np.uint8)
    if region.ink_mask is not None:
        x1, y1, x2, y2 = region.bbox
        mh, mw = region.ink_mask.shape[:2]
        y2, x2 = min(y2, y1 + mh, img.shape[0]), min(x2, x1 + mw, img.shape[1])
        if y2 > y1 and x2 > x1:
            ink[y1:y2, x1:x2] = region.ink_mask[: y2 - y1, : x2 - x1]
        ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    bg_px = img[inside & (ink == 0)]
    if bg_px.size < 30:
        bg_px = img[inside]
    return tuple(int(v) for v in np.median(bg_px.reshape(-1, 3), axis=0))


def _bg_stats(img: np.ndarray, region: Region) -> tuple[np.ndarray, float]:
    """Median warna + sebaran background. Sebaran rendah -> flat-fill.

    Diukur pada PITA TIPIS di sekeliling tinta, bukan seluruh kotak region.
    Kotak region sudah dilebarkan 6 px saat mask dibangun, jadi untuk balon yang
    pas ia ikut memakan garis luar balon yang hitam — dan itu yang membuat
    seluruh 13 region halaman uji lari ke LaMa, termasuk balon yang isinya putih
    bersih. LaMa lalu mengarang bercak kelabu di tempat yang flat-fill akan
    selesaikan sempurna. Pita di sekeliling glyph adalah tetangga yang benar-benar
    harus ditiru oleh isian.

    Sebaran dihitung dari MAD, bukan np.std maupun rentang persentil. Std tidak
    robust sama sekali, dan persentil 5-95 cuma tahan 5% outlier — pada balon yang
    teksnya mepet, pita ikut menyeberangi garis luar balon lebih dari itu, jadi
    tiga balon berlatar putih murni masih terbaca 999 dan lari ke LaMa. MAD tahan
    sampai 50% outlier: garis balon adalah BATAS, bukan tekstur, dan mayoritas
    pita tetap putih rata. Screentone tetap tertangkap karena di sana sebaran
    itulah mayoritasnya.
    """
    x1, y1, x2, y2 = region.bbox
    crop = img[y1:y2, x1:x2]
    if region.ink_mask is None or crop.size == 0:
        return np.array([255, 255, 255], dtype=np.uint8), 0.0

    ink = region.ink_mask[: crop.shape[0], : crop.shape[1]]
    ring = cv2.subtract(
        cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))), ink
    )
    bg_px = crop[ring > 0]
    if bg_px.size < 30:
        bg_px = crop[ink == 0]
    if bg_px.size < 10:
        return np.array([255, 255, 255], dtype=np.uint8), 0.0

    median = np.median(bg_px, axis=0).astype(np.uint8)
    dev = np.abs(bg_px.astype(np.float32) - median.astype(np.float32))
    spread_per_channel = np.median(dev, axis=0) * 1.4826  # MAD -> skala sigma
    noise = np.std(spread_per_channel)
    thresh = SETTINGS.flat_std_thresh_noisy if noise > 1 else SETTINGS.flat_std_thresh
    max_spread = float(spread_per_channel.max())

    return median, max_spread if max_spread < thresh else 999.0


def route_region(img: np.ndarray, region: Region) -> Region:
    """Putuskan flat vs lama. Region skip langsung return."""
    if region.is_protected or region.ink_mask is None:
        region.route = "skip"
        return region

    # Jalur isian-interior: warnanya diambil dari interior balon itu sendiri,
    # jadi tidak perlu ronde LaMa sama sekali — hasilnya rata sempurna secara
    # konstruksi dan mustahil menyisakan satu titik pun.
    fc = fill_color(img, region) if SETTINGS.bubble_fill else None
    if fc is not None:
        region.bg_color = fc
        region.route = "flat"
        return region

    median, std = _bg_stats(img, region)
    region.bg_color = tuple(int(v) for v in median)

    if std < SETTINGS.flat_std_thresh:
        region.route = "flat"
    else:
        region.route = "lama"
    return region


def erase_flat(img: np.ndarray, region: Region,
               guard: np.ndarray | None = None) -> np.ndarray:
    """Isi region dengan median background — instan, nol GPU.

    Kalau fill_mask tersedia (balon yang dikenali), yang diisi adalah SELURUH
    interior balon, bukan cuma stroke glyph — lihat textmask.build_fill_mask().
    `guard` adalah piksel yang tidak boleh disentuh isian (SFX + garis balon).
    """
    if region.bg_color is None:
        return img
    fill = _fill_on_page(region, img.shape[:2]) if SETTINGS.bubble_fill else None
    if fill is not None:
        sel = fill > 0
        if guard is not None:
            sel &= guard == 0
        img[sel] = region.bg_color
        return img
    if region.ink_mask is None:
        return img
    x1, y1, x2, y2 = region.bbox
    crop = img[y1:y2, x1:x2].copy()
    mh, mw = region.ink_mask.shape[:2]
    y2, x2 = min(y2, y1 + mh), min(x2, x1 + mw)
    sub_mask = region.ink_mask[: y2 - y1, : x2 - x1]
    m3 = (sub_mask > 0)[:, :, None]
    crop[: y2 - y1, : x2 - x1][m3[:, :, 0]] = region.bg_color
    img[y1:y2, x1:x2] = crop[: y2 - y1, : x2 - x1]
    return img



def erase_neural(img: np.ndarray, mask: np.ndarray, device: str = "cuda") -> np.ndarray:
    """LaMa skala halaman. mask biner 0/255 koordinat halaman."""
    if mask.max() == 0:
        return img
    return inp.inpaint(img, mask, device)


def protected_guard(img: np.ndarray, regions: list[Region]) -> np.ndarray:
    """Piksel yang tidak boleh disentuh isian: tinta SFX + garis balon.

    Wajib ada begitu erase mengisi INTERIOR PENUH. Mask stroke tidak pernah
    menyentuh SFX yang duduk di dalam balon karena bentuknya mengikuti glyph
    dialog; isian interior penuh akan menimpanya. Dua hal yang dijaga:

    * tinta region terlindungi (SFX/UNREADABLE), dilebarkan 5 px — kontrak
      pipeline paling keras: 'SFX DAN SYMBOL SYMBOL TETAP ADA',
    * garis balon itu sendiri (textmask.bubble_outline_guard) — interior sudah
      dikikis sekali, tapi pada balon berstroke tebal kikisan itu bisa kurang,
      dan garis balon yang termakan isian terlihat sebagai balon bocor.
    """
    h, w = img.shape[:2]
    guard = np.zeros((h, w), np.uint8)
    for r in regions:
        if not r.is_protected or r.ink_mask is None:
            continue
        x1, y1, x2, y2 = r.bbox
        mh, mw = r.ink_mask.shape[:2]
        y2, x2 = min(y2, y1 + mh, h), min(x2, x1 + mw, w)
        if y2 > y1 and x2 > x1:
            guard[y1:y2, x1:x2] = np.maximum(
                guard[y1:y2, x1:x2], r.ink_mask[: y2 - y1, : x2 - x1])
    if guard.any():
        guard = cv2.dilate(
            guard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.bitwise_or(guard, tm.bubble_outline_guard(img, regions))


def erase_page(img: np.ndarray, regions: list[Region], device: str = "cuda") -> np.ndarray:
    """Pipeline erase gabungan flat + neural, sesuai routing."""
    out = img.copy()
    lama_mask = np.zeros(img.shape[:2], np.uint8)
    guard = protected_guard(img, regions) if SETTINGS.bubble_fill else None

    for r in regions:
        route_region(out, r)
        if r.route == "flat":
            out = erase_flat(out, r, guard)
        elif r.route == "lama" and r.ink_mask is not None:
            x1, y1, x2, y2 = r.bbox
            mh, mw = r.ink_mask.shape[:2]
            y2, x2 = min(y2, y1 + mh), min(x2, x1 + mw)
            sub = r.ink_mask[: y2 - y1, : x2 - x1]
            lama_mask[y1:y2, x1:x2] = np.maximum(lama_mask[y1:y2, x1:x2], sub)

    if lama_mask.any():
        out = erase_neural(out, lama_mask, device)

    return out

