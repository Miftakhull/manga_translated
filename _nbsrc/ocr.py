%%writefile /content/mangatl/ocr.py

"""OCR Jepang via manga-ocr, dengan gate anti-halusinasi.

manga-ocr terdokumentasi menghasilkan teks acak pada input kosong. Gate
ink_ratio wajib jalan sebelum model dipanggil, kalau tidak halaman bersih
akan penuh terjemahan hantu.
"""

from __future__ import annotations

import re

import cv2
import numpy as np
from PIL import Image

from config import SETTINGS, Region, note

_OCR = None
_OCR_FAILED = False

# Halusinasi khas manga-ocr pada input kosong/hampir kosong.
_HALLUCINATION = frozenset({
    "。", "、", "…", "・", "！", "？", "「", "」", "ー", "～", ".", "..", "...",
})
_REPEAT = re.compile(r"^(.)\1{4,}$")


def get_ocr():
    """Muat manga-ocr sekali. None kalau paket tidak terpasang.

    Kegagalan di sini dicetak, tidak ditelan: kalau OCR mati, SETIAP region jadi
    UNREADABLE lalu ikut PROTECTED_LABELS, dan halaman keluar tanpa satu pun
    terjemahan — gejalanya terlihat seperti "model LLM tidak jalan", padahal
    penyebabnya cuma satu dependency hilang (`loguru`, di-skip oleh --no-deps).
    """
    global _OCR, _OCR_FAILED
    if _OCR is not None or _OCR_FAILED:
        return _OCR
    try:
        from manga_ocr import MangaOcr

        # force_cpu=False (default) = otomatis CUDA kalau GPU ada. MangaOcr 0.1.16
        # TIDAK punya parameter `device=` — jangan diganti, nanti TypeError.
        _OCR = MangaOcr(force_cpu=False)
    except (ImportError, OSError, RuntimeError) as exc:
        note("error", "ocr",
             f"manga-ocr tidak bisa dimuat ({exc}) — SEMUA region jadi UNREADABLE, "
             "artinya tidak ada teks yang dihapus maupun diterjemahkan")
        _OCR_FAILED = True
        _OCR = None
    return _OCR


def _prepare(img: np.ndarray, region: Region) -> Image.Image:
    """Crop + upscale kecil supaya glyph tipis tidak hilang saat resize model."""
    x1, y1, x2, y2 = region.bbox
    crop = img[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    if min(h, w) < 32 and min(h, w) > 0:
        scale = 32 / min(h, w)
        crop = cv2.resize(
            crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )
    # Padding putih membantu model: teks manga selalu punya margin.
    crop = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return Image.fromarray(crop)


def _is_hallucination(text: str) -> bool:
    t = text.strip()
    if not t or t in _HALLUCINATION:
        return True
    return bool(_REPEAT.match(t))


def read_region(img: np.ndarray, region: Region) -> str:
    """OCR satu region. String kosong = tidak ada teks terbaca."""
    if region.ink_ratio < SETTINGS.min_ink_ratio:
        return ""  # gate: terlalu sedikit tinta, model akan berhalusinasi
    ocr = get_ocr()
    if ocr is None:
        return ""
    try:
        text = ocr(_prepare(img, region))
    except (RuntimeError, ValueError, OSError):
        return ""
    text = (text or "").strip()
    return "" if _is_hallucination(text) else text


def read_all(img: np.ndarray, regions: list[Region]) -> list[Region]:
    """Isi src_text tiap region. Region tanpa teks ditandai UNREADABLE.

    UNREADABLE masuk PROTECTED_LABELS, jadi region gagal-baca tidak pernah
    dihapus dari halaman — lebih baik teks asli tertinggal daripada art hilang.
    """
    for r in regions:
        r.src_text = read_region(img, r)
        if not r.src_text:
            r.label = "UNREADABLE"
    return regions


def release() -> None:
    """Bebaskan ~444 MB model OCR."""
    global _OCR
    _OCR = None

