%%writefile /content/mangatl/imgio.py

"""Baca gambar apa pun jadi RGB, tulis hasil sebagai PNG + JPG + ZIP."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from config import OUTPUT, SETTINGS, SUPPORTED_EXT

Image.MAX_IMAGE_PIXELS = 300_000_000  # halaman manga resolusi tinggi itu wajar


def register_extra_formats() -> list[str]:
    """Daftarkan HEIC/AVIF kalau plugin-nya ada. Aman dipanggil berulang."""
    enabled: list[str] = []
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        enabled.append("heic")
    except (ImportError, AttributeError):
        pass
    # AVIF sudah native di Pillow >= 11; pillow-heif membuang
    # register_avif_opener, memanggilnya akan AttributeError.
    if "AVIF" in Image.registered_extensions().values():
        enabled.append("avif")
    return enabled


def load_any(path: str | Path) -> np.ndarray:
    """Buka file gambar apa pun -> array RGB uint8, HxWx3.

    Urutan normalisasi penting: exif_transpose dulu (kalau tidak, halaman
    ter-rotasi salah), baru urusan mode warna.
    """
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXT:
        raise ValueError(f"Ekstensi tidak didukung: {path.suffix}")

    with Image.open(path) as im:
        im.load()
        return normalize(im)


def normalize(im: Image.Image) -> np.ndarray:
    """Samakan orientasi, bit depth, dan mode warna jadi RGB uint8."""
    im = ImageOps.exif_transpose(im) or im

    # Palette dengan transparansi harus lewat RGBA dulu supaya alpha tidak hilang.
    if im.mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")

    # 16-bit / float perlu diskalakan sebelum convert, kalau tidak akan terpotong.
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.asarray(im).astype(np.float32)
        hi = float(arr.max()) or 1.0
        arr = (arr / hi * 255.0).clip(0, 255).astype(np.uint8)
        im = Image.fromarray(arr, mode="L")

    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg

    if im.mode != "RGB":
        im = im.convert("RGB")

    return np.asarray(im, dtype=np.uint8)


def save_outputs(img: np.ndarray, stem: str, outdir: Path | None = None) -> dict[str, Path]:
    """Simpan sesuai SETTINGS.output_format: PNG, JPG, atau keduanya.

    JPG wajib subsampling=0 (4:4:4) — chroma subsampling default bikin garis
    hitam tajam di manga jadi berbayang. Dict hasil hanya berisi format yang
    benar-benar ditulis, jadi konsumen (gallery/zip) tinggal pakai apa adanya.
    """
    outdir = outdir or OUTPUT
    outdir.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray(img)

    # Clamp ke nilai sah: format tak dikenal tetap harus menghasilkan file,
    # kalau tidak pipeline mati di sidecar JSON (next(iter(paths.values()))).
    fmt = SETTINGS.output_format if SETTINGS.output_format in ("png", "jpg", "both") else "both"
    paths: dict[str, Path] = {}
    if fmt in ("png", "both"):
        png_path = outdir / f"{stem}.png"
        pil.save(png_path, "PNG", optimize=True)
        paths["png"] = png_path
    if fmt in ("jpg", "both"):
        jpg_path = outdir / f"{stem}.jpg"
        pil.save(jpg_path, "JPEG", quality=SETTINGS.jpg_quality, subsampling=0)
        paths["jpg"] = jpg_path
    return paths


def make_zip(paths: list[Path], zip_path: Path) -> Path:
    """Bungkus hasil jadi satu ZIP supaya sekali unduh."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p.exists():
                zf.write(p, arcname=p.name)
    return zip_path


def to_png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    return buf.getvalue()

