#!/usr/bin/env python3
"""Cari ASAL paruh ATAS _cmp_jp_6.png / _cmp_jp_13.png.

Pertanyaannya satu dan menentukan: paruh atas itu HASIL PROGRAM kita 15/16 Agu,
atau REFERENSI buatan manusia (contoh/*.webp, kerja6/ref_6.JPG)? Kalau referensi,
tidak ada yang regresi; kalau hasil program, ada kode yang hilang.

Dibandingkan lewat ukuran + selisih piksel setelah disamakan ukurannya, bukan
lewat nama file, karena namanya memang tidak menyebut asalnya.
"""

from __future__ import annotations

import glob
import pathlib

import numpy as np
from PIL import Image


def half(p: str, top: bool = True) -> Image.Image:
    im = Image.open(p).convert("RGB")
    a = np.asarray(im)
    red = ((a[:, :, 0] > 150) & (a[:, :, 1] < 80) & (a[:, :, 2] < 80)).mean(axis=1)
    rows = np.where(red > 0.5)[0]
    if not len(rows):
        mid = im.height // 2
        return im.crop((0, 0, im.width, mid)) if top else im.crop((0, mid, im.width, im.height))
    return (im.crop((0, 0, im.width, int(rows.min())))
            if top else im.crop((0, int(rows.max()) + 1, im.width, im.height)))


def score(a: Image.Image, b: Image.Image) -> float:
    """MAE grayscale setelah b diskalakan ke ukuran a. 0 = identik."""
    bb = b.convert("L").resize(a.size, Image.LANCZOS)
    return float(np.abs(np.asarray(a.convert("L"), float)
                        - np.asarray(bb, float)).mean())


def main() -> int:
    cands = sorted(set(
        glob.glob("contoh/*") + glob.glob("kerja6/*.JPG") + glob.glob("kerja6/*.jpg")
        + glob.glob("hasilnew/*.JPG") + glob.glob("hasilnew2/*.JPG")
        + glob.glob("output/*.JPG") + glob.glob("output/*.png")
        + glob.glob("_cmp?_jp_*.png")
    ))
    cands = [c for c in cands
             if pathlib.Path(c).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]

    for tag in ("6", "13"):
        cmp_file = f"_cmp_jp_{tag}.png"
        if not pathlib.Path(cmp_file).exists():
            print(f"{cmp_file} TIDAK ADA")
            continue
        top, bot = half(cmp_file, True), half(cmp_file, False)
        print(f"===== {cmp_file}  ATAS={top.size}  BAWAH={bot.size}")
        rows = []
        for c in cands:
            if tag not in pathlib.Path(c).stem:
                continue
            try:
                im = Image.open(c)
            except Exception:  # noqa: BLE001
                continue
            rows.append((score(top, im), score(bot, im), im.size, c))
        for st, sb, size, c in sorted(rows)[:10]:
            flag = "  <== SAMA DENGAN ATAS" if st < 6 else ""
            print(f"   MAE_atas={st:7.2f}  MAE_bawah={sb:7.2f}  {str(size):<12} {c}{flag}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
