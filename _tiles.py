#!/usr/bin/env python3
"""Petak zoom besar hasilnew2/6.JPG saja — untuk MELIHAT coretan yang tersisa.

Tidak menyentuh kode pipeline apa pun. Murni memotong gambar hasil.
"""

from __future__ import annotations

import pathlib

from PIL import Image

OUT = pathlib.Path("_dbg")


def tiles(src: str, tag: str, cols: int, rows: int, scale: float) -> None:
    im = Image.open(src).convert("RGB")
    w, h = im.size
    big = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    W, H = big.size
    cw, ch = W // cols, H // rows
    for ry in range(rows):
        for rx in range(cols):
            x0, y0 = rx * cw, ry * ch
            x1 = W if rx == cols - 1 else x0 + cw
            y1 = H if ry == rows - 1 else y0 + ch
            p = OUT / f"t_{tag}_{ry}{rx}.png"
            big.crop((x0, y0, x1, y1)).save(p)
            print(f"   {p}  {x1 - x0}x{y1 - y0}")


def main() -> int:
    print("=== hasilnew2/6.JPG @5x, 4 kolom x 2 baris")
    tiles("hasilnew2/6.JPG", "n6", 4, 2, 5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
