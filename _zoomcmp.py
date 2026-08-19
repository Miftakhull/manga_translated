#!/usr/bin/env python3
"""Zoom hasilnew2/6.JPG dan 13.JPG jadi potongan besar, plus pembanding dari run
BERSIH 16 Agu, supaya 'coretan' dan balon yang belum diterjemah bisa DILIHAT
posisinya — bukan ditebak.

hasilnew2 dicrop manual oleh user, jadi ukurannya beda; disamakan lewat lebar.
"""

from __future__ import annotations

import pathlib

from PIL import Image

OUT = pathlib.Path("_dbg")


def strips(src: str, tag: str, n: int = 3, scale: float = 3.0) -> None:
    im = Image.open(src).convert("RGB")
    w, h = im.size
    big = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    W, H = big.size
    step = W // n
    for i in range(n):
        x0 = i * step
        x1 = W if i == n - 1 else (i + 1) * step
        p = OUT / f"z_{tag}_{i}.png"
        big.crop((x0, 0, x1, H)).save(p)
        print(f"   {p}  {x1 - x0}x{H}")


def main() -> int:
    print("=== hasilnew2/6.JPG (HASIL SEKARANG, dilaporkan kotor)")
    strips("hasilnew2/6.JPG", "now6", 3, 3.0)
    print("=== kerja6/debug_jp_6/10_typeset.png (run BERSIH 16 Agu 12:43)")
    strips("kerja6/debug_jp_6/10_typeset.png", "ok6", 3, 3.0)
    print("=== hasilnew2/13.JPG (HASIL SEKARANG, ada yang belum diterjemah)")
    strips("hasilnew2/13.JPG", "now13", 2, 3.0)
    print("=== debug/jp_13/10_typeset.png (run BERSIH 16 Agu 12:47)")
    strips("debug/jp_13/10_typeset.png", "ok13", 2, 3.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
