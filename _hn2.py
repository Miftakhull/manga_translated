#!/usr/bin/env python3
"""hasilnew2 itu keluaran dari INPUT yang mana, dan run-nya menyisakan jejak apa?

Ukuran hasilnew2/6.JPG (680x230) != jp_6.JPG (698x246), jadi belum tentu
halaman yang sama. Sebelum menyalahkan kode, tentukan dulu: (a) file sumber
tiap keluaran, (b) apakah pipeline memperkecil gambar, (c) apakah run hasilnew2
meninggalkan report/log di disk.
"""

from __future__ import annotations

import datetime
import glob
import os
import pathlib

from PIL import Image


def mt(p: str) -> str:
    return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%m-%d %H:%M:%S")


def show(pat: str) -> None:
    hits = sorted(glob.glob(pat), key=lambda p: os.path.getmtime(p))
    if not hits:
        print(f"   (kosong: {pat})")
        return
    for p in hits:
        if os.path.isdir(p):
            print(f"   {mt(p)}  <DIR>                       {p}")
            continue
        sz = os.path.getsize(p)
        dim = ""
        if pathlib.Path(p).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            try:
                with Image.open(p) as im:
                    dim = f"{im.size[0]}x{im.size[1]}"
            except Exception as exc:  # noqa: BLE001
                dim = f"?({exc})"
        print(f"   {mt(p)}  {sz:>9}  {dim:<12} {p}")


def main() -> int:
    for folder in ("hasilnew", "hasilnew2", "contoh", "kerja6", "jp", "input", "src"):
        print(f"\n########## {folder}/ ##########")
        show(f"{folder}/*")

    print("\n########## semua file bernama *6.* / *13.* di root proyek ##########")
    for pat in ("*6.JPG", "*6.jpg", "*6.png", "*13.JPG", "*13.jpg", "*13.png"):
        show(pat)

    print("\n########## log / catatan run terbaru (ERROR/notes/*.log) ##########")
    for pat in ("ERROR*.txt", "*.log", "output/*.log", "notes*.txt", "hasilnew2/*.json",
                "hasilnew2/*.txt", "hasilnew/*.json"):
        show(pat)

    print("\n########## KONSTANTA UKURAN di _nbsrc/config.py ##########")
    cfg = pathlib.Path("_nbsrc/config.py").read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(cfg, 1):
        low = ln.lower()
        if any(k in low for k in ("max_side", "resize", "scale", "long_side",
                                  "max_dim", "upscale", "dpi", "target_w")):
            print(f"   config.py:{i}: {ln.strip()[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
