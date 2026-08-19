#!/usr/bin/env python3
"""Apakah CONTOH/2.webp sejajar dengan halaman kita? Kalau tidak, semua crop
per-balon (probe_ref/probe_margin) mengukur balon SEBELAH dan angkanya palsu.

Sejajaran dicari lewat korelasi fase pada peta tinta, lalu dilaporkan sebagai
geseran (dx, dy) dalam piksel pada skala halaman kita.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / ".stage"
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))

STAGE.mkdir(exist_ok=True)
for _s in sorted((ROOT / "_nbsrc").glob("*.py")):
    _b = _MAGIC.sub("", _s.read_text(encoding="utf-8"), count=1)
    _d = STAGE / _s.name
    if not _d.exists() or _d.read_text(encoding="utf-8") != _b:
        _d.write_text(_b, encoding="utf-8")
sys.path.insert(0, str(STAGE))

import cv2      # noqa: E402
import imgio    # noqa: E402
from PIL import Image  # noqa: E402


def main() -> int:
    img = imgio.load_any(ROOT / "jepang_002.webp")
    h, w = img.shape[:2]
    refp = Image.open(ROOT / "CONTOH" / "2.webp").convert("RGB")
    print(f"kita = {w}x{h}   referensi asli = {refp.size[0]}x{refp.size[1]}  "
          f"rasio = {refp.size[0]/w:.4f} x {refp.size[1]/h:.4f}")
    ref = np.asarray(refp.resize((w, h), Image.LANCZOS), np.uint8)

    # Korelasi fase pada TEPI (garis panel & garis balon), bukan tinta teks:
    # teksnya memang beda bahasa, tapi panel dan balonnya identik.
    def edges(a: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(g, 60, 160).astype(np.float32)

    (dx, dy), resp = cv2.phaseCorrelate(edges(img), edges(ref))
    print(f"geseran global (dx, dy) = ({dx:+.2f}, {dy:+.2f}) px   "
          f"keyakinan = {resp:.3f}")

    # Cek per-kuadran: kalau geserannya beda-beda, skalanya yang salah, bukan
    # cuma offset, dan crop per-balon tetap tidak bisa dipercaya.
    for name, ys, xs in (("kiri-atas", slice(0, h // 2), slice(0, w // 2)),
                         ("kanan-atas", slice(0, h // 2), slice(w // 2, w)),
                         ("kiri-bawah", slice(h // 2, h), slice(0, w // 2)),
                         ("kanan-bawah", slice(h // 2, h), slice(w // 2, w))):
        (qx, qy), qr = cv2.phaseCorrelate(edges(img[ys, xs]), edges(ref[ys, xs]))
        print(f"  {name:<12} ({qx:+6.2f}, {qy:+6.2f})  keyakinan={qr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
