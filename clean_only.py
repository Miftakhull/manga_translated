#!/usr/bin/env python3
"""Bersihkan balon SATU halaman, TANPA terjemahan sama sekali.

    python clean_only.py hasilnew/jp_6.JPG

client=None -> process_page melewati seluruh jalur terjemahan, jadi yang keluar
murni hasil deteksi + mask + erase/inpaint. Ini yang dipakai untuk menilai
"balon bersih": begitu teks Inggris ditulis di atasnya, tinta baru tidak bisa
dibedakan lagi dari sisa tinta lama.

Ditulis ke clean_<nama>.png (halaman bersih) dan clean_<nama>_cmp.png
(asli | bersih berdampingan, supaya tiap coretan sisa langsung kelihatan).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
NBSRC = ROOT / "_nbsrc"
STAGE = ROOT / ".stage"
_MAGIC = re.compile(r"^%%writefile\b.*\n?")

os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))

STAGE.mkdir(exist_ok=True)
for src in sorted(NBSRC.glob("*.py")):
    body = _MAGIC.sub("", src.read_text(encoding="utf-8"), count=1)
    dest = STAGE / src.name
    if not dest.exists() or dest.read_text(encoding="utf-8") != body:
        dest.write_text(body, encoding="utf-8")
sys.path.insert(0, str(STAGE))

import imgio, pipeline, typeset                          # noqa: E402,E401

path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "hasilnew/jp_6.JPG")
typeset.setup_fonts(verbose=False)
img = imgio.load_any(path)
res = pipeline.process_page(img, "clean_" + path.stem, None, "", debug=True)
pipeline.release_all()

src = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cle = cv2.cvtColor(res.cleaned, cv2.COLOR_RGB2BGR)
cv2.imwrite(f"clean_{path.stem}.png", cle)
bar = np.zeros((6, src.shape[1], 3), np.uint8)
cv2.imwrite(f"clean_{path.stem}_cmp.png", np.vstack([src, bar, cle]))
print(f"region={len(res.regions)} residu={res.report['residue_count']} "
      f"-> clean_{path.stem}.png, clean_{path.stem}_cmp.png")
