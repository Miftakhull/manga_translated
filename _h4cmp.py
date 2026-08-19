"""Cocokkan hasil hasilnew4 dengan halaman Jepang ASLI, per region.

Sebelum halaman asli ada, tiga hal tidak bisa dipastikan:
  1. apakah `residue_idx` alarm palsu atau sisa tinta sungguhan
  2. apakah `src_text` tiap region benar-benar apa yang tertulis di balon
  3. apakah balon kanan-atas itu SATU balon (jadi r0 duplikat) atau dua lobus

Yang ditulis:
  _h4c_r<i>.png   crop asli | crop hasil, ditempel bersebelahan, diperbesar
  ringkasan       ukuran, median gray asli vs hasil di dalam bubble_mask

Tidak mengubah apa pun. Probe murni.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
H4 = ROOT / "hasilnew4"
JP = H4 / "hitomi_3740721_015.webp"
OUT = H4 / "hitomi_3740721_015 (1).png"
REP = json.loads((H4 / "hitomi_3740721_015.json").read_text(encoding="utf-8"))

jp = cv2.imread(str(JP), cv2.IMREAD_COLOR)
out = cv2.imread(str(OUT), cv2.IMREAD_COLOR)
if jp is None or out is None:
    sys.exit(f"gagal baca: jp={jp is not None} out={out is not None}")
print(f"asli  {jp.shape[1]}x{jp.shape[0]}", flush=True)
print(f"hasil {out.shape[1]}x{out.shape[0]}", flush=True)
if jp.shape[:2] != out.shape[:2]:
    print("  [!] UKURAN BEDA — crop bbox tidak bisa dibandingkan langsung",
          flush=True)

want = [int(a) for a in sys.argv[1:]] or list(range(len(REP["regions"])))
pad = 6
print("\n  r |      bbox           |  med_asli med_hasil | src_text", flush=True)
for reg in REP["regions"]:
    i = reg["idx"]
    if i not in want:
        continue
    x1, y1, x2, y2 = reg["bbox"]
    a = jp[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad]
    b = out[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad]
    ga = cv2.cvtColor(jp[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(out[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    print(f" r{i:<2}| {str(tuple(reg['bbox'])):20s}| {np.median(ga):9.1f}"
          f" {np.median(gb):9.1f} | {reg['src_text']!r}", flush=True)

    # Skala supaya sisi pendek >= 220 px: glyph kecil harus terbaca mata.
    h, w = a.shape[:2]
    s = max(1, int(np.ceil(220 / max(min(h, w), 1))))
    s = min(s, 8)
    a2 = cv2.resize(a, (w * s, h * s), interpolation=cv2.INTER_NEAREST)
    b2 = cv2.resize(b, (b.shape[1] * s, b.shape[0] * s),
                    interpolation=cv2.INTER_NEAREST)
    hh = max(a2.shape[0], b2.shape[0])
    gap = 8
    canvas = np.full((hh, a2.shape[1] + gap + b2.shape[1], 3), 255, np.uint8)
    canvas[:, a2.shape[1]:a2.shape[1] + gap] = (0, 0, 255)
    canvas[: a2.shape[0], : a2.shape[1]] = a2
    canvas[: b2.shape[0], a2.shape[1] + gap:] = b2
    cv2.imwrite(str(ROOT / f"_h4c_r{i}.png"), canvas)

print("\n -> _h4c_r*.png  (kiri=ASLI, kanan=HASIL, dipisah garis merah)",
      flush=True)
