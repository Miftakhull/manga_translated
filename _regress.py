#!/usr/bin/env python3
"""Dua pertanyaan sekaligus, dijawab dari file yang ada di disk.

1. Paruh ATAS _cmp_jp_6/_13.png (yang BERSIH + terjemah penuh) itu keluaran run
   yang MANA? Dicari dengan MAE piksel ke SEMUA png/jpg kandidat, termasuk
   10_typeset.png di tiap folder debug — bukan lewat nama file.

2. Kode yang menghasilkan run itu masih ada di kerja6/2.ipynb (ditulis 16 Agu
   12:56, tepat SESUDAH run bersih 12:43). Jadi diff-nya terhadap _nbsrc/
   sekarang adalah daftar tersangka regresi. Yang dicetak: nama fungsi/konstanta
   yang berubah, bukan seluruh diff.
"""

from __future__ import annotations

import difflib
import glob
import json
import os
import pathlib
import re

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


def mae(a: Image.Image, b: Image.Image) -> float:
    bb = b.convert("L").resize(a.size, Image.LANCZOS)
    return float(np.abs(np.asarray(a.convert("L"), float) - np.asarray(bb, float)).mean())


def q1() -> None:
    print("#" * 70)
    print("# 1. ASAL PARUH ATAS (yang bersih) — MAE terkecil = sumbernya")
    print("#" * 70)
    cands = sorted(set(
        glob.glob("debug/*/10_typeset.png") + glob.glob("debug/*/09_cleaned.png")
        + glob.glob("kerja6/*/10_typeset.png") + glob.glob("kerja6/*/09_cleaned.png")
        + glob.glob("output/*.png") + glob.glob("output/*.jpg")
        + glob.glob("hasilnew/*.JPG") + glob.glob("hasilnew2/*.JPG")
        + glob.glob("kerja6/*.JPG") + glob.glob("contoh/*")))
    for tag in ("6", "13"):
        f = f"_cmp_jp_{tag}.png"
        if not os.path.exists(f):
            continue
        top, bot = half(f, True), half(f, False)
        rows = []
        for c in cands:
            stem = pathlib.Path(c).parent.name + "/" + pathlib.Path(c).stem
            if not re.search(rf"(^|[^0-9]){tag}([^0-9]|$)", stem):
                continue
            try:
                im = Image.open(c)
            except Exception:  # noqa: BLE001
                continue
            rows.append((mae(top, im), mae(bot, im), im.size, c))
        print(f"\n=== {f}   ATAS={top.size}")
        for st, sb, size, c in sorted(rows)[:8]:
            who = "  <<< ATAS" if st < 4 else ("  (bawah)" if sb < 4 else "")
            print(f"   atas={st:7.2f} bawah={sb:7.2f}  {str(size):<12} {c}{who}")


NAME_RE = re.compile(r"^\s*(?:def|class)\s+(\w+)|^([A-Z_][A-Z0-9_]{2,})\s*[:=]")


def owners(lines: list[str]) -> list[str]:
    """Untuk tiap baris, nama fungsi/konstanta yang memuatnya."""
    cur, out = "(atas-modul)", []
    for ln in lines:
        m = NAME_RE.match(ln)
        if m:
            cur = m.group(1) or m.group(2)
        out.append(cur)
    return out


def q2() -> None:
    print("\n" + "#" * 70)
    print("# 2. DIFF kerja6/2.ipynb (kode run BERSIH 16 Agu 12:43) vs _nbsrc/ SEKARANG")
    print("#" * 70)
    nb = json.load(open("kerja6/2.ipynb", encoding="utf-8"))
    good = {}
    for c in nb["cells"]:
        s = "".join(c["source"])
        first = s.split("\n", 1)[0]
        if "%%writefile" in first:
            good[first.split("/")[-1].strip()] = s.split("\n", 1)[1]
    for name in ("typeset.py", "erase.py", "textmask.py", "pipeline.py",
                 "translate.py", "verify.py", "config.py", "ocr.py", "inpaint.py"):
        if name not in good:
            continue
        live = pathlib.Path("_nbsrc") / name
        a = good[name].splitlines()
        b = live.read_text(encoding="utf-8").splitlines()
        oa, ob = owners(a), owners(b)
        touched: dict[str, list[int]] = {}
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue
            for i in range(i1, i2):
                touched.setdefault(oa[i], [0, 0])[0] += 1
            for j in range(j1, j2):
                touched.setdefault(ob[j], [0, 0])[1] += 1
        if not touched:
            print(f"\n=== {name}: IDENTIK")
            continue
        tot = sum(v[0] + v[1] for v in touched.values())
        print(f"\n=== {name}: {tot} baris berubah, di {len(touched)} tempat")
        for k, v in sorted(touched.items(), key=lambda t: -(t[1][0] + t[1][1]))[:14]:
            print(f"      -{v[0]:<4} +{v[1]:<4}  {k}")


def main() -> int:
    q1()
    q2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
