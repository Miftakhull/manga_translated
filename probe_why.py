#!/usr/bin/env python3
"""Sekali pakai: mengapa validator menolak wording REFERENSI di r5/r9/r10?

Bukan probe permanen. Pertanyaannya sempit: `_max_feasible` (tanpa penggalan)
melaporkan referensi jatuh jauh di bawah plafon, padahal referensi itu memang
dirender rapi di CONTOH/2.webp. Yang perlu dilihat: berapa `feasible` tanpa
penggalan versus dengan penggalan, dan berapa ukuran yang benar-benar dipakai
render kita.
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path

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

import typeset                # noqa: E402
from config import SETTINGS   # noqa: E402

TESTS = {
    5: ["SHIZUKU-SAN.", "SHIZUKU...", "PREZ..."],
    9: ["I WAS PUTTING TOGETHER THE STUDENT COUNCIL'S ACTIVITY RECORDS.",
        "I WAS COMPILING THE EXECUTIVE RECORDS FOR THE STUDENT COUNCIL.",
        "JUST WRITING UP THE COUNCIL RECORDS."],
    10: ["A SUMMARY OF EVERYTHING WE'VE DONE SINCE THIS SPRING.",
         "THIS IS A SUMMARY OF OUR ACTIVITIES SINCE THIS SPRING, ISN'T IT?",
         "A ROUNDUP OF OUR WORK THIS TERM."],
}


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    rm = {r.idx: r for r in regions}
    for i, texts in TESTS.items():
        m = typeset._region_box_mask(rm[i])[1]
        cap = typeset.region_font_cap(m)
        print(f"r{i} mask={m.shape[1]}x{m.shape[0]} cap={cap}")
        for t in texts:
            fp_plain = typeset._max_feasible(t, m, fp)
            hy = typeset._search(t, m, SETTINGS.min_font_size, 30, fp,
                                 hyphenate=True)
            fit_size, lines, _y, ovf = typeset.fit(t, m, cap * 1.35, fp)
            print(f"   plain={fp_plain:>3} hyph={hy[0] if hy else 0:>3} "
                  f"fit={fit_size:>3} ovf={ovf!s:>5} len={len(t):>3} "
                  f"nlines={len(lines)}  {t[:46]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
