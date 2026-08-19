#!/usr/bin/env python3
"""Uji jalur penyedia BARU di translate.py — kode produksi, bukan salinan probe.

Bedanya dengan probe_llm2.py: probe itu punya SYSTEM/GLOSSARY/anggaran sendiri,
jadi lulusnya tidak membuktikan apa pun tentang notebook. Di sini yang dipanggil
translate.translate_page() APA ADANYA, dengan Region sungguhan dari
.probe_cache.pkl — persis seperti pipeline.process_page memanggilnya.

Yang dibuktikan:
  1. make_client + pick_model memilih jalur router dari nama UI
  2. _page_budget() mengukur balon dengan typeset.region_budget()
  3. translate_page() mengisi region.translation, SFX tidak tersentuh
  4. hasilnya muat: _max_feasible >= min_font_size untuk tiap balon

Kredensial dibaca lewat translate.get_api_key() dan TIDAK PERNAH dicetak.

    python probe_provider.py                 # router, anggaran ON
    BUDGET=0 python probe_provider.py        # router, anggaran OFF (pembanding)
"""

from __future__ import annotations

import os
import pathlib
import pickle
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
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

import translate as tl  # noqa: E402
import typeset  # noqa: E402
from config import PROTECTED_LABELS, SETTINGS  # noqa: E402

PROVIDER = "Router LLM (gorouter)"


def main() -> int:
    with (ROOT / ".probe_cache.pkl").open("rb") as f:
        regions = pickle.load(f)
    # Cache itu GEOMETRI saja — src_text-nya kosong karena pickle dibuat sebelum
    # tahap OCR di jalur probe. Kalimat Jepangnya diambil dari probe_llm2.SRC,
    # halaman yang sama, supaya yang diuji jalur produksinya bukan OCR-nya.
    from probe_llm2 import SRC

    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED

    SETTINGS.provider = PROVIDER
    SETTINGS.balloon_budget = os.environ.get("BUDGET", "1") != "0"
    key = tl.get_api_key(None, PROVIDER)
    client = tl.make_client(key, PROVIDER)
    model, _probes = tl.pick_model(client, verbose=True)
    print(f"[uji] {len(regions)} region  anggaran="
          f"{'ON' if SETTINGS.balloon_budget else 'OFF'}  (key tidak dicetak)")

    for r in regions:
        r.translation = None
        r.label = "DIALOGUE"
        r.src_text = SRC.get(r.idx, "")

    t0 = time.monotonic()
    tl.translate_page(client, model, regions, "English", "Manga Natural", True)
    dt = time.monotonic() - t0

    print(f"\n{'idx':>3} {'muat':>4} {'plafon':>6}  hasil")
    fail = []
    for r in sorted(regions, key=lambda x: x.idx):
        if r.label in PROTECTED_LABELS:
            print(f"{r.idx:>3} {'--':>4} {'--':>6}  [{r.label}] {r.src_text!r}")
            continue
        up = tl._clean_translation(r.translation or "").upper()
        mask = typeset._region_box_mask(r)[1]
        feas = typeset._max_feasible(up, mask, fp) if up else 0
        cap = typeset.region_font_cap(mask)
        mark = "" if feas >= SETTINGS.min_font_size else "  <- TIDAK MUAT"
        if mark:
            fail.append(r.idx)
        print(f"{r.idx:>3} {feas:>4} {cap:>6}  {up!r}{mark}")

    n_tr = sum(1 for r in regions if r.translation)
    print(f"\n{dt:.1f} s  diterjemah={n_tr}/{len(regions)}  "
          f"tidak muat={fail if fail else 'nol'}")
    return 1 if fail or not n_tr else 0


if __name__ == "__main__":
    raise SystemExit(main())
