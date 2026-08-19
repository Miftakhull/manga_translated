#!/usr/bin/env python3
"""Jalankan verifikasi typeset di Windows tanpa Colab. Satu perintah, satu laporan.

    python verify_local.py

Semua yang bisa diperiksa TANPA jaringan dan TANPA torch/manga-ocr dijalankan
di sini: stage modul, import, probe paket opsional, lalu `selftest.run()` yang
memuat kontrak balon ganda (tidak saling timpa, tidak keluar balon, ukuran font
proporsional ke balon, tanpa tanda hubung, punctuation CJK bersih).

`_nbsrc/*.py` TIDAK bisa diimpor langsung: baris pertamanya `%%writefile ...`
adalah magic IPython, bukan Python — `import config` langsung darinya berujung
SyntaxError. Colab menjalankan selnya, dan magic itulah yang menulis sisa sel ke
/content/mangatl/<nama>.py. Jadi di sini langkah yang sama ditiru: baris magic
dibuang, sisanya di-stage ke satu folder, folder itu yang masuk sys.path. Efek
sampingnya bagus — yang diuji adalah teks yang BENAR-BENAR dijalankan Colab.

Exit 0 = semua lolos. Exit 1 = ada yang gagal (rinciannya dicetak).
"""

from __future__ import annotations

import importlib
import os
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NBSRC = ROOT / "_nbsrc"
STAGE = ROOT / ".stage"

_MAGIC = re.compile(r"^%%writefile\b.*\n?")

# config.py membaca dua env var ini; tanpa keduanya ia menunjuk ke /content.
# WORK = repo, jadi WEIGHTS/FONTS langsung cocok dengan weights/ dan fonts/.
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))

# Paket yang HANYA dipakai jalur berat; ketiadaannya tidak boleh menggagalkan
# self-test typeset, tapi harus terlihat supaya jelas apa yang belum teruji.
_OPTIONAL = {
    "torch": "inpaint.py (LaMa)",
    "manga_ocr": "ocr.py",
    "onnxruntime": "detect.py + textmask.py (CTD)",
    "pyphen": "penggalan kata berbasis kamus (ada fallback vokal-konsonan)",
    "deepl": "translate.py (jalur DeepL sungguhan)",
}

# Modul yang wajib bisa diimpor supaya selftest.run() punya arti. inpaint/ocr
# sengaja TIDAK di sini: keduanya butuh torch/manga-ocr.
_CORE = ("config", "imgio", "detect", "textmask", "translate",
         "erase", "typeset", "verify", "selftest")


def _stage() -> int:
    """Tulis _nbsrc/*.py ke STAGE tanpa baris magic. Return jumlah file."""
    STAGE.mkdir(exist_ok=True)
    n = 0
    for src in sorted(NBSRC.glob("*.py")):
        text = src.read_text(encoding="utf-8")
        body = _MAGIC.sub("", text, count=1)
        dest = STAGE / src.name
        # Tulis hanya kalau beda — supaya .pyc tidak batal terus tiap jalan.
        if not dest.exists() or dest.read_text(encoding="utf-8") != body:
            dest.write_text(body, encoding="utf-8")
        n += 1
    return n


def _probe() -> None:
    """Laporkan paket opsional — informasi, bukan kriteria lulus."""
    print("Paket opsional:")
    for mod, why in _OPTIONAL.items():
        try:
            importlib.import_module(mod)
            print(f"  [ADA   ] {mod:14s} {why}")
        except Exception as exc:                       # noqa: BLE001
            print(f"  [TIADA ] {mod:14s} {why}  ({type(exc).__name__})")


def _import_core() -> tuple[dict[str, object], list[str]]:
    mods: dict[str, object] = {}
    bad: list[str] = []
    print("\nImport modul inti:")
    for name in _CORE:
        try:
            mods[name] = importlib.import_module(name)
            print(f"  [ OK   ] {name}")
        except Exception:                              # noqa: BLE001
            bad.append(name)
            print(f"  [GAGAL ] {name}")
            traceback.print_exc(limit=3)
    return mods, bad


def main() -> int:
    n = _stage()
    # sys.path DIPASANG setelah stage: kalau dipasang lebih dulu dan stage
    # gagal, import malah menjaring _nbsrc yang ber-magic dan pesan errornya
    # jadi SyntaxError yang menyesatkan.
    sys.path.insert(0, str(STAGE))
    print(f"MANGATL_WORK = {os.environ['MANGATL_WORK']}")
    print(f"MANGATL_ROOT = {os.environ['MANGATL_ROOT']}")
    print(f"stage        = {STAGE}  ({n} modul)\n")
    _probe()

    mods, bad = _import_core()
    if bad:
        print(f"\nGAGAL: modul inti tidak bisa diimpor: {', '.join(bad)}")
        return 1

    typeset = mods["typeset"]
    selftest = mods["selftest"]

    # anime_ace.ttf sudah ada di fonts/, jadi _download() langsung pulang dan
    # ini berjalan offline. Tanpa font, blok render selftest dilewati dan
    # justru kontrak inti balon ganda yang tidak teruji.
    print("\nSetup font:")
    try:
        used = typeset.setup_fonts(verbose=True)
        print(f"  FONT_USED = {used or '(kosong)'}")
    except Exception:                                  # noqa: BLE001
        print("  [GAGAL ] setup_fonts()")
        traceback.print_exc(limit=3)

    print("\nSelf-test:")
    try:
        ok = selftest.run(verbose=True)
    except Exception:                                  # noqa: BLE001
        print("  [GAGAL ] selftest.run() raise")
        traceback.print_exc()
        return 1

    print("\nHASIL: " + ("SEMUA LOLOS" if ok else "ADA YANG GAGAL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
