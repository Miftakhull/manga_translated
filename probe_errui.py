#!/usr/bin/env python3
"""Buktikan lapisan diagnostik BEKERJA, tanpa satu pun panggilan API.

Dijalankan lewat .stage (bukan _nbsrc) supaya yang diuji teks yang sama dengan
yang dijalankan Colab — lihat verify_local.py.

Dua simulasi, keduanya offline:

  A. KEY TIDAK ADA. provider LLM dipilih tapi env/file key-nya sengaja dikosongkan.
     Inilah jalur yang menghasilkan layar user: client None -> `diterjemah 0` di
     semua halaman -> dulu NOL pesan. Yang dinilai: banner memuat
     "TIDAK ADA TERJEMAHAN", galeri dan ZIP TETAP ada, dan run.log memuat "[!!]".

  B. FONT GAGAL. summary['font_used'] dipaksa None lewat tambalan pada
     process_batch. Dulu `Path(None).name` melempar TypeError SETELAH semua
     halaman diproses — seluruh run hilang tanpa pesan. Yang dinilai: _run
     mengembalikan hasil, bukan melempar.

Nol token: `_router_call_any` tidak pernah dipanggil karena client-nya None.
Sebagai jaring kedua, urllib.request.urlopen ditambal supaya MELEMPAR — kalau
ada jalur tersembunyi yang tetap mencoba jaringan, probe ini yang menangkapnya.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(ROOT / ".stage"))
# Kunci dikosongkan DI SINI, sebelum config diimpor: inilah yang disimulasikan.
for _k in ("FAUCET_API_KEY", "ROUTER_API_KEY", "DEEPL_API_KEY"):
    os.environ.pop(_k, None)
os.environ["MANGATL_NO_KEYFILE"] = "1"

sys.path.insert(0, str(ROOT / ".stage"))

PAGE = ROOT / "hasilnew" / "jp_6.JPG"


def _no_network() -> None:
    """Bikin jaringan MELEMPAR. Bukti bahwa probe ini tidak membakar token."""
    import urllib.request

    def _boom(*a, **k):
        raise AssertionError("probe mencoba jaringan — seharusnya tidak pernah")

    urllib.request.urlopen = _boom


def main() -> int:
    import app
    import config
    import translate as tl
    import typeset

    # Font disiapkan SEBELUM jaringan disegel. setup_fonts() mengunduh berkas
    # lisensi OFL saat pertama kali, dan itu bukan bagian yang diuji di sini —
    # versi pertama probe ini memblokirnya lalu melaporkan cacat yang tidak ada.
    typeset.setup_fonts(verbose=False)
    _no_network()

    # get_api_key tidak boleh menemukan apa pun, termasuk dari file lokal.
    tl.get_api_key = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("API key tidak ditemukan (disimulasikan)"))

    bad: list[str] = []
    out = app.OUTPUT / app.LOG_NAME
    out.unlink(missing_ok=True)

    # ---------------------------------------------------------------- A
    gallery, rar, zipf, md, raw, logtext, logfile = app._run(
        [str(PAGE)], "LLM (freetokenfaucet)", "", "English", "Manga Natural",
        True, False, None, "png", progress=None,
    )
    print("=" * 70)
    print(md[:1400])
    print("=" * 70)

    if "TIDAK ADA TERJEMAHAN" not in md:
        bad.append("A: banner tidak memuat 'TIDAK ADA TERJEMAHAN'")
    if not gallery:
        bad.append("A: galeri kosong — hasil hilang, padahal harus tetap keluar")
    if not zipf or not Path(zipf).exists():
        bad.append("A: ZIP tidak dibuat")
    if "[!!]" not in logtext:
        bad.append("A: log tidak memuat penanda error [!!]")
    if not logfile or not Path(logfile).exists():
        bad.append("A: run.log tidak ditulis")
    elif "[!!]" not in Path(logfile).read_text(encoding="utf-8"):
        bad.append("A: run.log ada tapi tanpa [!!]")
    print(f"galeri={len(gallery or [])} zip={bool(zipf)} log={logfile} "
          f"panjang_log={len(logtext)}")
    print("catatan error:",
          [n for n in config.RUN_NOTES if n[0] == "error"][:4])

    # ---------------------------------------------------------------- B
    import pipeline

    _orig = pipeline.process_batch

    def _fontless(*a, **k):
        res, summ = _orig(*a, **k)
        summ["font_used"] = None      # jalur yang dulu melempar TypeError
        return res, summ

    pipeline.process_batch = _fontless
    try:
        _g, _r, _z, md2, _raw2, _lt2, _lf2 = app._run(
            [str(PAGE)], "LLM (freetokenfaucet)", "", "English", "Manga Natural",
            True, False, None, "png", progress=None,
        )
    except Exception:
        bad.append("B: _run MELEMPAR saat font_used None:\n" + traceback.format_exc())
        md2 = ""
    finally:
        pipeline.process_batch = _orig
    if md2 and "PIPELINE BERHENTI" in md2:
        bad.append("B: font None malah jadi traceback, bukan ditangani halus")
    if md2 and "GAGAL DIMUAT" not in md2:
        bad.append("B: font None tidak dilaporkan di baris header")
    print("B header:", md2.splitlines()[0][:200] if md2 else "(kosong)")

    print("=" * 70)
    if bad:
        print("GAGAL:")
        for b in bad:
            print(" -", b)
        return 1
    print("PROBE LOLOS: kegagalan terlihat, hasil tetap keluar, nol token API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
