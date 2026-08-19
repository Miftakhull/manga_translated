%%writefile /content/mangatl/pipeline.py

"""Orkestrasi per halaman: detect -> mask -> OCR -> LLM -> erase -> verify -> typeset.

Urutan wajib: klasifikasi SFX dari LLM harus selesai SEBELUM compose mask,
karena exclusion SFX bergantung pada label. Menukar dua langkah ini membuat
pipeline menghapus persis apa yang diminta untuk dijaga.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import DEBUG_DIR, OUTPUT, RUN_NOTES, SETTINGS, Region, note, notes_since
import detect
import erase
import imgio
import ocr
import textmask
import translate as tl
import typeset
import verify


@dataclass
class PageResult:
    stem: str
    original: np.ndarray
    cleaned: np.ndarray
    final: np.ndarray
    regions: list[Region]
    report: dict
    paths: dict[str, Path]


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _dump(stem: str, name: str, img: np.ndarray) -> None:
    d = DEBUG_DIR / stem
    d.mkdir(parents=True, exist_ok=True)
    arr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(str(d / f"{name}.png"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def _draw_boxes(img: np.ndarray, regions: list[Region]) -> np.ndarray:
    """Kotak berwarna per kelas untuk debug: merah = SFX (dijaga)."""
    out = img.copy()
    colors = {
        "SFX": (255, 0, 0), "UNREADABLE": (255, 128, 0), "DIALOGUE": (0, 200, 0),
        "THOUGHT": (0, 160, 255), "NARRATION": (200, 0, 200), "SIGN": (255, 200, 0),
    }
    for r in regions:
        x1, y1, x2, y2 = r.bbox
        c = colors.get(r.label, (128, 128, 128))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.putText(out, f"{r.idx}:{r.label[:4]}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out


def process_page(
    img: np.ndarray, stem: str, client=None, model: str = "",
    debug: bool | None = None, outdir: Path | None = None,
    progress=None, target_lang: str = "English",
    style: str = "Manga Natural", keep_honorifics: bool = True,
) -> PageResult:
    """Satu halaman penuh. client None = jalan tanpa terjemahan (halaman bersih)."""
    debug = SETTINGS.debug if debug is None else debug
    dev = _device()
    original = img.copy()
    # Batas awal catatan halaman ini. Diambil SEBELUM apa pun jalan supaya
    # notes_since() nanti hanya mengembalikan kegagalan halaman INI — kalau
    # tidak, halaman ke-3 dalam satu batch mewarisi error halaman ke-1 dan
    # banner UI menuduh halaman yang sebenarnya bersih. RUN_NOTES di-mutate
    # (bukan di-rebind) jadi len() atas nama yang diimpor tetap akurat.
    note_mark = len(RUN_NOTES)

    def step(frac: float, msg: str) -> None:
        if progress is not None:
            progress(frac, desc=msg)

    if debug:
        _dump(stem, "01_input", img)

    # Lantai ukuran font berskala lebar halaman (typeset.min_font()). Di-set di
    # sini, SEBELUM terjemahan, karena anggaran karakter yang dikirim ke model
    # (translate._page_budget) memakai lantai itu — kalau baru di-set saat render,
    # model diberi anggaran halaman kalibrasi dan menulis terlalu pendek.
    typeset.set_page_width(img.shape[1])

    step(0.10, "deteksi region")
    regions, bubbles = detect.detect(img)
    if not regions:
        # Nol region bukan sukses: halaman keluar identik dengan aslinya. Dicatat
        # sebagai warn supaya baris ini muncul di banner UI, karena tabel hanya
        # akan menampilkan angka 0 di semua kolom dan itu tidak menjelaskan apa pun.
        note("warn", "detect",
             f"{stem}: tidak ada region terdeteksi — halaman keluar TANPA perubahan")
        rep = verify.report([], [], typeset.FONT_USED, notes_since(note_mark))
        paths = imgio.save_outputs(original, stem, outdir)
        return PageResult(stem, original, original, original, [], rep, paths)

    step(0.25, "bangun mask teks")
    soft = textmask.ctd_soft_mask(img)
    for r in regions:
        textmask.build_region_mask(img, r, soft)
    # Setelah SEMUA ink_mask terisi: balon ganda dipartisi per lobus dari
    # interior gabungan. Butuh ink_mask semua region, jadi tidak bisa di dalam
    # loop di atas.
    textmask.partition_shared_interiors(img, regions)
    # ...lalu balon bertetangga yang interiornya beririsan dibuat saling lepas.
    # Tanpa ini _clip_to_mask memotong glyph di zona irisan (lihat docstring-nya).
    textmask.disjoin_overlapping_interiors(img, regions)
    # ...lalu garis balonnya dilepas dari mask hapus. Harus SESUDAH kedua
    # langkah di atas, karena penjaganya dihitung dari bubble_mask final.
    textmask.protect_bubble_outline(img, regions)
    del soft
    gc.collect()

    step(0.40, "OCR Jepang")
    ocr.read_all(img, regions)
    # Model OCR TETAP di memori antar halaman (dimuat sekali per batch).
    # Kalau dilepas tiap halaman, batch multi membayar ~5-8 dtk reload
    # manga-ocr per halaman dan kecepatan satuan jadi turun. release()
    # cukup di release_all() di akhir batch (lihat _run).
    gc.collect()

    step(0.55, "klasifikasi SFX + terjemah")
    if client is not None and model:
        try:
            tl.translate_page(client, model, regions, target_lang,
                              style, keep_honorifics)
        except Exception as exc:  # noqa: BLE001 - jaringan tidak boleh membunuh halaman
            note("error", "pipeline",
                 f"{stem}: LLM gagal ({exc}); pakai label heuristik — "
                 "halaman keluar TANPA terjemahan")
            tl._fallback_labels(regions)
    else:
        tl._fallback_labels(regions)

    if debug:
        _dump(stem, "03_boxes", _draw_boxes(img, regions))

    # SFX exclusion terjadi di sini, sebelum erase apa pun.
    erase_mask, protected_mask = textmask.compose_page_mask(img, regions)
    if debug:
        raw = np.zeros(img.shape[:2], np.uint8)
        for r in regions:
            if r.ink_mask is None:
                continue
            x1, y1, x2, y2 = r.bbox
            mh, mw = r.ink_mask.shape[:2]
            y2, x2 = min(y2, y1 + mh), min(x2, x1 + mw)
            raw[y1:y2, x1:x2] = np.maximum(raw[y1:y2, x1:x2], r.ink_mask[: y2 - y1, : x2 - x1])
        _dump(stem, "05_mask", raw)
        _dump(stem, "07_mask_after_sfx_exclusion", erase_mask)

    if not verify.assert_sfx_intact(erase_mask, protected_mask):
        raise AssertionError("mask hapus menyentuh SFX — kontrak pipeline dilanggar")

    step(0.70, "hapus teks asli")
    cleaned = erase.erase_page(img, regions, dev)

    step(0.80, "verifikasi residu")
    failed = verify.find_residue(cleaned, regions)
    if failed:
        cleaned = verify.escalate(img, cleaned, failed, dev)
        failed = verify.find_residue(cleaned, regions)
    if debug:
        _dump(stem, "09_cleaned", cleaned)

    step(0.90, "typeset Inggris")
    final = typeset.render_page(cleaned, regions)
    if debug:
        _dump(stem, "10_typeset", final)

    rep = verify.report(regions, failed, typeset.FONT_USED, notes_since(note_mark))
    rep["bubble_count"] = len(bubbles)
    if debug:
        (DEBUG_DIR / stem).mkdir(parents=True, exist_ok=True)
        (DEBUG_DIR / stem / "report.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    step(0.97, "simpan")
    paths = imgio.save_outputs(final, stem, outdir)
    # Sidecar: teks asli tersimpan walau terjemahan gagal — kerja tidak hilang.
    # Ambil path pertama yang ditulis (png/jpg sesuai format terpilih).
    next(iter(paths.values())).with_suffix(".json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return PageResult(stem, original, cleaned, final, regions, rep, paths)


def process_batch(
    files: list[str | Path], api_key: str | None = None, debug: bool = False,
    progress=None, target_lang: str = "English",
    style: str = "Manga Natural", keep_honorifics: bool = True,
    outdir: Path | None = None, reset_notes: bool = True,
) -> tuple[list[PageResult], dict]:
    """Beberapa halaman sekali jalan. Model dipilih sekali, dipakai ulang.

    `reset_notes=False` dipakai app.py: UI sudah mencatat kegagalannya sendiri
    (mis. "API key tidak terbaca") SEBELUM memanggil ini, dan pembersihan di sini
    akan menghapus justru catatan yang menjelaskan kenapa `api_key` None. Default
    True supaya pemanggil skrip (run_full.py, sel 25) tidak mewarisi catatan run
    sebelumnya di sesi yang sama.
    """
    SETTINGS.debug = debug
    # Dikosongkan in-place (bukan rebind) supaya `from config import RUN_NOTES`
    # di modul lain tetap menunjuk daftar yang sama.
    if reset_notes:
        RUN_NOTES.clear()
    client, model, probes = None, "", []
    if api_key:
        try:
            # Penyedia diambil dari SETTINGS.provider (diisi UI) — bukan parameter
            # baru — supaya process_page dan pemanggil lain tidak perlu ikut
            # meneruskannya. make_client yang memutuskan kelas client-nya.
            client = tl.make_client(api_key)
            model, probes = tl.pick_model(client, verbose=False)
        except (RuntimeError, ImportError) as exc:
            # error, bukan warn: tanpa client SELURUH batch keluar berbahasa
            # Jepang. Ini penyebab paling sering "diterjemah 0" di semua halaman
            # sekaligus (kunci salah/kosong), dan harus jadi baris pertama banner.
            note("error", "pipeline",
                 f"tidak ada model LLM: {exc} — SEMUA halaman keluar TANPA terjemahan")
            client = None
    # Anggaran balon memanggil typeset.layout(), jadi fontnya harus sudah ada
    # sebelum halaman pertama. Tanpa ini balon pertama membayar unduhan font di
    # tengah pengukuran, dan waktunya tampak seperti biaya anggaran.
    if client is not None and not typeset.FONT_USED:
        typeset.setup_fonts(verbose=False)

    results: list[PageResult] = []
    total = max(len(files), 1)
    for i, f in enumerate(files):
        stem = Path(f).stem
        img = imgio.load_any(f)

        def sub(frac: float, desc: str = "", _i: int = i) -> None:
            if progress is not None:
                progress((_i + frac) / total, desc=f"[{_i + 1}/{total}] {desc}")

        results.append(
            process_page(img, stem, client, model, debug=debug, progress=sub,
                         target_lang=target_lang, style=style,
                         keep_honorifics=keep_honorifics, outdir=outdir)
        )
        del img
        gc.collect()

    summary = {
        "pages": len(results),
        "model": model or "none",
        "provider": SETTINGS.provider,
        "target_lang": target_lang,
        "style": style,
        "keep_honorifics": keep_honorifics,
        "font_used": typeset.FONT_USED,
        "probes": [p.as_row() for p in probes],
        "residue_total": sum(r.report["residue_count"] for r in results),
        "overflow_total": sum(r.report["overflow_count"] for r in results),
        "sfx_total": sum(len(r.report["sfx_idx"]) for r in results),
        # Seluruh catatan run, termasuk yang terjadi SEBELUM halaman pertama
        # (mis. "tidak ada model LLM") yang tidak dimiliki report halaman mana
        # pun. app.py butuh keduanya: per halaman untuk kolom tabel, batch untuk
        # sebab yang berlaku menyeluruh.
        "notes": [{"level": lv, "tag": tg, "msg": ms} for lv, tg, ms in RUN_NOTES],
        "error_total": sum(1 for lv, _t, _m in RUN_NOTES if lv == "error"),
        "warn_total": sum(1 for lv, _t, _m in RUN_NOTES if lv == "warn"),
        "translated_total": sum(r.report["translated_count"] for r in results),
        "untranslated_total": sum(r.report["untranslated_count"] for r in results),
    }
    zip_path = imgio.make_zip(
        [p for r in results for p in r.paths.values()], OUTPUT / "manga_translated.zip"
    )
    summary["zip"] = str(zip_path)
    return results, summary


def release_all() -> None:
    """Bebaskan semua sesi model — RAM Colab cuma ~12.7 GB."""
    import inpaint

    detect.release()
    textmask.release()
    ocr.release()
    inpaint.release()
    gc.collect()

