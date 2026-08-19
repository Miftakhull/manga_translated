%%writefile /content/mangatl/app.py

"""UI Gradio: upload -> TRANSLATE -> selesai.

Gradio 6: theme/css pindah ke launch(); di Colab share=True wajib dan
gr.Progress butuh .queue().
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import traceback
from pathlib import Path

import gradio as gr

from config import (LANGUAGES, OUTPUT, PROVIDER_DEFAULT, PROVIDERS, RUN_NOTES,
                    SETTINGS, TRANSLATION_STYLES, note)
import pipeline
import translate as tl
import typeset

CSS = """
.gradio-container {max-width: 1280px !important;}
#go {font-size: 18px; font-weight: 700; height: 56px;}
footer {display: none !important;}
"""

# Nama file log di OUTPUT. Ditulis ke disk, bukan hanya ditaruh di gr.Code,
# karena gr.File butuh path nyata untuk diunduh — dan karena kalau sesi Colab
# mati mendadak, log-nya masih ada di /content/mangatl/output.
LOG_NAME = "run.log"


class _Tee:
    """Tulis ke stream asli SEKALIGUS tampung di buffer.

    Bukan sekadar redirect_stdout(StringIO): kalau keluaran dialihkan sepenuhnya,
    sel Colab yang memang sedang dilihat user jadi bisu total dan progress bar
    library pihak ketiga hilang. Tee menjaga keduanya — Colab tetap dapat aliran
    langsung, UI dapat salinan lengkap.

    Sengaja TIDAK mewarisi io.TextIOBase: sebagian library memeriksa atribut
    seperti `.encoding` atau memanggil `.fileno()`, dan pewarisan setengah jalan
    membuat pemeriksaan itu lolos lalu gagal belakangan. Di sini semua yang tidak
    dikenal diteruskan ke stream asli lewat __getattr__, jadi objeknya berperilaku
    persis seperti stdout aslinya.
    """

    def __init__(self, real, buf: io.StringIO):
        self._real, self._buf = real, buf

    def write(self, s: str) -> int:
        self._buf.write(s)
        try:
            return self._real.write(s)
        except Exception:  # noqa: BLE001 - stream asli boleh mati, buffer tidak
            return len(s)

    def flush(self) -> None:
        with contextlib.suppress(Exception):
            self._real.flush()

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        # tqdm dan sebagian library memanggil ini; diteruskan supaya mereka
        # mengambil keputusan yang sama seperti tanpa Tee.
        return self._real.fileno()

    def __getattr__(self, name):
        return getattr(self._real, name)


@contextlib.contextmanager
def _capture():
    """Jalankan blok dengan stdout+stderr di-tee ke satu buffer.

    yield buffer-nya, bukan teksnya: pemanggil perlu membaca isi buffer JUGA
    saat blok gagal di tengah jalan lewat except di luar sini — dan pada saat itu
    nilai balik context manager sudah tidak bisa diambil lagi.
    """
    buf = io.StringIO()
    import sys

    with contextlib.redirect_stdout(_Tee(sys.stdout, buf)), \
            contextlib.redirect_stderr(_Tee(sys.stderr, buf)):
        yield buf


def _write_log(text: str) -> str | None:
    """Simpan log mentah ke OUTPUT/run.log. None kalau tidak bisa ditulis."""
    try:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        p = OUTPUT / LOG_NAME
        # unlink dulu: menimpa file yang sedang dipegang gr.File di klik
        # sebelumnya bisa menyisakan ekor log lama di unduhan.
        p.unlink(missing_ok=True)
        p.write_text(text or "(tidak ada keluaran)", encoding="utf-8")
        return str(p)
    except OSError:
        return None


def _archive(files: list[Path], dest: Path) -> tuple[str | None, str]:
    """Bungkus hasil jadi .rar. Kembalikan (path, catatan); path None = gagal.

    RAR itu format proprietary. Stdlib Python tidak punya penulis RAR dan
    `rarfile` cuma bisa MEMBACA, jadi biner `rar` resmi (dipasang di sel 3)
    satu-satunya cara membuat .rar asli. Kalau binernya tidak ada, pemanggil
    tetap punya ZIP yang sah; JANGAN pernah me-rename ZIP jadi .rar karena
    WinRAR menolaknya dan user baru sadar setelah unduhan selesai.
    """
    exe = shutil.which("rar")
    if exe is None:
        return None, "biner `rar` tidak terpasang"
    dest.unlink(missing_ok=True)
    # -ep1 buang path induk, -m5 kompresi maksimum, -idq senyap, -y ya ke semua.
    cmd = [exe, "a", "-ep1", "-m5", "-idq", "-y", str(dest), *(str(f) for f in files)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    if proc.returncode != 0 or not dest.exists():
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return None, tail[-1][:150] if tail else f"rar keluar kode {proc.returncode}"
    return str(dest), "ok"


# ---------------------------------------------------------------- diagnosa

# Penanda banner. Karakter, bukan warna CSS: Markdown Gradio tidak menjamin
# kelas CSS kustom lolos sanitizer-nya, sedangkan emoji selalu terlihat dan
# ikut terbaca kalau tabelnya di-copy ke tempat lain.
_RED, _YELLOW, _GREEN = "\U0001F534", "\U0001F7E1", "\U0001F7E2"


def _diagnose(results, summary: dict, notes: list | None = None) -> list[str]:
    """Baris banner untuk disisipkan di ATAS tabel. Murni, tanpa I/O.

    Tanpa I/O dan tanpa Gradio dengan sengaja: inilah satu-satunya bagian
    lapisan diagnostik yang punya logika keputusan, jadi ia harus bisa diuji di
    selftest offline dengan summary bikinan. Begitu fungsi ini menyentuh disk
    atau gr.*, kontraknya cuma bisa dibuktikan dengan menjalankan seluruh UI.

    `results` = daftar objek dengan `.stem` dan `.report`; `summary` = dict
    process_batch; `notes` = catatan tingkat batch (default: summary['notes']).
    Return daftar baris markdown, sudah urut dari yang paling parah.
    """
    notes = notes if notes is not None else (summary.get("notes") or [])
    notes = [n if isinstance(n, dict) else {"level": n[0], "tag": n[1], "msg": n[2]}
             for n in notes]
    out: list[str] = []

    def note_lines(level: str, limit: int = 6) -> list[str]:
        seen, picked = set(), []
        for n in notes:
            if n.get("level") != level:
                continue
            msg = str(n.get("msg", ""))
            # De-dup: satu kegagalan jaringan yang sama terulang di 3 halaman
            # menghasilkan 3 baris identik, dan banner jadi tembok teks yang
            # justru menyembunyikan sebab kedua yang berbeda.
            if msg in seen:
                continue
            seen.add(msg)
            picked.append(f">   `[{n.get('tag', '?')}]` {msg}")
            if len(picked) >= limit:
                picked.append(f">   ...dan {sum(1 for x in notes if x.get('level') == level) - limit} lagi, lihat **Log lengkap**")
                break
        return picked

    # 1. Paling parah: ada halaman yang butuh terjemahan tapi tidak dapat satu pun.
    # Syaratnya translatable_count, bukan region_count: halaman yang isinya SFX
    # semua memang SEHARUSNYA translated_count 0, dan menuduhnya gagal akan
    # melatih user mengabaikan banner ini.
    dead = [r for r in results
            if (r.report.get("translatable_count",
                             r.report.get("region_count", 0)) > 0
                and r.report.get("translated_count", 0) == 0)]
    if dead:
        names = ", ".join(f"`{r.stem}`" for r in dead[:8])
        more = f" (+{len(dead) - 8} lagi)" if len(dead) > 8 else ""
        out += [
            f"> ## {_RED} TIDAK ADA TERJEMAHAN",
            f"> {len(dead)} halaman keluar **masih berbahasa Jepang**: {names}{more}.",
            ">",
            "> Gambar dan ZIP tetap dibuat — teks aslinya tersimpan di sidecar "
            "JSON tiap halaman, jadi kerjanya tidak hilang. Sebab yang tercatat:",
        ]
        out += note_lines("error") or [">   _(tidak ada catatan error — lihat **Log lengkap** di bawah)_"]
        out.append("")
    else:
        # 2. Sebagian balon tertinggal. Bukan kegagalan total, tapi tetap merah:
        # halaman ini TERCETAK campur Jepang-Inggris dan tidak layak dipakai.
        partial = [r for r in results if r.report.get("untranslated_count", 0) > 0]
        if partial:
            det = "; ".join(
                f"`{r.stem}` balon {r.report.get('untranslated_idx', [])}"
                for r in partial[:6]
            )
            out += [
                f"> ## {_RED} ADA BALON YANG BELUM DITERJEMAH",
                f"> {sum(r.report.get('untranslated_count', 0) for r in partial)} balon "
                f"di {len(partial)} halaman masih berbahasa Jepang: {det}.",
                "",
            ]

    # 3. Error lain yang belum masuk banner di atas (OCR mati, klien gagal dibuat).
    if not dead:
        errs = note_lines("error")
        if errs:
            out += [f"> ## {_RED} ADA YANG GAGAL", *errs, ""]

    # 4. Kuning: hasil masih layak dipakai, cuma perlu dilihat.
    soft: list[str] = []
    res_tot = summary.get("residue_total", 0)
    ovf_tot = summary.get("overflow_total", 0)
    if res_tot:
        bad = [f"`{r.stem}` {r.report.get('residue_idx', [])}"
               for r in results if r.report.get("residue_count", 0)]
        soft.append(f"> {_YELLOW} **Sisa teks asli** di {res_tot} region: "
                    f"{'; '.join(bad[:6])} — teks Jepangnya masih terlihat di bawah hasil.")
    if ovf_tot:
        soft.append(f"> {_YELLOW} **{ovf_tot} balon overflow**: teks tercetak "
                    "melebihi ruang balon.")
    warns = note_lines("warn", limit=4)
    if warns:
        soft += [f"> {_YELLOW} **Peringatan:**", *warns]
    if soft:
        out += [*soft, ""]

    if not out:
        n_tr = summary.get("translated_total")
        extra = f" — {n_tr} balon diterjemah" if n_tr else ""
        out = [f"> {_GREEN} **Bersih:** tidak ada residu, overflow, "
               f"maupun balon yang gagal diterjemah{extra}.", ""]
    return out


def _probe_table(provider: str, api_key: str) -> str:
    """Cek key penyedia aktif: kuota DeepL, atau satu panggilan uji ke router.

    Dibungkus tangkap-keluaran + except: ini tombol PERTAMA yang diklik orang,
    jadi ia tidak boleh bisa gagal dalam diam. Sebelumnya hanya RuntimeError dan
    ImportError yang tertangkap — sebuah URLError atau TypeError dari dalam
    check_usage() membuat Gradio menampilkan toast merah tanpa isi, lalu user
    melihat UI yang diam persis seperti kasus `diterjemah 0`.
    """
    SETTINGS.provider = provider
    with _capture() as buf:
        try:
            key = tl.get_api_key(api_key, provider)
            client = tl.make_client(key, provider)
            if tl._is_router(provider):
                msg = (f"{provider}: **{tl.check_usage(client)}**. Anggaran balon "
                       f"{'AKTIF' if SETTINGS.balloon_budget else 'mati'} — teks dibuat "
                       "sependek balonnya sejak di sumber.")
            else:
                msg = (f"DeepL API: **{tl.check_usage(client)}** digunakan. "
                       "DeepL tidak menyensor konten apa pun.")
        except (RuntimeError, ImportError) as exc:
            msg = f"{_RED} **Gagal:** {exc}"
        except Exception:  # noqa: BLE001 - tombol probe tidak boleh diam
            msg = (f"{_RED} **Gagal tak terduga saat cek API.**\n\n"
                   f"```\n{traceback.format_exc()[-1500:]}\n```")
    log = buf.getvalue().strip()
    if log:
        msg += f"\n\n<details><summary>Log</summary>\n\n```\n{log[-2000:]}\n```\n\n</details>"
    return msg


def _run(files, provider: str, api_key: str, target_lang: str, style: str,
         balloon_budget: bool, debug: bool, font_file, out_format: str = "both",
         progress=gr.Progress()):
    """Handler tombol TRANSLATE — pembungkus yang menangkap log DAN exception.

    Isi sebenarnya ada di `_run_inner`. Dipisah supaya SATU tempat memegang dua
    jaminan yang berlaku untuk semua jalur keluar: (1) apa pun yang tercetak
    selama run masuk ke run.log dan ke accordion UI, (2) exception apa pun jadi
    banner traceback, bukan UI yang diam. Kalau logika ini ditempel di dalam
    _run_inner, tiap `return` awal harus mengulangnya sendiri.

    Selalu mengembalikan 7 nilai: gallery, rar, zip, tabel(+banner), json, log
    teks, log file.
    """
    with _capture() as buf:
        try:
            gallery, rar_path, zip_path, md, raw = _run_inner(
                files, provider, api_key, target_lang, style, balloon_budget,
                debug, font_file, out_format, progress,
            )
        except Exception:  # noqa: BLE001 - UI tidak boleh mati tanpa pesan
            tb = traceback.format_exc()
            print(tb)  # ikut ke buffer -> run.log, jadi tersimpan juga
            gallery, rar_path, zip_path, raw = None, None, None, None
            md = (
                f"> ## {_RED} PIPELINE BERHENTI KARENA ERROR\n"
                "> Ini bug, bukan kegagalan jaringan. Traceback lengkapnya:\n\n"
                f"```\n{tb[-3000:]}\n```\n\n"
                "Log lengkap ada di accordion di bawah dan bisa diunduh."
            )
    log_text = buf.getvalue()
    return gallery, rar_path, zip_path, md, raw, log_text, _write_log(log_text)


def _run_inner(files, provider: str, api_key: str, target_lang: str, style: str,
               balloon_budget: bool, debug: bool, font_file, out_format: str,
               progress):
    """Isi asli handler TRANSLATE. Return 5 nilai (tanpa bagian log)."""
    # RUN_NOTES hidup selama sesi Colab, jadi klik TRANSLATE kedua akan mewarisi
    # banner merah klik pertama kalau tidak dikosongkan. Dikosongkan DI SINI dan
    # bukan di process_batch, karena catatan "API key tidak terbaca" di bawah
    # terjadi sebelum process_batch dipanggil dan justru itu yang harus selamat.
    RUN_NOTES.clear()
    if not files:
        return None, None, None, "Belum ada gambar yang diunggah.", None

    SETTINGS.output_format = out_format
    SETTINGS.target_lang = target_lang
    SETTINGS.translation_style = style
    SETTINGS.provider = provider
    # Anggaran balon cuma berarti untuk router: DeepL tidak bisa diberi tahu
    # ukuran balon, jadi mengukurnya di sana hanya membakar 10 detik CPU.
    SETTINGS.balloon_budget = bool(balloon_budget) and tl._is_router(provider)
    # ALL CAPS hanya gaya lettering English; bahasa lain pakai huruf normal.
    SETTINGS.force_upper = target_lang == "English"

    if font_file:
        path = font_file if isinstance(font_file, str) else font_file.name
        typeset.set_user_font(path)

    if not typeset.FONT_USED:
        typeset.setup_fonts(verbose=False)

    paths = [f if isinstance(f, str) else f.name for f in files]
    try:
        key = tl.get_api_key(api_key, provider)
    except RuntimeError as exc:
        # Dulu ditelan tanpa jejak, dan inilah jalur yang menghasilkan "diterjemah
        # 0" paling sering: key tidak ada -> client None -> semua halaman keluar
        # berbahasa Jepang. Sekarang tercatat sebagai error supaya masuk banner.
        note("error", "app",
             f"API key tidak terbaca ({exc}) — jalan TANPA terjemahan, "
             "semua halaman keluar berbahasa Jepang")
        key = ""

    results, summary = pipeline.process_batch(paths, key or None, debug, progress,
                                              target_lang, style, reset_notes=False)
    pipeline.release_all()

    # Galeri ikut format terpilih; kalau "both", preview pakai PNG (lossless).
    gallery = [str(r.paths.get("png") or r.paths["jpg"]) for r in results]
    rows = [
        "| halaman | region | SFX dijaga | diterjemah | belum diterjemah | residu | overflow | catatan |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        rp = r.report
        n_err, n_warn = rp.get("error_count", 0), rp.get("warn_count", 0)
        # Kolom catatan menunjuk ke accordion, bukan memuat pesannya: pesan
        # aslinya bisa 200 karakter dan akan merusak lebar tabel.
        cat = " ".join(x for x in (f"{_RED}{n_err}" if n_err else "",
                                   f"{_YELLOW}{n_warn}" if n_warn else "") if x) or "-"
        belum = rp.get("untranslated_count", 0)
        rows.append(
            f"| {r.stem} | {rp['region_count']} | {len(rp['sfx_idx'])} | "
            f"{rp['translated_count']} | {f'**{belum}**' if belum else 0} | "
            f"{rp['residue_count']} | {rp['overflow_count']} | {cat} |"
        )
    rar_path, rar_note = _archive(
        [p for r in results for p in r.paths.values()],
        Path(summary["zip"]).with_name("manga_translated.rar"),
    )
    summary["rar"] = rar_path or f"gagal: {rar_note}"
    # Catatan disisipkan ke `rows` (yang sudah di-join) supaya tabelnya tetap utuh.
    if rar_path is None:
        rows[:0] = [f"> RAR tidak dibuat ({rar_note}) - pakai unduhan ZIP.", ""]
    # Banner diagnosa paling atas, di atas catatan RAR: sebab kegagalan harus
    # jadi hal pertama yang terbaca tanpa perlu men-scroll.
    rows[:0] = _diagnose(results, summary)

    # Konfirmasi GPU di UI: user melihat langsung bahwa proses jalan di T4.
    try:
        import torch as _torch
        _gpu = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "CPU (lambat)"
    except Exception:  # noqa: BLE001
        _gpu = "tidak terbaca"
    # font_used bisa None kalau setup_fonts() gagal total. Path(None) melempar
    # TypeError, dan dulu itu terjadi SETELAH semua halaman selesai diproses —
    # seluruh run hilang tanpa satu pesan pun. Jangan pernah kembalikan Path()
    # atas nilai yang boleh None.
    _font = summary.get("font_used")
    _font = Path(_font).name if _font else "GAGAL DIMUAT"
    md = (
        f"**GPU:** `{_gpu}` · **Penyedia:** `{summary.get('provider', '?')}` · "
        f"**Model:** `{summary['model']}` · **Font:** `{_font}`\n\n"
        + "\n".join(rows)
    )
    return (
        gallery, rar_path, summary["zip"], md,
        json.dumps(summary, ensure_ascii=False, indent=2),
    )


def build() -> gr.Blocks:
    with gr.Blocks(title="Manga Translator — Jepang ke semua bahasa") as demo:
        gr.Markdown(
            "# Manga Translator — Jepang ke semua bahasa\n"
            "Unggah halaman manga, pilih penyedia terjemahan dan bahasa tujuan, "
            "tekan **TRANSLATE**. SFX dibiarkan utuh, tidak ada penyensoran."
        )
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(
                    label="Halaman manga - bisa banyak sekaligus",
                    file_count="multiple",
                    file_types=["image"],
                    height=200,
                )
                provider = gr.Dropdown(
                    label="Penyedia terjemahan",
                    choices=PROVIDERS,
                    value=PROVIDER_DEFAULT,
                    info="LLM (freetokenfaucet): paham konteks halaman DAN ukuran "
                         "balon, GRATIS, mimo-v2.5-pro terukur 5-8 dtk per "
                         "halaman — pakai ini. "
                         "DeepL: cepat tapi tidak bisa diberi tahu ukuran balon. "
                         "Router gorouter: mutu bahasa paling rapi tapi memakai "
                         "kredit berbayar.",
                )
                api_key = gr.Textbox(
                    label="API key (kosongkan kalau sudah di Colab Secrets)",
                    type="password",
                    placeholder="FAUCET_API_KEY / DEEPL_API_KEY / ROUTER_API_KEY",
                    info="Disimpan di Colab Secrets sebagai FAUCET_API_KEY, "
                         "DEEPL_API_KEY, atau ROUTER_API_KEY sesuai penyedia — "
                         "jangan ditulis di kode.",
                )
                lang = gr.Dropdown(
                    label="Bahasa terjemahan (Jepang → ...)",
                    choices=LANGUAGES,
                    value="English",
                )
                style = gr.Dropdown(
                    label="Gaya terjemahan",
                    choices=list(TRANSLATION_STYLES.keys()),
                    value="Manga Natural",
                )
                with gr.Accordion("Opsi", open=False):
                    balloon_budget = gr.Checkbox(
                        label="Anggaran balon (penyedia LLM saja)",
                        value=True,
                        info="Kirim batas karakter tiap balon ke model, ukur ulang "
                             "jawabannya, minta perbaikan yang melanggar. +~10 dtk "
                             "CPU per halaman; ini yang menjaga teks tidak keluar "
                             "balon. Tidak berlaku untuk DeepL.",
                    )
                    out_format = gr.Radio(
                        label="Format output",
                        choices=[
                            ("PNG + JPG", "both"),
                            ("PNG saja (lossless)", "png"),
                            ("JPG saja (ringan)", "jpg"),
                        ],
                        value="both",
                    )
                    debug = gr.Checkbox(label="Debug mode (dump tahapan)", value=False)
                    font_file = gr.File(
                        label="Font sendiri (.ttf/.otf) — opsional",
                        file_types=[".ttf", ".otf"],
                    )
                    probe_btn = gr.Button("Cek API", size="sm")
                go = gr.Button("TRANSLATE", variant="primary", elem_id="go")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Hasil", columns=2, height=560)
                with gr.Row():
                    rar_out = gr.File(label="Unduh semua (RAR)")
                    zip_out = gr.File(label="Unduh semua (ZIP)")
        table = gr.Markdown()
        with gr.Accordion("Ringkasan JSON", open=False):
            raw = gr.Code(language="json")
        # Tertutup secara default: log mentah panjang dan tidak boleh mendorong
        # galeri keluar layar saat semuanya berjalan normal. Banner di atas tabel
        # yang memberi tahu kapan accordion ini perlu dibuka.
        with gr.Accordion("Log lengkap (buka kalau ada banner merah)", open=False):
            log_box = gr.Code(label="Keluaran mentah run terakhir")
            log_file = gr.File(label=f"Unduh {LOG_NAME}")

        go.click(
            _run,
            inputs=[files, provider, api_key, lang, style, balloon_budget,
                    debug, font_file, out_format],
            outputs=[gallery, rar_out, zip_out, table, raw, log_box, log_file],
        )
        probe_btn.click(_probe_table, inputs=[provider, api_key], outputs=[table])
    return demo


def launch(share: bool = True, debug: bool = False):
    """Di Colab share=True wajib: share=False + queueing -> ValueError."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SETTINGS.debug = debug
    demo = build().queue()
    try:
        return demo.launch(share=share, css=CSS, debug=False, show_error=True)
    except (ValueError, RuntimeError) as exc:
        # frpc sering 403; jatuh ke mode lokal supaya notebook tidak mati.
        print(f"[app] share gagal ({exc}); coba tanpa share")
        return demo.launch(share=False, css=CSS, show_error=True)

