#!/usr/bin/env python3
"""Penerjemah LLM dengan ANGGARAN BALON — jawaban atas "kan AI bisa disuruh?".

Bisa, tapi bukan dengan menyuruh "buatlah pendek". Percobaan pertama
(probe_llm.py) sudah menyuruh "PREFER THE SHORTER natural phrasing" dan
hasilnya tetap `SORRY TO BARGE IN!` (18 karakter) untuk balon yang cuma memuat
`SORRY.` (6). Model tidak melanggar perintah; model tidak PUNYA cara menaati
perintah yang tidak berisi angka. Ia tidak melihat balonnya.

Jadi yang dikirim di sini bukan selera melainkan hasil ukuran. Untuk tiap
balon, probe_budget.py mengukur lewat typeset.layout() yang SAMA dengan yang
merender:
    max_chars      berapa karakter yang muat pada ukuran font yang diinginkan
    max_word       kata terpanjang yang muat satu baris tanpa penggalan
    hard_chars     batas mutlak; melewatinya berarti font jatuh ke lantai darurat

Tiga lapis, karena satu lapis tidak cukup:
  1. PROMPT      — anggaran per baris ikut di dalam JSON masukan.
  2. VALIDASI    — jawabannya diukur lagi dengan layout() sungguhan, bukan
                   dipercaya. Model boleh mengaku patuh; yang menentukan mesin.
  3. PERBAIKAN   — hanya baris yang melanggar dikirim ulang, dengan angka
                   pelanggarannya disebut eksplisit. Baris yang sudah benar
                   TIDAK disentuh, supaya perbaikan tidak merusak yang lain.

Kredensial dibaca dari test.txt dan TIDAK PERNAH dicetak, bahkan sebagian.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
CRED = ROOT / "test.txt"
STAGE = ROOT / ".stage"
CACHE = ROOT / ".probe_cache.pkl"
_MAGIC = re.compile(r"^%%writefile\b.*\n?")
os.environ.setdefault("MANGATL_WORK", str(ROOT))
os.environ.setdefault("MANGATL_ROOT", str(STAGE))

STAGE.mkdir(exist_ok=True)
for _s in sorted((ROOT / "_nbsrc").glob("*.py")):
    _b = _MAGIC.sub("", _s.read_text(encoding="utf-8"), count=1)
    _d = STAGE / _s.name
    if not _d.exists() or _d.read_text(encoding="utf-8") != _b:
        _d.write_text(_b, encoding="utf-8")
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))

import typeset  # noqa: E402
import translate  # noqa: E402
from config import SETTINGS  # noqa: E402

# Plafon proporsional TIDAK dipakai sebagai kriteria lulus, dan itu keputusan
# hasil ukuran, bukan kelonggaran. probe_refcheck.py menjalankan wording
# REFERENSI melewati mask REFERENSI sendiri: tiga balon padat justru duduk di
# BAWAH plafonnya — r5 -6 px, r9 -4 px, r10 -5 px. Artinya typesetter
# profesional pun mengecilkan font di balon padat; plafon 0.117 itu MEDIAN
# (p25 0.108, p75 0.150), bukan lantai. Menjadikannya kriteria berarti menolak
# hasil yang justru ditiru — dan validator versi pertama memang menolaknya.
#
# Yang tersisa sebagai kriteria adalah dua cacat yang eksplisit di plan.txt dan
# tidak pernah dilanggar referensi:
#   feasible == 0                  -> kata tak muat, tanda hubung dipaksa
#   feasible < min_font_size        -> jatuh ke jalur darurat, tak terbaca
# Selisih ke plafon tetap DICETAK sebagai informasi, supaya terlihat balon mana
# yang sesak, tapi tidak memicu perbaikan.
_CAP_GAP_REPORT = 3  # selisih di atas ini disebut 'sesak' di laporan

SRC = {
    0: "あーっ！やっと見つけたっ", 1: "こんなとこに居たんですかっ",
    2: "探しましたよ", 3: "〈会長っ〉", 4: "あら", 5: "・雫さん．．．",
    6: "＼失礼しました", 7: "放課後はこちらが静かで落ち着くもので．．．",
    8: "何してたんです？こんな所で", 9: "性徒会の執行記録を作ってたんです",
    10: "この春からの活動まとめですね", 11: "あっシコ部のですかっ？",
    12: "すかっ？ちょっと見せてくださいよ〜っ",
}

REF = {
    0: "AH! FINALLY FOUND YOU!", 1: "SO THIS IS WHERE YOU WERE!",
    2: "I'VE BEEN LOOKING ALL OVER FOR YOU.", 3: "PREZ!", 4: "OH MY.",
    5: "SHIZUKU-SAN.", 6: "SORRY.",
    7: "IT'S JUST THAT IT'S QUIET AND RELAXING HERE AFTER SCHOOL...",
    8: "WHAT WERE YOU DOING IN A PLACE LIKE THIS?",
    9: "I WAS PUTTING TOGETHER THE STUDENT COUNCIL'S ACTIVITY RECORDS.",
    10: "A SUMMARY OF EVERYTHING WE'VE DONE SINCE THIS SPRING.",
    11: "OH, IS THAT FOR THE MILKING CLUB?", 12: "COME ON, LET ME SEE~!",
}

# Istilah yang TIDAK bisa disimpulkan model dari satu halaman, jadi diberikan.
# Bukan untuk menyensor pilihan model, tapi karena keduanya keliru secara
# faktual kalau dibaca sendiri-sendiri:
#   性徒会 — 生徒会 (dewan siswa) dengan 生 ditukar 性 (seks). Plesetan sengaja
#            pengarang, jadi terjemahannya harus TETAP terbaca sebagai dewan
#            siswa; kalau tidak, pembaca kehilangan rujukan ceritanya.
#   シコ部 — halaman lain seri ini menuliskannya 搾乳部 (搾乳 = memeras susu).
#            Tanpa petunjuk itu, シコ shortcut-nya ke シコシコ (masturbasi), dan
#            model memang memilih itu ('JERK-OFF CLUB'). Typeset referensi
#            memakai MILKING CLUB.
GLOSSARY = """Series glossary (use these renderings):
- 性徒会 = "student council" (the author swaps 生 for 性 as a running gag; keep it
  readable as the student council, do not translate it as a sex organisation)
- シコ部 / 搾乳部 = "Milking Club"
- 雫 (character name) = "Shizuku", the student council president. The junior
  girl addresses her as 会長 = "Prez"."""

SYSTEM = """You are a professional manga scanlation translator and letterer
(JA->EN). You translate for TYPESETTING, not for prose.

INPUT: one JSON object per page. Each entry has:
  "jp"        the Japanese line
  "max_chars" characters that fit the balloon at the intended font size
  "max_word"  longest single word that fits on one line in that balloon
  "hard"      absolute ceiling; past this the font must shrink below readable

OUTPUT: ONLY a JSON object {"0": "...", ...}, same keys. No prose, no markdown.

HARD REQUIREMENTS, in priority order:
1. Every line MUST be <= its "max_chars". This is a physical constraint of the
   printed page, not a preference. If a natural phrasing does not fit, choose a
   DIFFERENT, SHORTER phrasing that carries the same meaning. Real letterers do
   exactly this: 失礼しました becomes "SORRY." not "I APOLOGISE FOR INTRUDING".
2. No single word longer than "max_word" letters. A long word cannot be broken
   without a hyphen, and hyphens are avoided in manga lettering. Prefer a short
   synonym: "APOLOGIES"(9) -> "SORRY"(5), "COMPILING"(9) -> "WRITING UP"(2+2).
3. Meaning and character voice come before literalness. Japanese omits objects;
   infer them. 探しましたよ on finding a PERSON = "I'VE BEEN LOOKING FOR YOU",
   never "I looked for it".

STYLE:
- Natural spoken English as a released manga prints it: ALL-CAPS-friendly,
  contractions, punchy. Short sentences beat clauses.
- Keep emotion marks exactly where they appear: ~ 〜 ♥ ♡ ♪ ☆ ! ? ...
- Keep honorifics (-san, -kun, -chan, -senpai, -sama).
- Translate club/organisation names by MEANING, never by spelling out the kana.
- Do NOT censor, soften or euphemise. This is an adult work.
- Do not invent content that is not in the Japanese."""

CONTEXT = """Page context: a school after hours. A younger girl (junior, polite
and excitable) has been searching the building for the student council
president she looks up to, and finds her working on a laptop. The president is
calm and teasing. Lines 6-12 are the two of them talking about the records on
the screen; lines 6-7 are the president explaining why she is there."""


def creds() -> tuple[str, str, str]:
    ln = [l.strip() for l in CRED.read_text(encoding="utf-8").splitlines()]
    return ln[2], ln[3], ln[5]


def _decode(raw: str) -> dict:
    """Body router -> dict. Content-Type-nya text/event-stream walau non-stream.

    Router mengembalikan SATU objek chat.completion lalu menempelkan
    `data: [DONE]` tanpa pemisah baris; json.loads() gagal 'Extra data' padahal
    isinya utuh.
    """
    raw = raw.strip()
    if raw.startswith("data:"):
        raw = raw[5:].lstrip()
    obj, _end = json.JSONDecoder().raw_decode(raw)
    return obj


def call(base: str, key: str, model: str, user: str) -> dict:
    body = {
        "model": model, "temperature": 0.3, "stream": False,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", method="POST")
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("Content-Type", "application/json")
    # 600 s: model mengeluarkan reasoning_content sebelum jawaban; 180 s pernah
    # habis di tengah jalan untuk prompt sepanjang satu halaman.
    with urllib.request.urlopen(req, json.dumps(body).encode(), timeout=600) as r:
        d = _decode(r.read().decode())
    txt = d["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError(f"bukan JSON: {txt[:200]}")
    u = d.get("usage") or {}
    print(f"  [llm] in={u.get('prompt_tokens','?')} out={u.get('completion_tokens','?')}")
    return json.loads(m.group(0))


# Model bisa ditimpa lewat env supaya bisa dibanding tanpa menyentuh test.txt:
#   MODEL=gorouter/claude-opus-5 python probe_llm2.py
# test.txt tetap satu-satunya tempat KEY berada; ini hanya menimpa nama model.
_MODEL_ENV = os.environ.get("MODEL", "").strip()

# Urutan cadangan kalau model pilihan sedang 502 di sisi upstream. Ini BUKAN
# diam-diam mengganti pilihan: nama model yang benar-benar dipakai dicetak, dan
# urutannya dari yang paling dekat ke pilihan asli.
# Nama-nama model router lama DIHAPUS 17 Agu 2026: hostnya mati, dan model
# lamanya di faucet sekarang membalas HTTP 402 (berbayar, saldo akun 0). Yang
# tinggal cuma model yang terukur bekerja hari itu.
_FALLBACK = ("gorouter/claude-opus-5",)

# 502 dari router ini SEMENTARA, bukan tanda model tidak ada. Terukur: dalam satu
# menit satu model membalas 502 tiga kali lalu 200 dalam 4 s, sementara model
# lain justru sebaliknya — dan `/models` tetap 200 sepanjang waktu. Jadi
# berpindah model saja tidak cukup; yang menolong adalah MENCOBA LAGI. Tanpa ini
# probe mati oleh gangguan pihak lain dan hasilnya tampak seperti kegagalan kita.
_RETRY = 4
_BACKOFF = 8  # detik, dikali nomor percobaan


def call_any(base: str, key: str, model: str, user: str) -> tuple[dict, str]:
    """call() dengan percobaan ulang + cadangan model. Return (jawaban, model)."""
    order = (model, *(f for f in _FALLBACK if f != model))
    tried: list[str] = []
    for attempt in range(1, _RETRY + 1):
        for m in order:
            short = m.rsplit("/", 1)[-1]
            try:
                return call(base, key, m, user), m
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504):
                    raise
                tried.append(f"{short}={e.code}")
                print(f"  [skip] {short} -> HTTP {e.code}")
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                # ValueError = jawaban bukan JSON. Itu juga layak dicoba ulang:
                # penyebabnya biasanya jawaban terpotong, bukan prompt yang salah.
                tried.append(f"{short}={type(e).__name__}")
                print(f"  [skip] {short} -> {type(e).__name__}")
        if attempt < _RETRY:
            wait = _BACKOFF * attempt
            print(f"  [retry] semua model gagal di percobaan {attempt}; "
                  f"tunggu {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"semua model gagal {_RETRY}x: {', '.join(tried[-6:])}")


def _longest_word(s: str) -> int:
    """Panjang kata terpanjang, punctuation ekor tidak dihitung.

    'SORRY.' yang menyesakkan itu hurufnya; titik jauh lebih sempit daripada
    huruf, dan menghitungnya membuat validasi menolak wording yang sebenarnya
    muat.
    """
    w = [re.sub(r"[^A-Z']", "", x) for x in s.upper().split()]
    return max((len(x) for x in w), default=0)


def render_form(t: str) -> str:
    """Teks sebagaimana typeset akan MENERIMANYA, bukan sebagaimana model menulisnya.

    Wajib ada, dan ini ketemu dari validasi yang salah menuduh: model membalas
    '＼SORRY.' untuk r6 dan validator melaporkan 'tidak muat bahkan di ukuran
    font minimum (7 karakter, balon memuat ~39)' — mustahil, dan memang salah.
    Penyebabnya '＼', yang tidak punya glyph di Anime Ace sehingga lebarnya
    ngawur. Padahal pipeline TIDAK PERNAH merendernya: translate._PUNCT_MAP
    membuangnya (bersama 〈 〉 ・ 「 」) di _clean_translation, tepat sebelum
    typeset. Jadi yang divalidasi harus bentuk setelah pembersihan itu — kalau
    tidak, probe ini menghukum model atas glyph yang sudah ditangani orang lain.
    """
    return translate._clean_translation(t or "").upper()


def violations(texts: dict[int, str], budget: dict[str, dict],
               regions, font_path: str) -> dict[int, list[str]]:
    """Ukur jawaban model dengan mesin tata letak sungguhan. Lapis penentu.

    Anggaran karakter itu PROKSI — dihitung dengan teks pengisi, bukan dengan
    kalimat yang benar-benar dipakai. Yang mengikat cuma satu: apakah kalimat
    ini muat UTUH di balon ini pada ukuran yang masih terbaca. Jadi validasinya
    memanggil typeset._max_feasible() atas teks aslinya, dan anggaran hanya
    dipakai untuk MEMBERI TAHU model harus sependek apa.

    Pelanggaran = teks tidak muat utuh di atas min_font_size. Itu saja, dan
    alasannya di komentar _CAP_GAP_REPORT: kriteria yang lebih ketat (harus
    duduk di plafon proporsional) ditolak oleh ukuran — wording referensi
    sendiri melanggarnya di r5/r9/r10 pada mask referensi.
    """
    rmap = {r.idx: r for r in regions}
    bad: dict[int, list[str]] = {}
    for i, t in sorted(texts.items()):
        r = rmap.get(i)
        if r is None:
            continue
        d = budget[str(i)]
        mask = typeset._region_box_mask(r)[1]
        up = render_form(t)
        feas = typeset._max_feasible(up, mask, font_path)
        if feas < SETTINGS.min_font_size:
            bad[i] = [
                f"{len(up)} chars do not fit without a hyphen at any readable "
                f"size. This balloon holds about {d['soft']} characters and no "
                f"single word longer than {d['word_hard']} letters. Rewrite much "
                f"shorter, same meaning."
            ]
    return bad


def prompt_for(idxs, budget: dict[str, dict], extra: str = "") -> str:
    """Susun user-message: konteks + glosari + JSON berisi jp DAN anggarannya."""
    payload = {}
    for i in idxs:
        d = budget[str(i)]
        payload[str(i)] = {
            "jp": SRC[i],
            "max_chars": d["soft"],
            "max_word": d["word_soft"],
            "hard": d["hard"],
        }
    return (CONTEXT + "\n\n" + GLOSSARY + extra + "\n\nLines:\n"
            + json.dumps(payload, ensure_ascii=False, indent=1))


def main() -> int:
    typeset.setup_fonts(verbose=False)
    fp = typeset.FONT_USED
    with CACHE.open("rb") as f:
        regions = pickle.load(f)
    budget = json.loads((ROOT / "probe_budget.json").read_text(encoding="utf-8"))
    base, key, model = creds()
    if _MODEL_ENV:
        print(f"[model] env MODEL menimpa test.txt: {model} -> {_MODEL_ENV}")
        model = _MODEL_ENV
    print(f"[router] {base}  model={model}  (key tidak dicetak)")

    # Kalibrasi validator SEBELUM satu token pun dibelanjakan: jalankan wording
    # REFERENSI melewati kriteria yang sama. Referensi harus lolos — kalau tidak,
    # yang salah validatornya, dan setiap 'perbaikan' yang diminta ke model cuma
    # akan menjauhkan hasil dari target.
    #
    # Diuji pada MASK REFERENSI, bukan mask kita. Sebabnya terukur: interior kita
    # rata-rata 8.6 px lebih kecil daripada interior referensi pada skala yang
    # sama (probe_scale.py; r10 -10.3 px), dan kedua halaman bahkan bukan resize
    # satu sama lain — rasio lebar 0.886 vs rasio tinggi 0.870. Menuntut kalimat
    # referensi muat di mask kita berarti menuntut 53 karakter masuk ke balon 9%
    # lebih sempit daripada balon yang dipakai typesetter aslinya; itu menguji
    # kapasitas balon, bukan kesehatan kriteria.
    ref_pkl = ROOT / ".probe_ref_native.pkl"
    if ref_pkl.exists():
        with ref_pkl.open("rb") as f:
            rregions = pickle.load(f)
        ref_budget = {str(r.idx): {
            "cap": typeset.region_font_cap(typeset._region_box_mask(r)[1]),
            "soft": 0, "hard": 0, "word_soft": 0, "word_hard": 0,
        } for r in rregions}
        ref_bad = violations(dict(REF), ref_budget, rregions, fp)
        print(f"[kalibrasi] wording referensi (di mask referensi) lolos: "
              f"{'YA' if not ref_bad else 'TIDAK -> ' + str(sorted(ref_bad))}")
        for i, why in sorted(ref_bad.items()):
            print(f"   r{i}: {REF[i]!r}")
            for w in why:
                print(f"        - {w}")
        if ref_bad:
            print("  Validator menolak typeset profesional yang jadi target; "
                  "perbaiki kriterianya dulu, jangan panggil model.")
            return 1
    else:
        print("[kalibrasi] .probe_ref_native.pkl tidak ada -> dilewati")

    print("\n== pass 1: seluruh halaman, dengan anggaran per balon")
    try:
        got, used = call_any(base, key, model, prompt_for(sorted(SRC), budget))
    except (urllib.error.HTTPError, RuntimeError) as e:
        print(f"gagal: {e}")
        return 1
    if used != model:
        print(f"  [model] {model.rsplit('/', 1)[-1]} tidak tersedia -> "
              f"{used.rsplit('/', 1)[-1]}")
    got = {int(k): str(v) for k, v in got.items()}

    for rnd in range(1, 4):
        bad = violations(got, budget, regions, fp)
        if not bad:
            print(f"\n== semua baris lolos validasi layout() (setelah {rnd - 1} perbaikan)")
            break
        print(f"\n== pass {rnd + 1}: perbaiki {len(bad)} baris {sorted(bad)}")
        for i, why in sorted(bad.items()):
            print(f"   r{i}: {got[i]!r}")
            for w in why:
                print(f"        - {w}")
        # Hanya baris yang melanggar dikirim ulang. Yang sudah benar tidak
        # disentuh: mengirim ulang seluruh halaman membuat model 'memperbaiki'
        # baris yang tidak diminta dan hasil yang sudah lolos bisa rusak.
        extra = ("\n\nREVISION. Your previous attempt broke the balloon budget on "
                 "these lines. Rewrite ONLY these, shorter, same meaning:\n"
                 + "\n".join(f'  "{i}": you wrote {got[i]!r} -> ' + "; ".join(why)
                             for i, why in sorted(bad.items())))
        try:
            fix, _u = call_any(base, key, used,
                               prompt_for(sorted(bad), budget, extra))
        except (urllib.error.HTTPError, RuntimeError) as e:
            print(f"gagal: {e}")
            break
        for k, v in fix.items():
            if int(k) in bad:
                got[int(k)] = str(v)
    else:
        print("\n== masih ada pelanggaran setelah 3 perbaikan")

    print(f"\n{'idx':>3} {'plafon':>6} {'muat':>4}  hasil / referensi")
    tight = []
    for i in sorted(SRC):
        d = budget[str(i)]
        mask = typeset._region_box_mask({r.idx: r for r in regions}[i])[1]
        up = render_form(got.get(i, ""))
        feas = typeset._max_feasible(up, mask, fp)
        if feas < SETTINGS.min_font_size:
            mark = "GAGAL"
        elif d["cap"] - feas > _CAP_GAP_REPORT:
            mark = "sesak"
            tight.append(i)
        else:
            mark = "ok"
        print(f"{i:>3} {d['cap']:>6} {feas:>4} {mark:>5}  {up}")
        print(f"                     ref: {REF.get(i, '')}")
    print(f"\nbalon sesak (font > {_CAP_GAP_REPORT} px di bawah plafon): {tight}")
    print("  Sesak BUKAN kegagalan — referensi pun sesak di r5/r9/r10 "
          "(probe_refcheck.py). Ini cuma penanda balon mana yang paling padat.")

    # DUA file, dan pemisahannya perlu:
    #   *_raw.json     jawaban model apa adanya — bukti apa yang benar-benar
    #                  dikembalikan, termasuk 〈 〉 dan ＼ yang masih ikut.
    #   *_texts.json   bentuk siap-typeset (sudah lewat _clean_translation),
    #                  karena probe_font.py memasang teks LANGSUNG ke
    #                  r.translation dan tidak memanggil pembersih itu. Tanpa
    #                  pemisahan ini, uji typeset akan mengukur lebar glyph tofu
    #                  yang pipeline sungguhan tidak pernah render.
    raw = {str(k): v for k, v in sorted(got.items())}
    clean = {str(k): render_form(v) for k, v in sorted(got.items())}
    tag = re.sub(r"[^a-z0-9]+", "-", used.lower()).strip("-")
    for name, data in ((f"probe_llm2_{tag}_raw.json", raw),
                       (f"probe_llm2_{tag}.json", clean),
                       ("probe_llm2_texts.json", clean)):
        (ROOT / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> probe_llm2_{tag}_raw.json (mentah) + probe_llm2_{tag}.json (bersih)")
    print(f"   TEXTS=probe_llm2_{tag}.json python probe_font.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
