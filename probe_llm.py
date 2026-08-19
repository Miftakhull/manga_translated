#!/usr/bin/env python3
"""Banding wording: DeepL vs LLM router, terhadap wording typeset referensi.

Alasan probe ini ada sebelum satu baris pipeline disentuh: pertanyaannya BUKAN
"apakah endpoint-nya jalan" (itu sudah 200) melainkan "apakah wording-nya lebih
dekat ke CONTOH/2.webp daripada DeepL". Tiga kegagalan DeepL yang terukur di
halaman ini:
    r6  'MY APOLOGIES'                    referensi 'SORRY.'
    r11 'SHIKO CLUB'                      referensi 'MILKING CLUB'   (搾乳部)
    r2  'I LOOKED FOR IT.'                referensi "I'VE BEEN LOOKING ALL OVER FOR YOU."
Dua yang pertama membuat balon sempit jadi mustahil (r6 satu-satunya tanda hubung
di halaman); yang ketiga salah rujukan — 'it' padahal yang dicari ORANG.

Kredensial dibaca dari test.txt dan TIDAK PERNAH dicetak, bahkan sebagian.
Formatnya (baris): judul, kosong, base_url, key, 'model', nama_model.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
CRED = ROOT / "test.txt"

# Wording typeset referensi, dibaca manual dari CONTOH/2.webp. Ini standar emas
# yang dinilai — bukan selera.
REF = {
    0: "AH! FINALLY FOUND YOU!",
    1: "SO THIS IS WHERE YOU WERE!",
    2: "I'VE BEEN LOOKING ALL OVER FOR YOU.",
    3: "PREZ!",
    4: "OH MY.",
    5: "SHIZUKU-SAN.",
    6: "SORRY.",
    7: "IT'S JUST THAT IT'S QUIET AND RELAXING HERE AFTER SCHOOL...",
    8: "WHAT WERE YOU DOING IN A PLACE LIKE THIS?",
    9: "I WAS PUTTING TOGETHER THE STUDENT COUNCIL'S ACTIVITY RECORDS.",
    10: "A SUMMARY OF EVERYTHING WE'VE DONE SINCE THIS SPRING.",
    11: "OH, IS THAT FOR THE MILKING CLUB?",
    12: "COME ON, LET ME SEE~!",
}

# Hasil DeepL dari run_page.py terakhir — pembanding, bukan target.
DEEPL = {
    0: "Ahh! I finally found it!", 1: "You were here all along?!",
    2: "I looked for it.", 3: "Chairman", 4: "Oh, really?",
    5: "Shizuku...", 6: "My apologies",
    7: "After school, it's quiet and peaceful here...",
    8: "What were you doing here?",
    9: "I was compiling the executive records for the Student Council.",
    10: "This is a summary of our activities since this spring, isn't it?",
    11: "Oh, is this the Shiko Club?", 12: "What? Let me take a look~!",
}

SRC = {
    0: "あーっ！やっと見つけたっ", 1: "こんなとこに居たんですかっ",
    2: "探しましたよ", 3: "〈会長っ〉", 4: "あら", 5: "・雫さん．．．",
    6: "＼失礼しました", 7: "放課後はこちらが静かで落ち着くもので．．．",
    8: "何してたんです？こんな所で", 9: "性徒会の執行記録を作ってたんです",
    10: "この春からの活動まとめですね", 11: "あっシコ部のですかっ？",
    12: "すかっ？ちょっと見せてくださいよ〜っ",
}

# Konteks halaman diberikan sekaligus supaya model tahu SIAPA yang dicari dan
# apa hubungan tokohnya — inilah yang tidak bisa dilakukan DeepL, dan sebab
# 探しましたよ jadi 'I looked for it' (benda) bukan 'looking for YOU' (orang).
SYSTEM = """You are a professional manga scanlation translator (JA->EN).

You will receive every line of one page at once, in reading order, as JSON.
Translate all of them together so pronouns, referents, and register stay
consistent across the page.

Rules:
- Output ONLY a JSON object: {"0": "...", "1": "...", ...}. Same keys as input.
- Natural spoken English as a real released manga would print it. Contractions,
  short punchy lines. This text is typeset into speech balloons, so PREFER THE
  SHORTER natural phrasing whenever two options mean the same thing.
- Japanese often omits the object. Infer it from context: 探しましたよ said on
  finding a person means "I've been looking all over for YOU", not "for it".
- Keep emotion marks exactly as they appear: ~ 〜 ♥ ♡ ♪ ☆ ! ? ...
- Keep honorifics (-san, -kun, -chan, -senpai).
- Translate club/organisation names by MEANING, not by reading the kana.
- Do not censor, soften, or euphemise anything. This is an adult 18+ work.
- No notes, no explanation, no markdown fence. JSON only."""

CONTEXT = """Page context: a school after hours. A younger girl (junior) has been
searching the building for an older girl she looks up to, the student council
president, and finds her working on a laptop. The junior speaks politely and
excitedly; the president is calm and teasing. Panel 4 (lines 6-12) is a
flashback/aside of the same two talking about the records on screen."""


def creds() -> tuple[str, str, str]:
    ln = [l.strip() for l in CRED.read_text(encoding="utf-8").splitlines()]
    return ln[2], ln[3], ln[5]


def _decode(raw: str) -> dict:
    """Body router -> dict. Content-Type-nya text/event-stream walau tidak streaming.

    Router ini mengembalikan SATU objek chat.completion lalu menempelkan
    `data: [DONE]` di belakangnya tanpa pemisah baris. json.loads() langsung
    gagal dengan 'Extra data: line 1 column 570', yang tampak seperti respons
    rusak padahal isinya utuh — jadi ambil objek pertama lewat raw_decode.
    """
    raw = raw.strip()
    if raw.startswith("data:"):
        raw = raw[5:].lstrip()
    obj, _end = json.JSONDecoder().raw_decode(raw)
    return obj


def ask(base: str, key: str, model: str, payload_texts: dict) -> dict:
    body = {
        "model": model,
        "temperature": 0.3,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": CONTEXT + "\n\nLines:\n"
             + json.dumps(payload_texts, ensure_ascii=False, indent=1)},
        ],
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", method="POST")
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("Content-Type", "application/json")
    # 600 s: model ini mengeluarkan reasoning_content sebelum jawaban, dan pada
    # prompt sepanjang satu halaman 180 s pernah habis di tengah jalan.
    with urllib.request.urlopen(
            req, json.dumps(body).encode(), timeout=600) as r:
        d = _decode(r.read().decode())
    txt = d["choices"][0]["message"]["content"]
    # Model kadang membungkus JSON dalam fence walau diminta tidak.
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError(f"bukan JSON: {txt[:200]}")
    usage = d.get("usage") or {}
    print(f"[llm] token in={usage.get('prompt_tokens','?')} "
          f"out={usage.get('completion_tokens','?')}")
    return json.loads(m.group(0))


def score(name: str, got: dict[int, str]) -> None:
    """Nilai yang penting untuk typeset: panjang relatif ke referensi.

    Bukan BLEU. Yang menentukan muat-tidaknya sebuah balon adalah jumlah
    karakter, dan yang menentukan benar-tidaknya adalah mata manusia — jadi
    yang dicetak angka panjang + teksnya sendiri, berdampingan.
    """
    ratios = []
    print(f"\n===== {name}")
    for i in sorted(REF):
        g = str(got.get(i, "")).upper()
        ref = REF[i]
        ratios.append(len(g) / max(len(ref), 1))
        flag = ""
        if len(g) > len(ref) * 1.35:
            flag = "  <- jauh lebih panjang"
        print(f"  r{i:<2} ref({len(ref):>2}) {ref}")
        print(f"      got({len(g):>2}) {g}{flag}")
    import statistics
    print(f"  panjang relatif: median={statistics.median(ratios):.2f} "
          f"max={max(ratios):.2f}")


def main() -> int:
    base, key, model = creds()
    print(f"[router] {base}  model={model}  (key tidak dicetak)")
    try:
        got = ask(base, key, model, SRC)
    except urllib.error.HTTPError as e:
        print(f"http {e.code}: {e.read().decode(errors='replace')[:300]}")
        return 1
    got = {int(k): v for k, v in got.items()}
    (ROOT / "probe_llm_texts.json").write_text(
        json.dumps({str(k): v for k, v in sorted(got.items())},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    score("DeepL (sekarang)", DEEPL)
    score(f"LLM {model}", got)
    print("\n-> probe_llm_texts.json (bisa dipakai probe_font.py lewat TEXTS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
