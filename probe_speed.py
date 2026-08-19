#!/usr/bin/env python3
"""Berapa DETIK sebenarnya satu panggilan terjemahan, dan ke mana waktunya pergi.

Pertanyaan yang dijawab: "aku ingin cepat seperti DeepL tapi tetap pakai
anggaran balon — bisa disuruh jawab terjemahan saja?"

Yang diukur, bukan ditebak:
  1. waktu tembok satu panggilan seluruh halaman (13 balon)
  2. token keluar yang BERGUNA (jawaban) vs token REASONING yang dibuang
  3. apakah meminta "jawab JSON saja, tanpa berpikir" benar-benar mempercepat

Reasoning itu penting diukur terpisah karena itulah tersangka utama: laporan
probe_llm2.py mencatat out=113 token untuk seluruh halaman — jawaban sekecil itu
mustahil butuh menit. Kalau waktunya habis di reasoning yang tidak pernah
dipakai, maka melarangnya adalah percepatan gratis; kalau tidak, melarangnya
cuma menurunkan mutu tanpa imbalan.

Tiga varian dibandingkan pada prompt yang SAMA:
  A  apa adanya            — SYSTEM + anggaran balon, seperti probe_llm2.py
  B  larangan reasoning    — SYSTEM + "answer with the JSON only, do not think"
  C  tanpa anggaran balon  — pembanding, untuk membuktikan anggaran bukan
                             penyebab lambat (prompt lebih pendek, model sama)

Kredensial dibaca dari test.txt dan TIDAK PERNAH dicetak.

    python probe_speed.py            # varian A B C, satu panggilan masing-masing
    MODEL=gorouter/claude-opus-5 python probe_speed.py
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

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

from probe_llm2 import CONTEXT, GLOSSARY, SRC, SYSTEM, creds  # noqa: E402

# Larangan reasoning. Ditulis sebagai perintah keluaran, bukan permintaan halus:
# "be concise" tidak pernah menghentikan reasoning, karena reasoning bukan
# bagian dari jawaban yang model anggap sedang diminta pendek.
NO_THINK = """

OUTPUT DISCIPLINE (critical): reply with the JSON object and NOTHING else. Do
not think step by step, do not deliberate, do not draft alternatives, do not
explain. Emit the JSON immediately as your first and only output. You are a
fast lookup, not an essayist."""


def prompt_full() -> str:
    """Prompt dengan anggaran balon — sama seperti probe_llm2.prompt_for()."""
    budget = json.loads((ROOT / "probe_budget.json").read_text(encoding="utf-8"))
    payload = {}
    for i in sorted(SRC):
        d = budget[str(i)]
        payload[str(i)] = {"jp": SRC[i], "max_chars": d["soft"],
                           "max_word": d["word_soft"], "hard": d["hard"]}
    return (CONTEXT + "\n\n" + GLOSSARY + "\n\nLines:\n"
            + json.dumps(payload, ensure_ascii=False, indent=1))


def prompt_plain() -> str:
    """Prompt tanpa anggaran — hanya kalimat Jepangnya."""
    payload = {str(i): SRC[i] for i in sorted(SRC)}
    return (CONTEXT + "\n\n" + GLOSSARY + "\n\nLines:\n"
            + json.dumps(payload, ensure_ascii=False, indent=1))


def timed(base: str, key: str, model: str, system: str, user: str) -> dict:
    body = {"model": model, "temperature": 0.3, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 method="POST")
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("Content-Type", "application/json")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, json.dumps(body).encode(), timeout=900) as r:
        raw = r.read().decode()
    dt = time.monotonic() - t0
    raw = raw.strip()
    if raw.startswith("data:"):
        raw = raw[5:].lstrip()
    d, _ = json.JSONDecoder().raw_decode(raw)
    msg = d["choices"][0]["message"]
    txt = msg.get("content") or ""
    think = msg.get("reasoning_content") or msg.get("reasoning") or ""
    u = d.get("usage") or {}
    det = u.get("completion_tokens_details") or {}
    m = re.search(r"\{.*\}", txt, re.S)
    return {
        "detik": dt,
        "in": u.get("prompt_tokens"),
        "out": u.get("completion_tokens"),
        "reason_tok": det.get("reasoning_tokens"),
        "reason_char": len(think),
        "jawab_char": len(txt),
        "n": len(json.loads(m.group(0))) if m else 0,
        "sample": (json.loads(m.group(0)).get("6", "") if m else ""),
    }


def main() -> int:
    base, key, model = creds()
    model = os.environ.get("MODEL", "").strip() or model
    reps = int(os.environ.get("N", "1"))
    only = os.environ.get("ONLY", "").strip().upper()
    print(f"[router] model={model}  N={reps}  (key tidak dicetak)")
    full, plain = prompt_full(), prompt_plain()
    cases = (
        ("A anggaran balon, apa adanya", SYSTEM, full),
        ("B anggaran balon + larang reasoning", SYSTEM + NO_THINK, full),
        ("C tanpa anggaran (pembanding)", SYSTEM, plain),
    )
    rows = []
    for name, sysmsg, user in cases:
        if only and not name.startswith(only):
            continue
        print(f"\n== {name}")
        # Diulang N kali karena satu sampel tidak bisa membedakan pengaruh prompt
        # dari ributnya jaringan. Tanpa ini, selisih 14.9 s vs 28.0 s terbaca
        # seolah larangan reasoning MEMPERLAMBAT 88% — padahal larangan itu
        # mustahil punya jalur sebab ke sana, dan reasoning_char=0 di ketiga
        # varian membuktikan model ini tidak mengeluarkan reasoning sama sekali.
        for k in range(reps):
            try:
                r = timed(base, key, model, sysmsg, user)
            except (urllib.error.HTTPError, urllib.error.URLError,
                    TimeoutError, ValueError) as e:
                print(f"   [{k + 1}] GAGAL {type(e).__name__}: {str(e)[:160]}")
                continue
            rows.append((name, r))
            print(f"   [{k + 1}] {r['detik']:.1f} s | in={r['in']} out={r['out']} "
                  f"reasoning_char={r['reason_char']} | balon={r['n']} "
                  f"r6={r['sample']!r}")
    print("\n== ringkas per varian")
    for name, _s, _u in cases:
        ds = [r["detik"] for n, r in rows if n == name]
        if not ds:
            continue
        print(f"   {name:<38} n={len(ds)} min={min(ds):.1f} "
              f"median={sorted(ds)[len(ds) // 2]:.1f} maks={max(ds):.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
