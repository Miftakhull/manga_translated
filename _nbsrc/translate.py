%%writefile /content/mangatl/translate.py

"""Terjemahan: DeepL (MT murni) ATAU LLM OpenAI-compatible (faucet / gorouter).

Ketiganya ada dan dipilih di UI, karena menang di hal yang berbeda:

  DeepL       cepat, gratis 1 juta karakter/bulan, tidak pernah menyensor —
              tapi menerjemahkan kalimat LEPAS KONTEKS dan tidak bisa diberi
              tahu apa pun tentang halamannya. Panjang hasilnya kebetulan.
  LLM faucet  freetokenfaucet, mimo-v2.5-pro (GRATIS). Bisa diberi konteks
              halaman, glosari, DAN ukuran balon — jalan ke syarat 'NO KELUAR
              BUBBLE' di plan.txt: teksnya dibuat pendek di sumbernya, bukan
              dikecilkan di typeset. TERUKUR 4.5-7.8 s untuk halaman 19 balon.
              Tokennya terbatas, jadi thinking dimatikan (lihat FAUCET_EXTRA).
              Modelnya WAJIB model gratis — lihat catatan 402 di config.py.
  LLM router  gorouter/claude-opus-5. Sama protokolnya, mutu bahasanya paling
              rapi (TERUKUR 8.2 s), tapi memakai kredit berbayar — jadi bukan
              default. Host-nya di balik Cloudflare dan MENUNTUT header
              User-Agent; tanpa itu 403 "error code 1010" (lihat BROWSER_UA).

Faucet dan router memakai kelas client yang sama (FaucetClient turunan
RouterClient) karena protokolnya identik — yang beda cuma base URL, model,
batas waktu, header, dan satu parameter body.

Klasifikasi SFX heuristik (_fallback_labels) dipakai untuk SEMUA penyedia —
SFX tidak boleh diterjemah, dan itu keputusan yang tidak bergantung penyedia.

API key: FAUCET_API_KEY / DEEPL_API_KEY / ROUTER_API_KEY (Colab Secrets -> env
-> field UI -> file lokal untuk penyedia LLM). Tidak ada key di kode.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from config import (BROWSER_UA, DEEPL_API_BASE, DEEPL_TARGET, FAUCET_API_BASE,
                    FAUCET_DEADLINE, FAUCET_EXTRA, FAUCET_FALLBACK,
                    FAUCET_MAX_TOKENS, FAUCET_MODEL, FAUCET_TIMEOUT,
                    PROTECTED_LABELS,
                    ROUTER_API_BASE, ROUTER_BACKOFF, ROUTER_DEADLINE,
                    ROUTER_FALLBACK, ROUTER_MODEL, ROUTER_RETRY, ROUTER_TIMEOUT,
                    SETTINGS, TRANSLATION_STYLES, Region, note)

_VALID_LABELS = frozenset(
    {"DIALOGUE", "THOUGHT", "NARRATION", "SIGN", "SFX", "UNREADABLE"}
)

# Gaya -> parameter formality DeepL (hanya dipakai untuk bahasa yang
# mendukungnya: DE/FR/ES/IT/NL/PL/PT/RU). Uncensored = default: DeepL
# memang tidak pernah menyensor.
FORMALITY_BY_STYLE: dict[str, str] = {
    "Formal": "more",
    "Casual & Slang": "less",
    "Manga Natural": "default",
    "Literal": "default",
    "Uncensored": "default",
    "Fully Localized": "default",
}

# DeepL tidak menerima request kosong; batch maksimal 50 teks per request.
_CHUNK = 50

# ---------------------------------------------------------------- SFX heuristic
#
# DeepL murni mesin terjemahan — dia tidak tahu mana SFX mana dialog, jadi
# klasifikasi 100% heuristik. Heuristik di sini jauh lebih agresif dari versi
# lama (kana murni <=3 char / ABAB 4 char) yang bocor di kasus nyata:
#   "フー．．．"   -> kana + tanda baca: tidak tertangkap, diterjemah "Phew..."
#   "ぴくぴくっ"  -> ABAB + っ ekor: tidak tertangkap, jadi "Twitch..."
#   "ドキドキドキ"-> pengulangan 6 char: tidak tertangkap, jadi "Thump..."
#
# Aturan baru:
#   inti kana  = sumber minus semua simbol/tanda baca (．。、・！？〜…♥ dll)
#   di luar balon: pendek(<=3) / ada っッ/ー / pola ulang / kamus -> SFX
#   di dalam balon: pola ulang / kamus / seluruhnya-katakana ber-っー /
#                   batang-ganda ada di _SFX_DICT -> SFX (lihat _sfx_in_bubble;
#                   aturan '3-karakter-っッ' yang lama sudah DICABUT karena
#                   mengunci seruan dialog dan mencetak balon Jepang)
#                   KECUALI teks aslinya bertanda BICARA (？ atau jeda di
#                   ANTARA kana) — itu suara tokoh, bukan bunyi latar:
#                   ヒ．．．ッ！？ di balon hitam bergerigi (hasilnew5)

_KANA = re.compile(r"^[\u3040-\u30ff\u31f0-\u31ff]+$")
_LONG = "ー"
_SMALL = frozenset("っッ")

# Tanda BICARA. Dipakai HANYA untuk menahan aturan K (lihat _sfx_in_bubble):
# bunyi tidak bertanya, dan bunyi tidak terputus di tengah mora.
# _PAUSE sengaja TIDAK memuat ・ (U+30FB) dan ー (U+30FC) walau keduanya di blok
# kana — ・ pemisah, ー pemanjang; tidak satu pun jeda ucapan.
_ASK = frozenset("？?")
_PAUSE = frozenset("．.。、,，…‥")

# Onomatope panjang / silang-rima yang tidak tertangkap pola ABAB.
_SFX_DICT = frozenset({
    # rangsangan / fisik
    "ガッタンゴットン", "がったんごっとん", "がたんごとん", "ガタンゴトン",
    "ドクンドクン", "どくんどくん", "ドキドキドキ", "どきどきどき",
    # Bentuk DASAR ganda dua suku. Sebelumnya hanya bentuk 3-ulangan
    # (ドキドキドキ) dan turunannya (ぴくぴく/ひくひく) yang tercatat, sehingga
    # aturan batang-ganda di _label_region tidak bisa membuktikan どきっ dan
    # びくっ sebagai onomatope — keduanya lolos jadi DIALOGUE dan ikut
    # diterjemah. _label3.py mengukur itu: keduanya satu-satunya sisa
    # kesalahan arah SFX sebelum empat entri ini ditambahkan.
    "ドキドキ", "どきどき", "ビクビク", "びくびく",
    "バクバク", "ばくばく", "はあはあ", "はぁはぁ", "ハァハァ", "ふうふう",
    "ずるずる", "じゅるじゅる", "チュパチュパ", "ちゅぱちゅぱ", "ぢゅぱぢゅぱ",
    "くちゅくちゅ", "ぐちゅぐちゅ", "にゅるにゅる", "ぬるぬる", "とろとろ",
    "ぐちょぐちょ", "びちゃびちゃ", "じゅくじゅく", "ぬちゃぬちゃ",
    "ぷるぷる", "ぶるぶる", "がくがく", "がたがた", "わなわな", "ぞくぞく",
    "そわそわ", "もじもじ", "うずうず", "むずむず", "はらはら",
    "むらむら", "わくわく", "ぽかぽか", "ほかほか",
    "ごくごく", "ごくん", "ごくり", "ごくりっ", "ごくっ", "こくん",
    "ひゅるひゅる", "ひゅーひゅる", "ふーふー", "スースー", "すーすー",
    # makan / suara tubuh
    "むしゃむしゃ", "もぐもぐ", "ぱくぱく", "がつがつ", "がりがり",
    "ぼりぼり", "ばりばり", "めきめき", "ぼきぼき", "ばきばき", "ぱきぱき",
    "ぐきぐき", "がんがん", "ごんごん", "どんどん", "とんとん", "こんこん",
    # gerak / angin / air
    "ばたばた", "ぱたぱた", "ひらひら", "ふわふわ", "ふわり", "ふわっ",
    "ほわっ", "ぽわん", "ゆらゆら", "ふらふら", "ぐらぐら", "ふらっ",
    "よろよろ", "とぼとぼ", "うろうろ", "そろそろ", "めそめそ",
    "しとしと", "ぽつぽつ", "ざあざあ", "じゃあじゃあ", "ざーざー",
    "ぱらぱら", "ばらばら", "ざわざわ", "がやがや", "わいわい",
    "ひそひそ", "こそこそ", "ごそごそ", "がさがさ", "かさかさ",
    "ざくざく", "じゃりじゃり", "がしゃがしゃ",
    # benturan / suara keras
    "ごとごと", "ごとん", "がたん", "どしん", "ずしん", "どすん",
    "どかん", "どーん", "どどーん", "ががーん", "ばーん", "がーん",
    "ばきっ", "ぼきっ", "ぐきっ", "ぱきっ", "べきっ", "ばりっ",
    "びりっ", "ぷちっ", "ぷつん", "ちぎっ", "がちゃっ", "ぱちん",
    "ばちん", "ぱちっ", "ちょん",
    # desis / ledakan kecil
    "しゅっ", "ひゅっ", "ひゅー", "ぴゅー", "びゅー", "びゅん",
    "しゅー", "しゅーっ", "ぷしゅー", "ぷしゅう", "じゅわっ",
    "じゅうじゅう", "ぐつぐつ", "ふつふつ", "ぼこぼこ", "ぷくぷく",
    "ぷくっ", "ぷかぷか", "ぷらぷら",
    # emosi / kondisi
    "がくっ", "がっくり", "しゅん", "しょんぼり", "ぼんやり",
    "ぼーっ", "ぼうっと", "うとうと", "すやすや", "ぐっすり",
    "ぐうぐう", "すぴすぴ", "ぴょこん", "ぴょんぴょん",
    "ぴくっ", "ぴくぴく", "ひくっ", "ひくひく", "ぷるん", "ぷるっ",
    "ぽよん", "ぽよぽよ", "ぷにぷに", "むにむに", "むちむち",
    "もみもみ", "ぎゅっ", "ぎゅー", "ぎゅうっ", "きゅっ", "きゅー",
    "ひしっ", "がっつり", "がつん", "ずっぽり", "ずぽっ", "ぽんっ",
    "ぽん", "ぽちっ", "ぽんぽん",
})

# Kata kana pendek yang umum dipakai DIALOGUE — tidak boleh dikunci SFX
# walau di luar balon. Heuristik 'kana pendek = SFX' terlalu rakus:
# "それは" dan "ちょっと" juga 2-4 kana. Daftar ini menangani kata yang
# paling sering muncul sebagai narasi/dialog di atas art.
_KA_DIALOGUE = frozenset({
    # kata tunjuk & sambung
    "それ", "それで", "それは", "それに", "それも", "それから", "それな", "それか",
    "あれ", "あれは", "これ", "これは", "これで", "これも", "どこ", "だれ",
    "なに", "なん", "どう", "どうだ", "どうして", "なぜ", "なんで", "なぜか",
    "そう", "そうだ", "そうね", "そうか", "そうよ", "そうそう", "そんな",
    "こんな", "あんな", "どの", "この", "その", "あの", "つまり", "だから",
    "でも", "けど", "そして", "ところで", "ちなみに", "じゃあ", "では",
    # jawaban & seruan
    "はい", "いいえ", "うん", "うーん", "うーむ", "ええ", "えー", "ええと",
    "あのね", "ねえ", "ねぇ", "まあ", "もう", "まだ", "もっと", "だめ",
    "ダメ", "やめ", "やめて", "やめろ", "まって", "ちょっと", "ごめん",
    "ゴメン", "ごめんなさい", "ありがとう", "すみません", "お願い", "おねがい",
    "こんにちは", "こんばんは", "さようなら", "おはよう", "おやすみ",
    "がんばれ", "がんばって", "いいね", "いいよ", "いいの", "いいんだ",
    "いや", "いやいや", "あら", "あらあら", "まあまあ", "なるほど",
    "なるほどね", "うんうん", "えっ", "あっ", "んっ",
    "うっ", "むっ", "ふむ",
    # Seruan yang JELAS ucapan tapi belum tercatat, jadi cabang di dalam balon
    # menguncinya SFX hanya karena 3 huruf + っ (うわっ やだっ まてっ ...). SFX
    # berarti translation=None + PROTECTED, yaitu balon Jepang tercetak tanpa
    # satu pun pesan error — itulah cacat "short bubble untranslate". Ditulis
    # dalam bentuk hiragana saja; pencarian menormalkan katakana lebih dulu,
    # jadi satu entri menutup オイ maupun おい.
    "おい", "まて", "うそ", "ちょ", "ふぇ", "やだ", "そこ", "うわ",
    # adverb ABAB & kata pendek lain yang sering jadi dialog (di luar balon)
    "ときどき", "そろそろ", "だんだん", "ぼちぼち", "ぼつぼつ", "じきじき",
    "きっと", "たぶん", "まさか", "さすが", "やはり", "やっぱり", "やっぱ",
    "あとで", "あと", "いま", "ここ", "あそこ", "そこで", "こっち",
    "そっち", "あっち", "どっち", "じゃ", "さて", "まず", "おまたせ",
    "いったい", "なんと", "なんて", "そうかな", "そうかも", "そうだね",
    "そうなの", "あれれ", "えーっと", "うーん", "うーーん",
    # ekspresi pendek yang sering jadi narasi
    "わかった", "わかりました", "だめだ", "よかった", "すごい", "うれしい",
    "かなしい", "さみしい", "つらい", "やばい", "むり", "ムリ", "むりだ",
    "できない", "できる", "わからない", "しらない", "しってる", "つかれた",
    "きもい", "うざい", "きたない", "きれい", "かわいい", "かっこいい",
    "たのしい", "おもしろい", "つまらない", "へんだ", "おかしい", "こわい",
})

# Simbol emosi yang wajib bertahan di hasil terjemahan.
_EMOTION = frozenset("♥♡❤💕💗💓♪♫♬☆★〜～")

# Punctuation Jepang yang lolos apa adanya dari DeepL. Anime Ace cuma ~159
# glyph, jadi 「 」 … ： dirender jadi kotak tofu di dalam balon. Kurung sudut
# tidak bermakna di Inggris -> dibuang (None); sisanya dipetakan ke ASCII.
# ♥ ♡ ♪ ☆ TIDAK ada di sini — lihat _EMOTION.
_PUNCT_MAP = str.maketrans({
    "「": None, "」": None, "『": None, "』": None,
    "【": None, "】": None, "〔": None, "〕": None,
    "〈": None, "〉": None, "《": None, "》": None,
    # ＼…／ = tanda penekanan dekoratif Jepang yang mengapit teks ('＼失礼しました').
    # DeepL meloloskannya apa adanya, Anime Ace tidak punya glyph-nya, dan hasilnya
    # kotak tofu di depan baris pertama. Di Inggris tanda ini tidak bermakna.
    "＼": None, "＿": None,
    # Wave dash 〜 (U+301C) dan fullwidth tilde ～ (U+FF5E) -> tilde ASCII.
    # Simbolnya BERTAHAN, cuma dipindah ke lebar ASCII: keduanya tidak ada di
    # Anime Ace sementara '~' ada, jadi versi lebar dirender oleh font fallback
    # dengan wajah huruf yang berbeda dari balon lain — dan lebarnya salah diukur,
    # karena layout() mengukur baris dengan font utama saja. Ini juga syarat
    # plan.txt "simbol seperti ! dan love tetap ada" tanpa mengorbankan kerapian.
    "〜": "~", "～": "~",
    "　": " ", "・": " ",
    "。": ".", "．": ".", "、": ",", "，": ",",
    "：": ":", "；": ";", "／": "/", "！": "!", "？": "?",
    "（": "(", "）": ")", "…": "...", "‥": "..",
    "“": '"', "”": '"', "‘": "'", "’": "'",
})

class DeepLClient:
    """Pegang key saja — cukup untuk semua panggilan DeepL."""

    def __init__(self, api_key: str) -> None:
        self.key = api_key


class RouterClient:
    """Router OpenAI-compatible: base URL + key + nama model.

    Beda dari DeepLClient karena base URL-nya bukan konstanta (host Funnel bisa
    berubah, dan model bisa ditimpa env) — jadi keduanya ikut di client, bukan
    dibaca ulang di tiap fungsi.

    Kelas ini juga memegang BATAS WAKTU, HEADER, dan PARAMETER BODY tambahan,
    bukan membacanya dari config di dalam _router_call(). Alasannya: penyedia
    OpenAI-compatible kedua (faucet) sehat pada 3 s sementara router butuh
    120 s, dan modelnya butuh thinking dimatikan. Kalau angka-angka itu dibaca
    dari konstanta global, satu penyedia memaksakan batasnya ke penyedia lain.
    """

    extra: dict = {}
    # Host gorouter ada di balik Cloudflare dan MENOLAK klien tanpa User-Agent
    # dengan 403 "error code 1010" — terukur 17 Agu 2026, dua bentuk auth
    # sama-sama 403, dan key yang sama dengan UA ini membalas 200. Header ini
    # milik KELAS, bukan global, karena faucet TERUKUR sehat tanpa UA dan
    # menambahkannya di sana berarti mengubah yang bekerja tanpa mengukurnya.
    headers: dict = {"User-Agent": BROWSER_UA}
    timeout = ROUTER_TIMEOUT
    deadline = ROUTER_DEADLINE
    fallback: tuple[str, ...] = ROUTER_FALLBACK
    max_tokens: int = 0
    tag = "router"

    def __init__(self, api_key: str, base: str = "", model: str = "") -> None:
        self.key = api_key
        self.base = (base or ROUTER_API_BASE).rstrip("/")
        self.model = model or ROUTER_MODEL

class FaucetClient(RouterClient):
    """freetokenfaucet: OpenAI-compatible, jadi seluruh jalur router dipakai ulang.

    Yang berbeda hanya empat angka dan satu parameter body — lihat FAUCET_* di
    config.py untuk alasan tiap nilainya, terutama thinking.type=disabled yang
    WAJIB (tanpa itu jawaban bisa keluar sebagai string kosong tanpa error).
    """

    extra = FAUCET_EXTRA
    # Sengaja KOSONG, bukan warisan RouterClient: faucet TERUKUR membalas 200
    # tanpa User-Agent, jadi tidak ada alasan mengirim header yang belum diukur
    # ke sana.
    headers: dict = {}
    timeout = FAUCET_TIMEOUT
    deadline = FAUCET_DEADLINE
    fallback = FAUCET_FALLBACK
    max_tokens = FAUCET_MAX_TOKENS
    tag = "faucet"

    def __init__(self, api_key: str, base: str = "", model: str = "") -> None:
        self.key = api_key
        self.base = (base or FAUCET_API_BASE).rstrip("/")
        self.model = model or FAUCET_MODEL


@dataclass
class ProbeResult:
    model: str
    listed: bool
    ok: bool
    reason: str
    latency: float
    sample: str = ""

    def as_row(self) -> dict[str, str]:
        return {
            "model": self.model,
            "listed": "yes" if self.listed else "no",
            "verdict": "OK" if self.ok else "FAIL",
            "reason": self.reason,
            "latency": f"{self.latency:.1f}s",
            "sample": self.sample[:60],
        }


def _is_router(provider: str | None = None) -> bool:
    """Provider aktif = LLM OpenAI-compatible (router ATAU faucet)?

    Namanya tetap _is_router karena semua pemanggilnya menanyakan hal yang sama:
    "boleh pakai anggaran balon dan prompt sistem?" — dan jawabannya sama untuk
    kedua penyedia LLM. Yang membedakan router dari faucet cuma kelas client.
    """
    p = (provider or SETTINGS.provider or "").lower()
    return "router" in p or "faucet" in p


def _is_faucet(provider: str | None = None) -> bool:
    p = (provider or SETTINGS.provider or "").lower()
    return "faucet" in p


def _secret(name: str) -> str:
    """Satu rahasia dari Colab Secrets -> env. Nilainya tidak pernah dicetak."""
    try:
        from google.colab import userdata

        val = userdata.get(name)
        if val:
            return val.strip()
    except (ImportError, KeyError, Exception):  # noqa: BLE001 - SecretNotFound
        pass
    return os.environ.get(name, "").strip()


def get_api_key(ui_key: str | None = None, provider: str | None = None) -> str:
    """Colab Secrets -> env -> field UI. Jangan pernah hardcode.

    Nama rahasianya ikut penyedia (DEEPL_API_KEY vs ROUTER_API_KEY) supaya
    keduanya bisa tersimpan berdampingan dan berganti penyedia di UI tidak
    menuntut menempel key lagi.
    """
    router = _is_router(provider)
    faucet = _is_faucet(provider)
    name = "FAUCET_API_KEY" if faucet else ("ROUTER_API_KEY" if router else "DEEPL_API_KEY")
    key = _secret(name)
    if key:
        return key
    if ui_key and ui_key.strip():
        return ui_key.strip()
    # Jalur terakhir khusus penyedia LLM: file kredensial lokal di luar repo/
    # notebook. Sengaja TIDAK dipakai untuk DeepL — deepl.txt tidak pernah
    # dibaca kode, dan itu tetap begitu.
    #
    # Formatnya beda per file, jadi diambil dengan regex bukan indeks baris:
    # test.txt = baris 3 berisi key mentah; freetokenfaucet.txt = potongan kode
    # Python berisi api_key="tf_..."; gorouter.txt = baris `set` gaya Windows
    # berisi ANTHROPIC_AUTH_TOKEN=... Regex tahan terhadap baris yang bergeser.
    if faucet:
        from config import WORK

        try:
            raw = (WORK / "freetokenfaucet.txt").read_text(encoding="utf-8")
        except OSError:
            raw = ""
        m = re.search(r'api_key\s*=\s*["\']([^"\']+)["\']', raw)
        if m:
            return m.group(1).strip()
    elif router:
        from config import WORK

        # gorouter.txt dulu, karena itulah host yang dilayani ROUTER_API_BASE
        # sekarang. test.txt dibiarkan sebagai jalur kedua supaya konfigurasi
        # lama tidak mendadak kehilangan key-nya.
        try:
            graw = (WORK / "gorouter.txt").read_text(encoding="utf-8")
        except OSError:
            graw = ""
        gm = re.search(r'ANTHROPIC_AUTH_TOKEN\s*=\s*["\']?(\S+?)["\']?\s*$',
                       graw, re.M)
        if gm:
            return gm.group(1).strip()
        for cand in (WORK / "test.txt",):
            try:
                ln = [x.strip() for x in cand.read_text(encoding="utf-8").splitlines()]
            except OSError:
                continue
            if len(ln) > 3 and ln[3]:
                return ln[3]
    raise RuntimeError(
        f"API key tidak ditemukan. Isi Colab Secrets '{name}' "
        "atau tempel di field API Key pada UI."
    )


def make_client(api_key: str, provider: str | None = None):
    if _is_faucet(provider):
        return FaucetClient(api_key.strip())
    if _is_router(provider):
        return RouterClient(api_key.strip())
    return DeepLClient(api_key.strip())


def pick_model(client, verbose: bool = True) -> tuple[str, list[ProbeResult]]:
    """Nama model yang dipakai + tabel probe.

    DeepL tidak butuh pemilihan model — tidak ada yang menolak konten. Router
    memakai model dari client (test.txt/env), dan verifikasi ketersediaannya
    TIDAK dilakukan di sini: router ini membalas 502 untuk model yang ada dan
    berhasil di panggilan berikutnya, jadi probe yang gagal akan memilih model
    cadangan tanpa alasan. call_any() yang menangani itu saat penerjemahan.
    """
    if isinstance(client, RouterClient):
        if verbose:
            print(f"[{client.tag}] {client.base}  model={client.model}  (key tidak dicetak)")
        return client.model, []
    return "deepl", []


def probe_model(client, model: str, listed: bool) -> ProbeResult:
    if isinstance(client, RouterClient):
        t0 = time.monotonic()
        try:
            got = _router_call(client, model, "Reply with the JSON {\"0\": \"OK\"}.",
                               "Lines:\n{\"0\": \"テスト\"}",
                               timeout=min(60, client.timeout))
            return ProbeResult(model, True, bool(got), f"{client.tag} ok",
                               time.monotonic() - t0, str(got)[:60])
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(model, True, False,
                               f"{type(exc).__name__}: {str(exc)[:80]}",
                               time.monotonic() - t0)
    return ProbeResult("deepl", True, True, "deepl ok", 0.0)


def check_usage(client) -> str:
    """Kuota DeepL: 'dipakai / limit' dari endpoint /usage.

    Router tidak punya endpoint kuota; yang berguna di sana adalah apakah dia
    MENJAWAB, jadi yang dilaporkan hasil probe satu panggilan.
    """
    if isinstance(client, RouterClient):
        p = probe_model(client, client.model, True)
        return f"{client.model}: {p.reason} ({p.latency:.1f}s)"
    try:
        status, data = _http_json(client, "GET", "/usage", None)
        if status == 200:
            return (
                f"{data.get('character_count', 0):,} / "
                f"{data.get('character_limit', 0):,} karakter"
            )
        return f"http {status}"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {str(exc)[:120]}"


# ---------------------------------------------------------------- HTTP


def _http_json(client, method: str, path: str, payload: dict | None,
               timeout: int = 60) -> tuple[int, dict | str]:
    req = urllib.request.Request(DEEPL_API_BASE + path, method=method)
    req.add_header("Authorization", f"DeepL-Auth-Key {client.key}")
    body = None
    if payload is not None:
        req.add_header("Content-Type", "application/json")
        body = json.dumps(payload).encode()
    try:
        with urllib.request.urlopen(req, body, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:400]
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {str(exc)[:200]}"


def _translate_texts(client, texts: list[str], target_lang: str,
                     formality: str) -> list[str]:
    """Satu batch terjemahan DeepL dengan retry transien (3x)."""
    last: Exception | None = None
    for attempt in range(3):
        payload: dict = {
            "text": texts,
            "target_lang": target_lang,
            "source_lang": "JA",
        }
        if formality and formality != "default":
            payload["formality"] = formality
        status, data = _http_json(client, "POST", "/translate", payload)
        if status == 200 and isinstance(data, dict):
            return [t.get("text", "") for t in data.get("translations", [])]
        if status in (429, 500, 502, 503, 504):
            last = RuntimeError(f"http {status}: {str(data)[:120]}")
            time.sleep(2 * (attempt + 1))
            continue
        raise RuntimeError(f"DeepL http {status}: {str(data)[:200]}")
    raise last  # type: ignore[misc]


# ---------------------------------------------------------------- HTTP router


def _decode_router(raw: str) -> dict:
    """Body router -> dict. Content-Type-nya text/event-stream walau non-stream.

    Router ini mengembalikan SATU objek chat.completion lalu menempelkan
    `data: [DONE]` TANPA pemisah baris. json.loads() gagal dengan 'Extra data'
    padahal objeknya utuh — jadi dipakai raw_decode() yang berhenti di akhir
    objek pertama dan mengabaikan sisanya.
    """
    raw = (raw or "").strip()
    if raw.startswith("data:"):
        raw = raw[5:].lstrip()
    obj, _end = json.JSONDecoder().raw_decode(raw)
    return obj


def _router_call(client, model: str, system: str, user: str,
                 timeout: int = ROUTER_TIMEOUT) -> dict:
    """Satu panggilan chat/completions -> objek JSON hasil terjemahan.

    Body-nya diambil dari client, bukan dari konstanta: client.extra membawa
    parameter khusus penyedia (faucet butuh thinking.type=disabled, kalau tidak
    jatah keluarannya habis untuk reasoning dan content keluar KOSONG tanpa
    error HTTP) dan client.max_tokens membatasi keluaran kalau penyedia
    menghitung token — router tidak, faucet ya, dan tokennya terbatas.
    """
    body = {
        "model": model, "temperature": 0.3, "stream": False,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        **getattr(client, "extra", {}),
    }
    if getattr(client, "max_tokens", 0):
        body["max_tokens"] = client.max_tokens
    req = urllib.request.Request(client.base + "/chat/completions", method="POST")
    req.add_header("Authorization", "Bearer " + client.key)
    req.add_header("Content-Type", "application/json")
    for _hk, _hv in getattr(client, "headers", {}).items():
        req.add_header(_hk, _hv)
    with urllib.request.urlopen(req, json.dumps(body).encode(), timeout=timeout) as r:
        d = _decode_router(r.read().decode())
    ch = (d.get("choices") or [{}])[0]
    txt = ch.get("message", {}).get("content") or ""
    u = d.get("usage") or {}
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        # finish_reason + jumlah token keluaran IKUT di pesan, bukan cuma
        # cuplikan teksnya. Sebabnya konkret: "bukan JSON" punya dua penyebab
        # yang penanganannya berlawanan. finish_reason="length" berarti jawaban
        # TERPOTONG dan max_tokens-lah yang kurang (FAUCET_MAX_TOKENS=1200 cukup
        # untuk 8 balon — terukur out=147 — tapi halaman 19 balon bisa
        # melampauinya); finish_reason="stop" dengan teks utuh berarti model
        # memang tidak membalas JSON dan yang salah promptnya. Tanpa angka ini
        # pembaca log harus menebak di antara keduanya.
        raise ValueError(
            f"bukan JSON (finish_reason={ch.get('finish_reason')!r}, "
            f"out={u.get('completion_tokens', '?')}/{body.get('max_tokens', '-')} "
            f"token): {txt[:200]}"
        )
    print(f"[{getattr(client, 'tag', 'router')}] in={u.get('prompt_tokens', '?')} "
          f"out={u.get('completion_tokens', '?')}")
    return json.loads(m.group(0))


def _router_call_any(client, model: str, system: str, user: str) -> tuple[dict, str]:
    """_router_call() dengan percobaan ulang + model cadangan, BERBATAS WAKTU.

    Mengulang model YANG SAMA sebelum pindah, karena 502 dari router ini
    SEMENTARA dan bukan tanda model tidak ada: terukur, satu model membalas 502
    tiga kali lalu 200 dalam 4 detik sementara /models tetap 200 sepanjang waktu.
    Berpindah model saja tidak menolong — yang menolong mencoba lagi.

    Tapi mencoba lagi HARUS ada batasnya, dan batas itu satu angka untuk seluruh
    rangkaian (ROUTER_DEADLINE), bukan hasil perkalian timeout x percobaan x
    model. Versi pertama tanpa deadline: satu halaman menggantung 3 jam karena
    router tidak menutup koneksi — di UI kelihatan seperti "GPU lambat" padahal
    tidak ada satu pun kernel yang jalan. Deadline dilewatkan juga ke timeout
    tiap percobaan, supaya percobaan terakhir tidak melompati batasnya sendiri.
    """
    order = (model, *(f for f in client.fallback if f != model))
    tried: list[str] = []
    deadline = client.deadline
    t_end = time.monotonic() + deadline
    for attempt in range(1, ROUTER_RETRY + 1):
        for m in order:
            left = t_end - time.monotonic()
            if left <= 1:
                raise TimeoutError(
                    f"{client.tag} tidak menjawab dalam {deadline}s "
                    f"({', '.join(tried[-6:]) or 'tanpa balasan'})"
                )
            short = m.rsplit("/", 1)[-1]
            try:
                return _router_call(client, m, system, user,
                                    timeout=int(min(client.timeout, left))), m
            except urllib.error.HTTPError as e:
                if e.code == 402:
                    # 402 bukan gangguan sementara: modelnya berbayar dan saldo
                    # akun 0, jadi mencoba lagi PASTI gagal dan cuma memakan
                    # deadline. Body aslinya berbahasa Mandarin
                    # ("为付费模型，但你的资金账户余额不足"), jadi pesan sendiri lebih
                    # berguna daripada meneruskannya — yang perlu dibaca user
                    # adalah nama modelnya dan tindakannya.
                    raise RuntimeError(
                        f"{client.tag}: model '{m}' BERBAYAR dan saldo akun 0 "
                        f"(HTTP 402). Ganti ke model gratis lewat env "
                        f"{'FAUCET_MODEL' if client.tag == 'faucet' else 'ROUTER_MODEL'}"
                        " (faucet gratis: mimo-v2.5-pro, mimo-v2.5,"
                        " gpt-5.6-terra) atau top-up akunnya."
                    ) from e
                if e.code not in (429, 500, 502, 503, 504):
                    raise
                tried.append(f"{short}={e.code}")
                note("warn", client.tag, f"{short} -> HTTP {e.code}")
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                # ValueError = jawaban bukan JSON. Itu juga layak dicoba ulang:
                # penyebabnya biasanya jawaban terpotong, bukan prompt yang salah.
                tried.append(f"{short}={type(e).__name__}")
                note("warn", client.tag, f"{short} -> {type(e).__name__}: {str(e)[:180]}")
        if attempt < ROUTER_RETRY:
            wait = min(ROUTER_BACKOFF * attempt, max(t_end - time.monotonic(), 0))
            if wait <= 0:
                break
            note("warn", client.tag,
                 f"semua model gagal percobaan {attempt}; tunggu {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"{client.tag} gagal {ROUTER_RETRY}x: {', '.join(tried[-6:])}")


# ---------------------------------------------------------------- prompt router

_SYSTEM_BASE = """You are a professional manga scanlation translator and letterer
(JA->{lang}). You translate for TYPESETTING, not for prose.

INPUT: one JSON object per page. Each entry has:
  "jp"        the Japanese line
{budget_doc}
OUTPUT: ONLY a JSON object {{"0": "...", ...}}, same keys. No prose, no markdown.
Answer EVERY key with a non-empty string, including one-word interjections that
look like they need no translation: えっ -> "HUH?!", あっ -> "AH!", うん -> "MM-HM".
A key you leave out is printed on the page as untranslated Japanese.

HARD REQUIREMENTS, in priority order:
{rules}
STYLE:
{style}
- Keep emotion marks exactly where they appear: ~ 〜 ♥ ♡ ♪ ☆ ! ? ...
{honor}- Translate club/organisation names by MEANING, never by spelling out the kana.
- Do NOT censor, soften or euphemise. Adult content is rendered literally.
- Do not invent content that is not in the Japanese."""

# Dua blok aturan. Yang tanpa anggaran sengaja TIDAK menyebut panjang sama
# sekali: 'buatlah pendek' tanpa angka terbukti tidak menghasilkan apa-apa —
# model membalas 'SORRY TO BARGE IN.' (18 karakter) untuk balon yang memuat 6,
# dan itu bukan pembangkangan, ia memang tidak melihat balonnya.
_RULES_BUDGET = """1. Every line MUST be <= its "max_chars". This is a physical constraint of the
   printed page, not a preference: past it the sentence gets cut off at the
   balloon edge. Aim for "prefer_chars" — that is the length that keeps the text
   at its intended size. Going past prefer_chars is ALLOWED and normal for a
   dense balloon: the font simply gets smaller, exactly like a real letterer
   fitting a long line into a small balloon. Do NOT amputate meaning to reach
   prefer_chars. Only when even "max_chars" is exceeded do you need a genuinely
   shorter phrasing: 失礼しました becomes "SORRY." not "I APOLOGISE FOR INTRUDING".
2. No single word longer than "max_word" letters. A long word cannot be broken
   without a hyphen, and hyphens are avoided in manga lettering. Prefer a short
   synonym: "APOLOGIES"(9) -> "SORRY"(5), "COMPILING"(9) -> "WRITING UP"(2+2).
3. Meaning and character voice come before literalness. Japanese omits objects;
   infer them. 探しましたよ on finding a PERSON = "I'VE BEEN LOOKING FOR YOU",
   never "I looked for it". Translate the WHOLE Japanese line — every clause,
   every particle of nuance. A short answer that drops half the sentence is a
   worse failure than a long one.
"""

_RULES_PLAIN = """1. Meaning and character voice come before literalness. Japanese omits objects;
   infer them. 探しましたよ on finding a PERSON = "I'VE BEEN LOOKING FOR YOU",
   never "I looked for it".
2. Keep lines short. Speech balloons are small; long clauses do not fit.
"""

# Kenapa DUA angka dan bukan satu: keputusan user 'boleh panjang, font mengecil'.
# max_chars diambil dari char_budget pada LANTAI ukuran font (= yang benar-benar
# masih tercetak), prefer_chars dari plafon proporsional balon. Versi sebelumnya
# mengirim plafon proporsional sebagai 'max_chars' berbunyi MUST, dan pada
# hasilnew/jp_6.JPG itu berarti model diperintah menulis 2-39 karakter untuk
# balon yang wording typeset referensinya 15-71 karakter — hasilnya 'SO?' untuk
# balon yang referensinya satu kalimat penuh.
_BUDGET_DOC = """  "max_chars"    hard ceiling; past this the line is cut off at the balloon edge
  "prefer_chars" length that keeps the intended font size; going over just
                 shrinks the font, which is fine and normal
  "max_word"     longest single word that fits on one line in that balloon
"""


def _system_prompt(target_lang: str, style: str, keep_honorifics: bool,
                   with_budget: bool) -> str:
    return _SYSTEM_BASE.format(
        lang=(target_lang or "English").upper(),
        budget_doc=_BUDGET_DOC if with_budget else "",
        rules=_RULES_BUDGET if with_budget else _RULES_PLAIN,
        style="- " + TRANSLATION_STYLES.get(
            style, TRANSLATION_STYLES["Manga Natural"]),
        honor=("- Keep honorifics (-san, -kun, -chan, -senpai, -sama).\n"
               if keep_honorifics else
               "- Localise honorifics into natural address in the target language.\n"),
    )


def _user_prompt(items: list[Region], budget: dict[int, dict] | None) -> str:
    """JSON berisi jp DAN (kalau ada) anggaran per balon."""
    payload: dict[str, object] = {}
    for r in items:
        if budget and r.idx in budget:
            d = budget[r.idx]
            payload[str(r.idx)] = {"jp": r.src_text,
                                   "max_chars": d["hard"],
                                   "prefer_chars": d["soft"],
                                   "max_word": d["word_hard"]}
        else:
            payload[str(r.idx)] = r.src_text
    return "Lines:\n" + json.dumps(payload, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- label (SFX)


def _sfx_core(text: str) -> str:
    """Inti kana: buang simbol/tanda baca, sisakan kana + ー + っ/ッ.

    "フー．．．"  -> "フー"
    "ドキッ♥"    -> "ドキッ"
    "（はぁ…）"  -> "はぁ"
    """
    return "".join(ch for ch in text if _KANA.match(ch))


def _broken_kana(raw: str) -> bool:
    """Apakah ada JEDA di ANTARA dua kana (bukan di ujung)?

    "ヒ．．．ッ"  -> True   jeda memutus mora: napas tertahan = suara tokoh
    "キャ．．．ッ" -> True
    "フー．．．"   -> False  jeda di UJUNG: bunyi yang memanjang lalu berhenti
    "ゴクッ．．．" -> False
    "ドキッ"     -> False  tidak ada jeda sama sekali

    Dipakai hanya sebagai penahan aturan K di _sfx_in_bubble. Perhatikan bahwa
    _sfx_core MEMBUANG semua tanda baca, jadi informasi ini sudah lenyap dari
    `core` — penahannya wajib melihat teks ASLI.
    """
    idx = [i for i, ch in enumerate(raw) if _KANA.match(ch)]
    if len(idx) < 2:
        return False
    return any(raw[i] in _PAUSE for i in range(idx[0] + 1, idx[-1]))


def _sfx_pattern(core: str) -> bool:
    """Pola ulang onomatope: ドキドキ, ドキドキドキ, ぴくぴくっ, ばたばた...

    Kepala harus pengulangan PENUH dari satu unit (>= 2 ulangan), lalu ekor
    opsional っ/ッ/ー 1-2 char (ぴくぴくっ, どきどきっ). Bentuk ini sengaja
    menolak kata pinjaman seperti サッカー (サッ+カ+ー — bukan ulangan) dan
    kata 2-char seperti うう / ええ (dialog) yang tidak boleh dikunci.
    """
    n = len(core)
    if n < 4:
        return False
    for tail_len in (0, 1, 2):
        tail = core[n - tail_len:] if tail_len else ""
        if tail and not all(c in _SMALL or c == _LONG for c in tail):
            continue
        head = core[: n - tail_len]
        m = len(head)
        if m < 2:
            continue
        for u in range(1, m // 2 + 1):
            unit = head[:u]
            if all(c in _SMALL or c == _LONG for c in unit):
                continue
            if m % u == 0 and m // u >= 2 and head == unit * (m // u):
                return True
    return False


def _kata2hira(s: str) -> str:
    """Katakana -> hiragana, HANYA untuk pencarian kamus (bukan untuk render).

    Tanpa ini kamus dialog yang isinya hiragana tidak pernah bisa cocok dengan
    dialog yang di manga ditulis katakana (ダメッ ハイッ ウンッ ムリッ オイッ),
    dan semuanya jatuh ke SFX = balon Jepang tercetak. Tidak ada normalisasi
    kana lain di modul ini, jadi ini satu-satunya jembatannya.
    """
    return "".join(
        chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s
    )


def _all_katakana(core: str) -> bool:
    """Inti kana yang SELURUHNYA katakana (tanpa satu pun hiragana).

    Konvensi manga: bunyi ditulis katakana, ucapan hiragana. Yang ditulis
    katakana tapi memang ucapan sudah diselamatkan kamus (lewat _kata2hira)
    sebelum aturan ini dipakai.
    """
    ada = any("ァ" <= c <= "ヶ" for c in core)
    hira = any("ぁ" <= c <= "ゖ" for c in core)
    return ada and not hira


def _sfx_stem(core: str) -> str:
    """Inti minus ekor っ/ッ/ー: どきっ -> どき, あーっ -> あ, ダメッ -> ダメ."""
    i = len(core)
    while i and (core[i - 1] in _SMALL or core[i - 1] == _LONG):
        i -= 1
    return core[:i]


def _has_kanji(text: str) -> bool:
    """Kalimat dengan kanji pasti dialog — jangan pernah dikunci sebagai SFX."""
    return any(
        0x4E00 <= ord(ch) <= 0x9FFF
        or 0x3400 <= ord(ch) <= 0x4DBF
        or 0xF900 <= ord(ch) <= 0xFAFF
        for ch in text
    )


def _sfx_in_bubble(core: str, n: int, raw: str) -> bool:
    """Apakah inti kana DI DALAM balon adalah bunyi (SFX), bukan ucapan.

    Cabang lama di sini satu baris: `n == 3 and has_small` -> SFX. Itu penyebab
    cacat "balon pendek tidak diterjemah". SFX berarti translation=None +
    PROTECTED, jadi translate_page MELEWATI region itu tanpa satu pun pesan
    error, dan yang tercetak adalah balon Jepang asli. _label2.py mengukur
    cabang itu pada 32 kasus: 21 salah — 20 seruan yang jelas ucapan (ええっ
    うんっ だめっ いやっ まてっ うそっ なにっ ちょっ そこっ はいっ ねえっ
    もうっ やめっ あーっ ふぇっ ...) dikunci SFX hanya karena panjangnya 3 dan
    ada っ, plus satu arah sebaliknya (ハッ, katakana, SFX sejati -> DIALOGUE).

    Penggantinya tiga bagian, diukur dua arah sekaligus di _label3.py:
      N  kamus dialog dicari pada bentuk hiragana DAN pada batangnya
         (_kata2hira + _sfx_stem), supaya dialog yang ditulis katakana
         (ダメッ ハイッ ウンッ ムリッ オイッ) ketemu lewat entri hiragana-nya.
         Tanpa ini tidak ada normalisasi kana sama sekali di modul ini, jadi
         katakana-ditulis-ucapan tidak pernah bisa cocok kamus.
      K  kana ber-っ/ー yang SELURUHNYA katakana = bunyi (konvensi manga: bunyi
         katakana, ucapan hiragana). Yang katakana tapi memang ucapan sudah
         diselamatkan N lebih dulu, jadi urutannya wajib N sebelum K.
      S  yang hiragana hanya SFX kalau BATANGNYA terbukti onomatope, yaitu
         batang-gandanya ada di _SFX_DICT (どきっ -> どきどき). Beban buktinya
         sengaja DIBALIK dari yang lama: dulu cukup 'pendek dan ada っ'.
      V  K adalah satu-satunya aturan yang menang HANYA lewat BENTUK — tanpa
         bukti kamus maupun pola ulang. Jadi K tidak boleh menang kalau teks
         ASLINYA membawa tanda BICARA: intonasi tanya, atau jeda DI ANTARA
         kana. Ini yang membedakan ヒ．．．ッ！？ (jeritan tokoh di balon hitam
         bergerigi, hasilnew5) dari ハッ — dua-duanya satu mora katakana + ッ
         di dalam balon, jadi panjang batang TIDAK bisa memisahkannya.
         V ditaruh di dalam K, bukan menggantikannya, supaya bunyi berkamus
         tetap lolos lewat S sesudahnya (ドキッ！？ -> どきどき di _SFX_DICT).

    Terukur pada 72 kasus (45 dialog + 27 SFX): arah dialog 30 salah -> 0,
    arah SFX 4 salah -> 0. Asimetrinya sengaja: salah menuduh dialog = balon
    Jepang tercetak dan pembaca tidak bisa membacanya, salah melepas SFX =
    SFX ikut diterjemah tapi masih tertahan kamus. はぁっ tetap SFX karena
    はぁはぁ ada di _SFX_DICT — itu embusan napas, bukan ucapan.

    V diukur terpisah di _h5lbl.py atas 60 kasus selftest + 14 kasus baru,
    melawan 4 kandidat lain. Hanya V yang nol kesalahan di KEDUA arah:
    "cuma ？" melewatkan ヒ．．．ッ tanpa tanda tanya; "？ atau ！" merusak
    ズドンッ！ dan パチンッ！ (batangnya tidak berkamus, jadi S tidak
    menyelamatkannya); "batang <= 1 kana wajib berkamus" merusak ハッ.

    `raw` WAJIB, tanpa nilai bawaan: raw="" membuat V mati diam-diam, dan
    aturan yang mati diam-diam adalah cacat yang sama seperti label SFX yang
    melewati balon tanpa pesan error.
    """
    hira = _kata2hira(core)
    stem = _kata2hira(_sfx_stem(core))
    # N — pemanggil sudah menguji `core in _KA_DIALOGUE`; di sini bentuk
    # hiragana dan batangnya.
    if hira in _KA_DIALOGUE or (stem and stem in _KA_DIALOGUE):
        return False
    if _sfx_pattern(core):
        return True                             # どきどき ぴくぴくっ
    if core in _SFX_DICT or hira in _SFX_DICT:
        return True
    # K — bunyi ditulis katakana. Yang menandai bunyi adalah っ/ッ di UJUNG,
    # bukan sembarang っ: サッカー juga seluruhnya katakana dengan ッ dan ー,
    # tapi ッ-nya di tengah dan ujungnya ー — itu kata pinjaman, dan selftest
    # menjaganya tetap DIALOGUE. Batas 6 supaya pinjaman panjang ber-ッ di ujung
    # (kalau ada) tidak terjaring.
    if n <= 6 and _all_katakana(core) and core[-1] in _SMALL:
        # V — kecuali teks aslinya bertanda bicara. Bunyi tidak bertanya, dan
        # bunyi tidak terputus di tengah mora. Jatuh ke S, tidak return False,
        # supaya bunyi yang PUNYA bukti kamus tetap bisa menang.
        if not (any(c in _ASK for c in raw) or _broken_kana(raw)):
            return True                         # ハッ ドキッ ズドンッ パチンッ
    # S — hiragana harus punya catatan onomatope-nya.
    if stem and (stem + stem) in _SFX_DICT:
        return True                             # どきっ びくっ ごくっ
    return False


def _label_region(r: Region) -> None:
    """Klasifikasi satu region: SFX (dijaga utuh) atau DIALOGUE (diterjemah).

    Ambang di luar balon sengaja lebih longgar — SFX manga hampir selalu
    di luar balon, sedangkan kata-kata kana panjang di luar balon jarang.
    Di dalam balon lebih konservatif agar dialog pendek (うん, ええ, はい)
    tidak ikut dikunci.
    """
    t = r.src_text.strip()
    if not t or r.label == "UNREADABLE":
        return

    # Kalimat ber-kanji = dialog sungguhan. Inti kana hanya dinilai bila
    # teks aslinya murni kana + simbol.
    if _has_kanji(t):
        r.label = "DIALOGUE"
        r.label_conf = 0.5
        return

    core = _sfx_core(t)
    if not core:
        r.label = "DIALOGUE"
        r.label_conf = 0.5
        return

    # Kata dialog umum menang atas pola SFX (それは, ちょっと, ごめん...).
    if core in _KA_DIALOGUE:
        r.label = "DIALOGUE"
        r.label_conf = 0.5
        return

    n = len(core)
    in_bubble = r.bubble_bbox is not None
    has_small = any(c in _SMALL for c in core)
    has_long = _LONG in core

    if not in_bubble:
        if n <= 3:
            is_sfx = True                       # ドン バン ピクッ フー ドキッ
        elif has_small or has_long:
            is_sfx = n <= 8                     # ガーン ぴくぴくっ ガッタンゴットン
        elif _sfx_pattern(core):
            is_sfx = True                       # ドキドキ ばたばた どきどきどき
        elif core in _SFX_DICT:
            is_sfx = True
        else:
            is_sfx = False
    else:
        is_sfx = _sfx_in_bubble(core, n, t)

    r.label = "SFX" if is_sfx else "DIALOGUE"
    r.label_conf = 0.6 if is_sfx else 0.5
    r.translation = None if is_sfx else r.translation


def _fallback_labels(regions: list[Region]) -> list[Region]:
    """Label heuristik: SFX = kana di luar/di dalam balon yang berpola.

    Ini klasifikasi BAWAAN pipeline — di versi DeepL dipakai SELALU
    (DeepL tidak bisa menilai SFX).
    """
    for r in regions:
        if r.label != "UNREADABLE":
            _label_region(r)
    return regions


# ---------------------------------------------------------------- simbol


def _restore_symbols(src: str, translation: str) -> str:
    """Jaring pengaman simbol emosi yang hilang saat lewat DeepL.

    DeepL umumnya mempertahankan ♥ ♡ ♪ ☆ 〜 … dan mengubah ！？ jadi !?.
    Tapi kadang simbol di ujung kalimat terbuang (mis. '大好き♥' -> 'I love
    you'). Kalau sumber berakhiran simbol emosi dan hasilnya kehilangan,
    simbol itu disalin ulang ke ujung terjemahan.
    """
    out = (translation or "").strip()
    if not src or not out:
        return out
    out = out.replace("！", "!").replace("？", "?")
    src_end = src.rstrip()
    for ch in _EMOTION:
        if ch in src and ch not in out and src_end.endswith(ch):
            out = out + ch
    return out


# Karakter yang MEMBAWA kata: kanji, hiragana, katakana, katakana setengah-lebar,
# latin, angka. Sengaja TIDAK memuat ー (U+30FC, tanda panjang) dan ・ (U+30FB,
# pemisah) walau keduanya duduk di blok katakana — keduanya tidak pernah menjadi
# kata sendirian, jadi 'ー．．．' harus tetap dihitung tanpa kata.
_WORDY = re.compile(
    r"[一-鿿㐀-䶿ぁ-ゖァ-ヺｦ-ﾝA-Za-z0-9]"
)


def _symbols_only(text: str) -> bool:
    """src_text tanpa satu pun karakter berkata: '．．．', '！？', '♥', '〜'."""
    return bool((text or "").strip()) and not _WORDY.search(text)


def _symbols_as_text(src: str) -> str:
    """Simbol sumber dipetakan ke ASCII, TANPA lstrip _clean_translation().

    _clean_translation() membuang tanda baca di AWAL string ('. SHIZUKU' ->
    'SHIZUKU'), dan pada balon yang isinya cuma simbol SELURUH isinya ada di awal
    — '．．．' akan keluar sebagai string kosong dan balonnya tercetak hampa.
    """
    return " ".join((src or "").translate(_PUNCT_MAP).split())


def _clean_translation(text: str) -> str:
    """Buang glyph yang tidak ada di font komik, samakan punctuation ke ASCII.

    Kurung sudut Jepang lolos apa adanya dari DeepL ('「会長っ」' -> '「Prez」'),
    dan Anime Ace cuma ~159 glyph sehingga 「 」 dirender jadi kotak tofu ⟦ ⟧
    di dalam balon. Kurungnya memang tidak bermakna di Inggris — dibuang saja.

    _EMOTION (♥ ♡ ♪ ☆) TIDAK dibuang: simbol itu wajib bertahan. 〜 dan ～
    dinormalkan ke '~' — bentuknya tetap ada, lebarnya jadi ASCII supaya
    dirender oleh font balon yang sama (lihat komentar di _PUNCT_MAP).
    """
    out = (text or "").translate(_PUNCT_MAP)
    # Titik/koma nyasar di awal terjemahan ('. SHI ZUKU...') — sisa tanda baca
    # Jepang yang kurungnya sudah dibuang. Ujung kanan tidak diusik: '...' dan
    # '?!' di akhir kalimat memang disengaja.
    return " ".join(out.lstrip(" .,:;-–—").split())


def translate_page(client, model: str, regions: list[Region],
                   target_lang: str = "English",
                   style: str = "Manga Natural",
                   keep_honorifics: bool = True) -> list[Region]:
    """Label heuristik dulu, lalu terjemahkan lewat penyedia aktif.

    SFX (dan teks terlindungi lain) tidak pernah dikirim ke mana pun.
    """
    _fallback_labels(regions)
    # Balon yang isinya HANYA simbol ('．．．', '！？', '♥') tidak punya kata untuk
    # diterjemahkan. Dikirim ke model, ia membalas kosong — wajar, tidak ada yang
    # bisa dijawab — lalu jalur perbaikan menuduhnya "BELUM DITERJEMAHKAN" dan
    # satu-satunya error di laporan halaman jadi alarm palsu. Terukur di
    # hitomi_3740721_015: r12='．．．' menghasilkan error_count 1 dengan
    # final_font_size 0, jadi balonnya keluar KOSONG — simbolnya hilang pula.
    #
    # Diselesaikan di sini, SEBELUM items dibangun, sehingga region ini tidak
    # pernah dikirim ke penyedia mana pun, tidak masuk _missing_ids, tidak
    # menghasilkan error, dan tidak dihitung untranslated oleh verify.report().
    # Simbolnya dipetakan ke ASCII dan dicetak apa adanya — '．．．' -> '...' —
    # jadi balon tetap berisi apa yang memang tertulis di halaman aslinya.
    for r in regions:
        if (r.label not in PROTECTED_LABELS and r.translation is None
                and _symbols_only(r.src_text)):
            r.translation = _symbols_as_text(r.src_text) or None
    items = [
        r for r in regions
        if r.label not in PROTECTED_LABELS and r.src_text and r.translation is None
    ]
    if not items:
        return regions
    if isinstance(client, RouterClient):
        return _translate_router(client, model, regions, items, target_lang,
                                 style, keep_honorifics)
    return _translate_deepl(client, items, regions, target_lang, style)


def _translate_deepl(client, items: list[Region], regions: list[Region],
                     target_lang: str, style: str) -> list[Region]:
    code = DEEPL_TARGET.get(target_lang or "English", "EN")
    formality = FORMALITY_BY_STYLE.get(style, "default")

    try:
        for i in range(0, len(items), _CHUNK):
            chunk = items[i : i + _CHUNK]
            out = _translate_texts(client, [r.src_text for r in chunk], code, formality)
            for r, t in zip(chunk, out):
                t = (t or "").strip() or None
                if t is not None:
                    t = _clean_translation(_restore_symbols(r.src_text, t)) or None
                r.translation = t
    except Exception as exc:  # noqa: BLE001 - kegagalan API tidak boleh membunuh halaman
        note("error", "translate",
             f"DeepL gagal ({exc}); halaman keluar TANPA terjemahan, "
             "teks asli tetap di sidecar JSON")
    # Sama seperti jalur router: balon yang tidak dapat terjemahan tercetak
    # berbahasa Jepang, dan itu tidak memicu error apa pun. Jadi disebut di log.
    left = [r for r in items if not r.translation]
    if left:
        note("error", "translate", "BELUM DITERJEMAHKAN (DeepL): "
             + "; ".join(f"r{r.idx}={r.src_text!r}" for r in left))
    return regions


# ------------------------------------------------------------- anggaran + revisi


def _page_budget(items: list[Region]) -> dict[int, dict]:
    """Anggaran karakter per balon, diukur dengan mesin tata letak SUNGGUHAN.

    Dihitung di sini, bukan ditaksir dari lebar bbox, karena angkanya harus
    keluar dari layout() yang nanti merender — anggaran yang dihitung terpisah
    bisa melenceng tanpa ada yang tahu. Balon tanpa mask dilewati: tidak ada
    geometri untuk diukur, jadi baris itu dikirim tanpa batas.
    """
    import typeset

    fp = typeset.FONT_USED or typeset.setup_fonts(verbose=False)
    out: dict[int, dict] = {}
    for r in items:
        try:
            out[r.idx] = typeset.region_budget(r, fp)
        except Exception as exc:  # noqa: BLE001 - satu balon gagal != halaman gagal
            note("warn", "budget",
                 f"r{r.idx} dilewati ({type(exc).__name__}: {exc})")
    return out


def _violations(texts: dict[int, str], budget: dict[int, dict],
                items: list[Region]) -> dict[int, str]:
    """Ukur jawaban model dengan layout() sungguhan. Lapis PENENTU.

    Anggaran karakter itu PROKSI — dihitung dengan teks pengisi, bukan dengan
    kalimat yang akhirnya dipakai. Yang mengikat cuma satu: apakah kalimat INI
    bisa DICETAK di balon INI tanpa terpotong. Jadi yang diukur
    typeset.renders_ok() atas teks aslinya, dan anggaran hanya dipakai untuk
    MEMBERI TAHU model harus sependek apa.

    Plafon proporsional sengaja BUKAN kriteria lulus. Terukur: wording typeset
    profesional halaman referensi sendiri duduk di bawah plafonnya di tiga balon
    padat (-6, -4, -5 px). Menjadikannya kriteria berarti menolak hasil yang
    justru ditiru.

    Kriterianya juga BUKAN 'muat utuh tanpa penggalan di atas lantai'
    (_max_feasible), dan itu juga terukur: pada hasilnew/jp_6.JPG wording
    referensi r3 "CAN'T HELP IT ♥" dan r4 "UH... I-IT'S EMBARRASSING..."
    keduanya memberi feasible 0, padahal fit() merendernya bersih di 6 dan 8 px
    (probe_r34.py). Validator versi itu menolak wording yang sedang ditiru dan
    memaksa model menulis makin pendek — persis keluhan 'translate-nya sedikit
    banget'. Yang tersisa sebagai cacat sungguhan: fit() melaporkan LUBER, yaitu
    barisnya benar-benar terpotong di tepi balon.

    Diukur pada bentuk SETELAH _clean_translation + huruf besar, bukan apa yang
    model tulis. Sebabnya konkret: model membalas '＼SORRY.' dan validator versi
    pertama melaporkan 'tidak muat di ukuran minimum' untuk 7 karakter di balon
    yang memuat 39 — penyebabnya '＼' yang tidak punya glyph di Anime Ace,
    padahal pipeline membuangnya sebelum typeset. Menghukum model atas glyph
    yang sudah ditangani orang lain hanya menghasilkan revisi yang sia-sia.
    """
    import typeset

    fp = typeset.FONT_USED or typeset.setup_fonts(verbose=False)
    rmap = {r.idx: r for r in items}
    bad: dict[int, str] = {}
    for i, t in sorted(texts.items()):
        r, d = rmap.get(i), budget.get(i)
        if r is None or d is None:
            continue
        up = _clean_translation(t or "").upper()
        if not up:
            continue
        try:
            mask = typeset._region_box_mask(r)[1]
            ok, _size = typeset.renders_ok(up, mask, fp)
        except Exception:  # noqa: BLE001
            continue
        if not ok:
            bad[i] = (
                f"{len(up)} chars get cut off at the balloon edge even at the "
                f"smallest readable size. This balloon holds about {d['soft']} "
                f"characters and no single word longer than {d['word_hard']} "
                f"letters. Rewrite much shorter, same meaning."
            )
    return bad


def _missing_ids(got: dict[int, str], items: list[Region]) -> list[int]:
    """idx yang TIDAK punya jawaban terpakai dari model.

    "Tidak punya jawaban" mencakup tiga hal yang semuanya berakhir sama di atas
    kertas: kunci tidak ada di JSON balasan, isinya string kosong/spasi, atau
    isinya habis setelah _clean_translation (mis. model membalas cuma '．．．').
    Diukur pada bentuk AKHIR, bukan pada apa yang model tulis, karena yang
    menentukan balon tercetak berbahasa Inggris atau tidak adalah bentuk akhir.
    """
    out = []
    for r in items:
        t = (got.get(r.idx) or "").strip()
        if not t or not _clean_translation(_restore_symbols(r.src_text, t)):
            out.append(r.idx)
    return out


def _translate_router(client, model: str, regions: list[Region],
                      items: list[Region], target_lang: str, style: str,
                      keep_honorifics: bool) -> list[Region]:
    """Terjemah lewat router, dengan tiga lapis penjaga panjang.

    1. PROMPT   — batas per balon ikut di JSON masukan, sebagai angka.
    2. VALIDASI — jawabannya diukur ulang dengan layout() sungguhan, tidak
                  dipercaya. Model bisa saja mengaku patuh dan tetap melanggar.
    3. PERBAIKAN— hanya baris yang MASIH melanggar dikirim ulang, dengan angka
                  pelanggarannya disebut. Mengirim ulang seluruh halaman membuat
                  model 'memperbaiki' baris yang sudah benar dan merusaknya.

    Lalu satu lapis lagi yang sifatnya berbeda: KELENGKAPAN. Ketiga lapis di
    atas menjaga jawaban yang ADA tetap muat; tidak satu pun menjaga jawabannya
    ADA. Terukur di hasilnew/13.JPG: balon 'えっ！？' terdeteksi (conf .807),
    terbaca OCR (ink .494), berlabel DIALOGUE, ikut terkirim — lalu model
    memutuskan seruan sependek itu tidak perlu diterjemahkan dan kuncinya
    hilang dari JSON. Loop lama menelan itu tanpa suara (`if not t: continue`),
    translation tetap None, render_region() keluar lebih awal, dan pembaca
    melihat satu balon Jepang di tengah halaman Inggris. Sekarang kunci yang
    hilang diminta ulang, dan kalau tetap hilang dicetak dengan sebutan idx-nya
    supaya tidak pernah lagi lolos tanpa terlihat.
    """
    use_budget = bool(SETTINGS.balloon_budget)
    budget = _page_budget(items) if use_budget else {}
    system = _system_prompt(target_lang, style, keep_honorifics, bool(budget))
    got: dict[int, str] = {}
    try:
        raw, used = _router_call_any(client, model or client.model, system,
                                     _user_prompt(items, budget or None))
        got = {int(k): str(v) for k, v in raw.items()}
    except Exception as exc:  # noqa: BLE001 - jaringan tidak boleh membunuh halaman
        note("error", "translate",
             f"{client.tag} gagal ({exc}); halaman keluar TANPA terjemahan "
             f"({len(items)} balon tetap berbahasa Jepang), teks asli di sidecar JSON")
        return regions

    rmap = {r.idx: r for r in items}
    if budget:
        for rnd in range(int(SETTINGS.budget_repair_rounds)):
            bad = _violations(got, budget, items)
            if not bad:
                break
            print(f"[budget] perbaiki {len(bad)} baris {sorted(bad)}")
            extra = ("\n\nREVISION. Your previous attempt broke the balloon budget "
                     "on these lines. Rewrite ONLY these, shorter, same meaning:\n"
                     + "\n".join(f'  "{i}": you wrote {got[i]!r} -> {why}'
                                 for i, why in sorted(bad.items())))
            sub = [rmap[i] for i in sorted(bad) if i in rmap]
            try:
                fix, _u = _router_call_any(client, used, system,
                                           _user_prompt(sub, budget) + extra)
            except Exception as exc:  # noqa: BLE001
                note("warn", "budget", f"revisi {rnd + 1} gagal ({exc}); pakai yang ada")
                break
            for k, v in fix.items():
                if int(k) in bad:
                    got[int(k)] = str(v)
        else:
            left = _violations(got, budget, items)
            if left:
                # Bukan kegagalan halaman: fit() masih akan mengecilkan fontnya.
                # Dicetak supaya balon yang mepet terlihat, bukan diam-diam kecil.
                note("warn", "budget",
                     f"{sorted(left)} masih melebihi balon setelah "
                     f"{SETTINGS.budget_repair_rounds} revisi; fit() yang menangani")

    # Lapis KELENGKAPAN. Dijalankan setelah ronde anggaran, bukan sebelum:
    # revisi anggaran bisa saja menjatuhkan kunci yang tadinya ada, jadi
    # pemeriksaan harus melihat keadaan terakhir. Yang dikirim ulang HANYA idx
    # yang kosong — sama seperti pola revisi anggaran di atas — dan promptnya
    # menyebut sebabnya, karena model yang menghapus 'えっ！？' melakukannya
    # dengan sengaja: ia butuh diberi tahu bahwa balon kosong itu tercetak.
    for rnd in range(int(SETTINGS.missing_repair_rounds)):
        miss = _missing_ids(got, items)
        if not miss:
            break
        print(f"[translate] {len(miss)} balon belum dijawab {miss}; minta ulang")
        sub = [rmap[i] for i in miss if i in rmap]
        if not sub:
            break
        extra = (
            "\n\nMISSING. Your previous reply had no usable text for these ids. "
            "Every one of them is a real speech balloon on the page: if you leave "
            "it out the reader sees raw Japanese. Answer ALL of them with a short "
            "non-empty English line — even a bare interjection gets one "
            "(えっ！？ -> \"HUH?!\"). Reply with JSON for these ids only:\n"
            + "\n".join(f'  "{r.idx}": {r.src_text!r}' for r in sub)
        )
        try:
            fix, _u = _router_call_any(client, used, system,
                                       _user_prompt(sub, budget or None) + extra)
        except Exception as exc:  # noqa: BLE001 - jaringan tidak boleh membunuh halaman
            note("warn", "translate", f"permintaan ulang gagal ({exc}); pakai yang ada")
            break
        for k, v in fix.items():
            try:
                ki = int(k)
            except (TypeError, ValueError):
                continue
            if ki in miss and str(v).strip():
                got[ki] = str(v)

    for r in items:
        t = (got.get(r.idx) or "").strip()
        if not t:
            continue
        r.translation = _clean_translation(_restore_symbols(r.src_text, t)) or None

    # Kalau setelah semua ronde masih ada yang kosong, itu HARUS terlihat.
    # Diam adalah cacatnya: satu balon Jepang di tengah halaman Inggris tidak
    # menghasilkan error apa pun, cuma hasil yang salah. Bukan exception —
    # tiga balon lain sudah benar dan halamannya tetap layak keluar — tapi
    # namanya disebut di log dan src_text-nya dikutip supaya bisa dicari.
    left = [r for r in items if not r.translation]
    if left:
        note("error", "translate", "BELUM DITERJEMAHKAN setelah "
             f"{SETTINGS.missing_repair_rounds} permintaan ulang: "
             + "; ".join(f"r{r.idx}={r.src_text!r}" for r in left))
    return regions
