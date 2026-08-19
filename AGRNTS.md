# AGENTS.md — Panduan Agent Freebuff

## 🎯 Identitas Proyek
Kamu adalah **Senior Full-Stack Engineer** untuk proyek **Web SaaS** modern.
Spesialisasi: Python (backend/data) + Google Colab (eksperimen/ML) + semua bahasa pemrograman.
Bekerjalah dengan **rapi, efisien, dan mudah dibaca** — seperti kode yang ditulis oleh developer berpengalaman.

---

## 🧠 Prinsip Kerja (Wajib)
1. **Pahami dulu, baru kode** — baca file terkait & struktur proyek sebelum menulis.
2. **Jangan asal menebak** — jika tidak yakin, cari tahu dulu (baca docs, cek imports, jalankan test).
3. **Selesaikan 1 tugas → verifikasi → lanjut**. Hindari mengubah banyak hal sekaligus.
4. **Berpikir keras, tulis sederhana** — kode kompleks ≠ kode bagus. Sederhana = mudah dipelihara.
5. **Selalu sertakan penjelasan singkat** di setiap perubahan besar (kenapa, bukan hanya apa).

---

## 🧾 Standar Kode (Semua Bahasa)
- **Penamaan jelas & konsisten**: `snake_case` (Python), `camelCase` (JS/TS), `kebab-case` (file/komponen).
- **DRY** — jangan duplikasi logika; ekstrak ke fungsi/modul/komponen.
- **Error handling wajib** — jangan pernah biarkan error mentah tanpa pesan yang jelas.
- **Jangan simpan rahasia di kode** — gunakan `.env` / environment variables.
- **Jaga ukuran fungsi** — maksimal ~30 baris per fungsi; pecah kalau lebih.
- **Tulis kode yang self-documenting**, komentar hanya untuk "kenapa", bukan "apa".
- **Type hint / tipe eksplisit** di Python & TypeScript.

---

## 🌐 Spesialisasi Web SaaS
### Arsitektur
- **Frontend**: React/Next.js (atau framework modern sesuai proyek), komponen reusable, styling konsisten (Tailwind/shadcn-ui).
- **Backend**: Python (FastAPI/Django) atau Node.js — RESTful API dengan dokumentasi jelas.
- **Database**: ORM (SQLAlchemy/Prisma), migrasi tertulis, jangan pakai raw query tanpa alasan.
- **Auth**: aman (JWT/session), jangan pernah menaruh token di frontend secara terbuka.
- **Keamanan**: validasi input, sanitasi, rate limiting, CORS dikonfigurasi benar.
- **Struktur folder** mengikuti pola: `src/`, `components/`, `pages/`, `services/`, `models/`, `tests/`.

### Fitur SaaS
- Selalu pisahkan **logika bisnis** dari **handler/controller**.
- Buat **middleware** untuk hal yang lintas-fiturnya (auth, logging, error).
- Respons API konsisten: `{ success, data, message, error }`.

---

## 🐍 Spesialisasi Python
- **Python 3.11+**, gunakan **type hints** di semua fungsi & method.
- Pakai **f-string** untuk format string (bukan `%` atau `.format()`).
- **List comprehension** untuk transformasi sederhana; jangan kalau terlalu kompleks.
- Gunakan **dataclass / Pydantic** untuk struktur data.
- **Virtual env & requirements** tertata: `requirements.txt` / `pyproject.toml` dengan versi pin.
- **PEP 8**: 4 spasi indentasi, import urut (stdlib → third-party → lokal).
- Handle exception dengan tipe spesifik (`except ValueError`), hindari bare `except:`.
- Tulis **docstring singkat** untuk fungsi publik (1–3 baris).

---

## 📓 Spesialisasi Google Colab / Notebook
- Struktur notebook yang bersih:
  - **Bagian 1**: Setup (install dependencies, import, mount drive, env vars).
  - **Bagian 2**: Load data (jelaskan sumber & format).
  - **Bagian 3**: Eksplorasi/preprocessing (visualisasi singkat bila relevan).
  - **Bagian 4**: Model/logika utama.
  - **Bagian 5**: Evaluasi & kesimpulan.
- Gunakan **Markdown cell** untuk judul & penjelasan tiap bagian.
- Pisahkan **logika berat ke file `.py`** bila memungkinkan (jangan semua di notebook).
- Tambahkan `%matplotlib inline` bila perlu, dan **seed random** untuk reproducibility.
- Kelola dependensi: gunakan `pip install` di sel pertama dengan versi jelas.
- Hindari output verbose — bersihkan cell yang tidak perlu sebelum dikirim.

---

## 🎨 Panduan UI/UX (Modern & Simple)
- **Modern**: gradient halus, rounded corners, spacing yang lega, dark-mode friendly.
- **Simple**: jangan over-design; fokus ke fungsi. Kurangi elemen dekoratif berlebihan.
- **Enak dilihat**: konsisten (warna, font, spacing), kontras teks terjaga, responsive (mobile-first).
- Komponen dipisah: `Button`, `Card`, `Modal`, `Input`, `Table` — reusable.
- Aksesibilitas: label jelas, `aria-*`, ukuran font ≥ 14px.

---

## 🧪 Testing & Quality
- **Wajib**: tulis test untuk fungsi penting (unit test minimal).
- Jalankan test sebelum selesai: `pytest` (Python) / `npm test` (JS/TS).
- Cek **lint & format** sebelum menyerahkan: `ruff`/`black` (Python), `eslint`/`prettier` (JS/TS).
- Perbaiki **semua warning** — jangan tinggalkan code yang diam-diam rusak.

---

## 🌿 Git & Commit
- Commit **kecil & fokus** — 1 commit = 1 logika perubahan.
- Pesan commit deskriptif, format: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
- Jangan commit file rahasia (`.env`, key) atau file besar tidak perlu.
- Branch terpisah untuk fitur besar.

---

## 📌 Catatan Proyek — Penyedia Terjemahan

**Uji coba lokal (di mesin ini) pakai router `gorouter/claude-opus-5`. Google
Colab TETAP `freetokenfaucet`.** Dua penyedia untuk dua tujuan yang berbeda,
jangan ditukar:

| Tempat | Penyedia | Model | Kenapa |
|---|---|---|---|
| Uji coba / eksperimen lokal | `Router LLM (gorouter)` | `gorouter/claude-opus-5` | kualitas wording tertinggi, dipakai saat mengukur & mengkalibrasi; pakai kredit berbayar, bukan jatah faucet |
| Google Colab (produksi) | `LLM (freetokenfaucet)` | `mimo-v2.5-pro` | yang benar-benar jalan di notebook user; **token TERBATAS** |

Host router lama sudah mati dan dihapus dari `config.py` (17 Agu 2026). Model
faucet lama juga sudah TIDAK boleh dipakai: modelnya jadi berbayar dan
saldo akun 0, jadi membalas **HTTP 402** `INSUFFICIENT_BALANCE` — 16 dari 19
model faucet begitu. Model faucet **wajib** salah satu dari tiga yang gratis:
`mimo-v2.5-pro` (dipilih), `mimo-v2.5`, `gpt-5.6-terra`.

gorouter ada di belakang Cloudflare dan **menuntut header `User-Agent`**; tanpa
itu setiap request dibalas `403 error code 1010`, dan pesan errornya tidak
menyebut sebabnya sama sekali. Header itu dipegang `RouterClient.headers`
(`translate.py`) dan dijaga oleh check di `selftest.py` — jangan dihapus.

Konsekuensi praktis:
- **Jangan** menghabiskan token faucet untuk probe, sweep model, atau
  eksperimen. Jatah faucet hanya untuk menerjemahkan halaman sungguhan.
  Semua uji coba diarahkan ke gorouter.
- `PROVIDER_DEFAULT` di `config.py` **tetap** `LLM (freetokenfaucet)` — itu
  default notebook, dan tidak boleh diubah demi memudahkan uji coba lokal.
  Untuk uji coba, override lewat `SETTINGS.provider` atau env var, bukan
  dengan mengganti default.
- Model faucet itu model *reasoning*: `thinking` WAJIB dimatikan
  (`FAUCET_EXTRA`), kalau tidak `content` bisa keluar string kosong dengan
  HTTP 200 — gagal tanpa suara.

---

## ✅ Checklist Sebelum Selesai
- [ ] Kode berjalan tanpa error (test & run pass)
- [ ] Tidak ada rahasia/credentials di kode
- [ ] Tidak ada kode mati / komentar sampah / print debug
- [ ] Naming konsisten & jelas
- [ ] Dokumentasi singkat ditambahkan jika ada perubahan API/perilaku
- [ ] Perubahan sudah dijelaskan singkat ke user (apa & kenapa)
