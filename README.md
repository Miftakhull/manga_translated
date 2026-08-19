# manga_translated

Penerjemah manga Jepang → bahasa lain untuk dijalankan di Google Colab:
deteksi balon → OCR → terjemah → hapus tinta Jepang → tata ulang huruf Inggris
di dalam balon yang sama.

Repo ini **tidak memuat satu pun aset biner**. Bukan karena lupa — kodenya
memang mengunduh semuanya saat runtime. Lihat [Yang sengaja tidak
ada](#yang-sengaja-tidak-ada-di-repo-ini) sebelum menyimpulkan ada yang
hilang.

## Menjalankan (Colab)

Buka `2.ipynb` di Colab dan jalankan selnya **berurutan**. Dua sel butuh
perhatian:

| Sel | Kenapa perlu diperhatikan |
|---|---|
| 4 | **WAJIB restart runtime** setelah sel 3 memasang dependensi |
| 20 | Mengunduh weight + font. Idempoten — aman dijalankan ulang |
| 21 | Probe API, berdiri sendiri. Di sinilah kunci penyedia dibaca |
| 24 | Menjalankan UI; link share hidup 72 jam |

Sel 23 menjalankan self-test tanpa masukan user (notebook menggambar sendiri
halaman ujinya), dan sel 26 mengaudit kebersihan balon tanpa memakai token.

## Susunan repo

```
_nbsrc/*.py     SUMBER KEBENARAN kode pipeline (14 modul)
2.ipynb         notebook Colab — sel %%writefile-nya CERMIN dari _nbsrc/
sync_nbsrc.py   mencerminkan _nbsrc/ ke 2.ipynb, dan --check memverifikasinya
verify_local.py rangkaian uji offline; halaman ujinya sintetis, tanpa ONNX
_*.py           probe pengukur (lihat di bawah)
probe_*.py      probe generasi lebih awal
```

Aturan yang penting: **sunting `_nbsrc/`, jangan sel notebook.** Sel
`%%writefile` di `2.ipynb` dihasilkan ulang oleh `sync_nbsrc.py`, jadi
suntingan langsung di notebook akan tertimpa.

```bash
python sync_nbsrc.py --notebook 2.ipynb          # cerminkan
python sync_nbsrc.py --notebook 2.ipynb --check  # exit 0 = 14 sel sinkron
```

### Probe `_*.py`

Probe men-stage `_nbsrc/` ke `.stage/` (membuang baris `%%writefile`), lalu
mengukur **satu** pertanyaan dan mencetak angkanya. Probe tidak ikut ke
notebook. Setiap perbaikan cacat di repo ini punya probe pendampingnya — itu
catatan pengukurannya, bukan sekadar coretan.

Probe yang memakai `detect.detect()` butuh weight ONNX. `verify_local.py`
tidak: halamannya digambar sendiri, jadi bisa jalan di mesin bersih.

```bash
python verify_local.py    # -> "HASIL: SEMUA LOLOS"
```

## Yang sengaja tidak ada di repo ini

| Tidak ada | Kenapa | Dari mana didapat |
|---|---|---|
| `weights/` (≈447 MB) | `lama_large_512px.ckpt` sendiri 195 MB, di atas batas keras GitHub 100 MB/file | `assets.py:download_weights()` mengunduh dari `WEIGHT_URLS` (`config.py`), tiap weight punya rantai mirror |
| `fonts/` | Kode mengunduh semuanya saat runtime. Untuk `anime_ace.ttf` membundelnya juga **dilarang** — lisensi Blambot melarang redistribusi | `FONT_CHAIN` / `FONT_SHOUT` / `FONT_SYMBOL` / `FONT_CJK` di `config.py`; sel 20 memanggilnya |
| halaman manga | Scan berhak cipta, dan repo ini publik | — |
| berkas kunci API | Rahasia tidak pernah masuk riwayat git | kamu sendiri, lihat di bawah |

Weight yang diunduh: `detector.onnx` (RT-DETR-v2, 3 kelas),
`comictextdetector.pt.onnx` (block head + soft mask + line head), dan
`lama_large_512px.ckpt` (LaMa, di-finetune anime/manga).

## Kunci penyedia

Kunci dibaca dari berkas teks di direktori kerja (`MANGATL_WORK`, lokal =
akar repo). Semuanya ter-`.gitignore`; buat sendiri, jangan pernah di-commit:

| Berkas | Bentuk yang diharapkan kode |
|---|---|
| `gorouter.txt` | baris `set` gaya Windows berisi `ANTHROPIC_AUTH_TOKEN` dan `ANTHROPIC_MODEL` |
| `freetokenfaucet.txt` | potongan Python berisi `api_key="tf_..."` |
| `test.txt` | baris ke-3 berisi key mentah — jalur kedua untuk konfigurasi router |

Dua catatan dari kodenya sendiri: router **mewajibkan header `User-Agent`**
(faucet tidak), dan model default faucet adalah `mimo-v2.5-pro` karena model
lain mengembalikan 402.

**Jangan pernah menaruh token langsung di dalam berkas kode.** Repo ini punya
riwayatnya: sebuah skrip probe menyimpan token Bearer ter-hardcode dan hampir
ikut terbit; gerbang pra-commit sekarang menyapu **bentuk** kunci di seluruh
berkas ter-stage, bukan cuma berkas yang namanya mencurigakan. Alasan
lengkapnya ada sebagai komentar di `.gitignore`.

## Jalur konfigurasi

| Env | Bawaan | Isinya |
|---|---|---|
| `MANGATL_ROOT` | `/content/mangatl` | modul pipeline hasil `%%writefile` |
| `MANGATL_WORK` | `/content/work` | `weights/`, `fonts/`, `output/`, `debug/`, berkas kunci |

Untuk menjalankan probe secara lokal, arahkan keduanya ke akar repo — itu yang
dilakukan tiap probe `_*.py` di baris pembukanya.
