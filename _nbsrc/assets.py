%%writefile /content/mangatl/assets.py

"""Unduh weight model. Idempoten — file yang sudah ada dilewati."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from config import WEIGHT_URLS, WEIGHTS

_MIN_BYTES = 1_000_000  # weight terkecil ~94 MB; file kecil = halaman error HTML


def _fetch(url: str, dest: Path, chunk: int = 1 << 20) -> bool:
    """Unduh streaming supaya file 200 MB tidak menghabiskan RAM."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = done * 100 // total
                    print(f"\r  {dest.name}: {pct:3d}%  ({done >> 20} MB)", end="")
        print()
        if tmp.stat().st_size < _MIN_BYTES:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
        print(f"\n  {dest.name}: GAGAL ({exc})")
        tmp.unlink(missing_ok=True)
        return False


def download_weights(verbose: bool = True) -> dict[str, bool]:
    """Return {nama_file: tersedia}. Pipeline tetap jalan walau sebagian gagal.

    Tiap weight punya rantai mirror; mirror berikutnya dicoba hanya kalau yang
    sebelumnya gagal, jadi jalur normal tetap satu request per file.
    """
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    status: dict[str, bool] = {}
    for name, urls in WEIGHT_URLS.items():
        dest = WEIGHTS / name
        if dest.exists() and dest.stat().st_size >= _MIN_BYTES:
            status[name] = True
            if verbose:
                print(f"  {name}: sudah ada ({dest.stat().st_size >> 20} MB)")
            continue
        status[name] = any(_fetch(u, dest) for u in urls)
    return status

