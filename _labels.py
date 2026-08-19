"""Cetak label tiap region dari sidecar JSON — mencari jejak cacat #2."""
import json

for f in ("output/jp_13.json", "output/jp_6.json", "output/jepang_002.json"):
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception as e:                      # noqa: BLE001
        print(f, "ERR", e)
        continue
    print(f"== {f} regions={d.get('region_count')} "
          f"transl={d.get('translated_count')} "
          f"untransl={d.get('untranslated_idx')} sfx={d.get('sfx_idx')}")
    for r in d.get("regions", []):
        print("   r%-2s %-10s conf=%-4s bub=%-5s src=%r -> %r" % (
            r.get("idx"), r.get("label"), r.get("label_conf"),
            bool(r.get("bubble_bbox")), (r.get("src_text") or "")[:26],
            (r.get("translation") or "")[:26]))
