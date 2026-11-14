"""Jalankan model NER final pada teks apa pun dan tampilkan entitas yang ditemukan.

Ini cara termudah membuktikan modelnya bekerja tanpa menjalankan pipeline apa pun.
Model yang dipakai adalah `models/securebert_ner/`, yaitu model final hasil
`scripts/train_final_model.py`.

Contoh pemakaian:

    # langsung dari argumen
    python scripts/predict_text.py --text "APT28 exploited CVE-2021-4034 on FortiOS."

    # dari sebuah file teks
    python scripts/predict_text.py --file artikel.txt

    # dari salah satu artikel hasil crawl
    python scripts/predict_text.py --article 0

    # keluaran JSON supaya bisa diolah skrip lain
    python scripts/predict_text.py --text "..." --json

Kolom `confidence` adalah keyakinan model (0 sampai 1). Fase 3 hanya memakai
entitas dengan confidence >= 0.7; di sini semuanya ditampilkan agar terlihat apa
yang sebenarnya dibuang oleh filter itu.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.preprocessing.text_cleaning import clean_text
from src.ner.inference import EntityExtractor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ekstraksi entitas CTI dari teks memakai model NER final.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="teks yang mau dianalisis, langsung di sini")
    src.add_argument("--file", help="path ke file teks (.txt)")
    src.add_argument("--article", type=int,
                     help="indeks artikel hasil crawl di data/scraped/raw")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="sembunyikan entitas di bawah nilai ini (default 0 = tampilkan semua)")
    parser.add_argument("--json", action="store_true", help="keluarkan JSON, bukan tabel")
    args = parser.parse_args()

    cfg = load_config()

    if args.text:
        text = args.text
        origin = "argumen --text"
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            sys.exit(f"File tidak ditemukan: {path.resolve()}")
        raw = path.read_bytes()
        if not raw.strip():
            sys.exit(f"File {path.resolve()} kosong (0 karakter di disk). "
                     f"Kalau teksnya terlihat di editor, simpan dulu (Ctrl+S).")
        # PowerShell 5 `>` / Out-File writes UTF-16 with a BOM
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = raw.decode("utf-16")
        else:
            text = raw.decode("utf-8-sig", errors="replace")
        origin = args.file
    else:
        files = sorted(glob.glob(
            os.path.join(str(cfg.path("paths.scraped_raw")), "**", "*.json"),
            recursive=True))
        if not files:
            sys.exit("Tidak ada artikel hasil crawl. Jalankan dulu: "
                     "python -m src.collection.run_crawl --all")
        if not 0 <= args.article < len(files):
            sys.exit(f"--article harus antara 0 dan {len(files) - 1} "
                     f"(ada {len(files)} artikel).")
        with open(files[args.article], "r", encoding="utf-8") as fh:
            art = json.load(fh)
        text = art.get("content", "")
        origin = f"{art.get('source')} — {str(art.get('title'))[:70]}"

    text = clean_text(text)
    if not text.strip():
        sys.exit("Teksnya kosong setelah dibersihkan (isinya hanya HTML/spasi/"
                 "karakter yang dibuang oleh clean_text).")

    model_dir = cfg.path("paths.securebert_dir")
    if not (model_dir / "config.json").exists():
        sys.exit(f"Model belum ada di {model_dir}. Jalankan dulu: "
                 f"python scripts/train_final_model.py")

    extractor = EntityExtractor(model_path=str(model_dir)).load()
    entities = [e for e in extractor.extract_unique(text)
                if e["confidence"] >= args.min_confidence]
    entities.sort(key=lambda e: (e["type"], -e["confidence"]))

    if args.json:
        print(json.dumps(entities, indent=2, ensure_ascii=False))
        return

    print(f"\nSumber teks : {origin}")
    print(f"Panjang     : {len(text)} karakter")
    print(f"Model       : {model_dir}")
    print(f"Entitas unik: {len(entities)}\n")

    if not entities:
        print("Tidak ada entitas yang ditemukan.")
        return

    width = max(len(e["value"]) for e in entities)
    width = min(max(width, 5), 60)
    print(f"{'TIPE':<14} {'NILAI':<{width}} CONFIDENCE")
    print("-" * (14 + width + 12))
    for e in entities:
        value = e["value"] if len(e["value"]) <= width else e["value"][:width - 1] + "…"
        print(f"{e['type']:<14} {value:<{width}} {e['confidence']:.3f}")

    counts: dict = {}
    for e in entities:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    print("\nRingkasan per tipe:")
    for t, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {t:<14} {n}")


if __name__ == "__main__":
    main()
