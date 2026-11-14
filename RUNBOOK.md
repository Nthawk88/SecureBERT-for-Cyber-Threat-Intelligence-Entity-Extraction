# RUNBOOK — Langkah Menjalankan Platform

Panduan operasional. Bagian **[SELESAI]** sudah dijalankan dan diverifikasi di
mesin ini. Bagian **[MANUAL]** masih perlu Anda kerjakan sendiri (butuh Docker,
jaringan, atau keputusan kualitas/penulisan).

Semua perintah di dokumen ini diasumsikan dijalankan dari root project dengan
interpreter virtualenv yang benar. Lihat bagian A untuk alasan kenapa venv-nya
berada di luar folder OneDrive.

```powershell
cd "c:\Users\user\OneDrive\Docouments\Pritesis\ThesisProject\Prithesis New"
$env:PYTHONPATH = (Get-Location).Path
$py = "C:\Users\user\.venvs\cti\Scripts\python.exe"
```

---

## Status saat ini

| Item | Status | Catatan |
|---|---|---|
| Struktur project + kode + tests | 21 unit test lulus |
| Environment GPU lokal | RTX 5060 Laptop (sm_120), torch 2.10.0+cu128 — lihat A |
| Dataset CyNER (Stream A) | `data/cyner/{train,dev,test}.conll` |
| Crawl penuh 5 sumber (Fase 1) | 80 artikel: talos 20, crowdstrike 20, bleepingcomputer 19, securelist 17, unit42 4 |
| Training multi-seed (Fase 2) | **10 seed**, 33 run, ~69 menit → `results/phase2_multiseed.csv` |
| Ablation study imbalance | 5 arm × 3 seed = 15 run → `results/ablation.csv` |
| Model final | SecureBERT + oversampling, seed 42 → `models/securebert_ner/` |
| Lapisan korelasi (Fase 3) | community detection Louvain → `results/phase3_casestudy.json` |
| Evaluasi kuantitatif (Fase 4) | `results/phase4_evaluation.json` + `confusion_matrix.png` |
| Notebook 01–05 | tereksekusi penuh, output tersimpan untuk lampiran |
| Neo4j (Docker) | opsional, hanya untuk eksplorasi visual — lihat C |
| Checklist QC manual artikel | lihat D |
| Penulisan BAB II, IV, V | lihat F |

### Angka kunci hasil akhir

| Metrik | Nilai | Sumber |
|---|---|---|
| Overall F1 SecureBERT vs baseline | +0,104, **p = 2,6 × 10⁻⁷** (n=10 seed) | `phase4_evaluation.json` |
| Vulnerability F1 model final | 0,889 (presisi 1,000, recall 0,800) | `phase4_evaluation.json` |
| Macro F1 / micro F1 model final | 0,779 / 0,771 | `phase4_evaluation.json` |
| Graf korelasi | 1.111 node, 39.590 edge, coverage 99,9% | `phase3_casestudy.json` |
| Modularitas Louvain | **0,785** (10 komunitas) | `phase3_casestudy.json` |
| Modularitas connected components | 0,0004 (2 komponen) — pembanding | `phase3_casestudy.json` |

> **Catatan dataset:** CyNER publik (`mitre`) berisi **4.372 kalimat**
> (train 2.811 / dev 813 / test 748). Proposal menyebut "5.164 record" — selisih
> ini wajar (versi/penghitungan berbeda) dan sebaiknya disebut di skripsi.

---

## A. [SELESAI] Environment GPU lokal

GPU mesin ini **NVIDIA RTX 5060 Laptop**, arsitektur Blackwell dengan compute
capability **sm_120**. Ini penting: PyTorch baru menyertakan kernel sm_120 sejak
versi 2.7.0 yang dibangun dengan CUDA 12.8. Build `cu121`/`cu124` akan selalu
gagal dengan `CUDA error: no kernel image is available for execution on the
device`, jadi **jangan** ikuti instruksi `cu121` mana pun yang masih beredar.

Venv sengaja diletakkan di **luar** folder OneDrive. OneDrive menyinkron ribuan
file kecil di `site-packages` dan checkpoint training berukuran gigabyte, yang
memperlambat training dan berisiko merusak file saat sinkronisasi berjalan.

```powershell
python -m venv C:\Users\user\.venvs\cti
& C:\Users\user\.venvs\cti\Scripts\python.exe -m pip install `
    torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
& C:\Users\user\.venvs\cti\Scripts\python.exe -m pip install -r requirements-local.txt
```

Verifikasi harus benar-benar menjalankan kernel di GPU, bukan sekadar mengecek
`torch.cuda.is_available()` yang bisa `True` padahal kernelnya tidak ada:

```powershell
& $py -c "import torch; a=torch.randn(2048,2048,device='cuda'); print((a@a).sum())"
```

Terpasang saat ini: Python 3.13.7, torch 2.10.0+cu128, transformers 5.16.1,
networkx 3.6.1. Dokumentasi lengkapnya ada di `requirements-local.txt`.
`requirements.txt` dipertahankan sebagai referensi versi asli (torch 2.5.1) dan
**tidak** cocok untuk GPU ini.

> **Saran saat training:** jeda OneDrive selama training berjalan, dan bersihkan
> `models/checkpoints/` setelahnya — folder itu bisa menumpuk beberapa GB berisi
> `optimizer.pt` yang tidak diperlukan setelah model final tersimpan.

---

## B. [SELESAI] Menjalankan ulang seluruh pipeline

Urutan ini yang menghasilkan seluruh angka di tabel status. Jalankan berurutan;
setiap tahap memakai output tahap sebelumnya.

```powershell
# Fase 2 — perbandingan multi-seed + pemilihan gamma (~69 menit, 33 run)
& $py scripts/run_phase2.py --gamma-search

# Ablation — 5 strategi imbalance × 3 seed (~38 menit, 15 run)
& $py scripts/run_ablation.py

# Model final — strategi pemenang ablation, seed dikunci ke project.seed (~5 menit)
# 10 epoch / 5e-5 dipilih dari validation macro F1 (results/tuning_securebert.csv)
& $py scripts/train_final_model.py --epochs 10 --lr 5e-5

# Fase 3 — NER pada artikel hasil crawl -> graf korelasi (~25 detik)
& $py scripts/run_phase3.py --refresh

# Fase 4 — evaluasi CyNER test + rekap t-test + overhead (~30 detik)
& $py scripts/run_phase4.py

# Opsional — BERT-base-uncased + oversampling sebagai pembanding (~20 menit, 6 run)
& $py scripts/run_ablation.py --model baseline --arms ce oversample

# Opsional — kedua model + oversampling, 10 seed, satu CSV (~52 menit, 20 run)
$s = 42,123,456,789,1011,1213,1415,1617,1819,2021
& $py scripts/run_ablation.py --arms oversample --seeds $s --out oversample_10seeds.csv --drop-checkpoints --resume
& $py scripts/run_ablation.py --model baseline --arms oversample --seeds $s --out oversample_10seeds.csv --drop-checkpoints --resume

# Opsional — ulang Fase 2 dengan skor per-kelas ke CSV terpisah (~95 menit, 30 run)
& $py scripts/run_phase2.py --out phase2_perclass.csv --drop-checkpoints --resume

# Rekap semua perbandingan BERT vs SecureBERT (beberapa detik, tanpa training)
& $py scripts/compare_models.py

# Perbandingan dengan angka TSTEM (Table 6). Tuning dipilih dari VALIDATION macro F1
# (results/tuning_securebert.csv): 10 epoch, lr 5e-5. Setiap baris ~2,5-5 menit x 10 seed.
& $py scripts/run_ablation.py --arms oversample --seeds 42 123 --epochs 10 --lr 5e-5 --out tuning_securebert.csv --drop-checkpoints --resume
$o = "tstem_comparison_runs.csv"
& $py scripts/run_ablation.py --arms oversample --seeds $s --epochs 10 --lr 5e-5 --out $o --drop-checkpoints --resume
& $py scripts/run_ablation.py --model baseline --arms oversample --seeds $s --epochs 10 --lr 5e-5 --out $o --drop-checkpoints --resume
& $py scripts/run_ablation.py --model baseline --arms ce --seeds $s --out $o --drop-checkpoints --resume
# protokol TSTEM: split acak per seed (--random-split)
& $py scripts/run_ablation.py --model baseline --arms ce --seeds $s --random-split --out $o --drop-checkpoints --resume
& $py scripts/run_ablation.py --arms oversample --seeds $s --epochs 10 --lr 5e-5 --random-split --out $o --drop-checkpoints --resume
# -> results/tstem_proof_per_seed.csv + results/tstem_proof_summary.csv
& $py scripts/tstem_proof.py
```

`run_phase2.py` mendukung `--resume` untuk melewati kombinasi arm/seed yang sudah
ada di CSV, berguna jika training terputus.

### Pemisahan tanggung jawab yang penting

**Model final hanya boleh berasal dari `scripts/train_final_model.py`.** Skrip itu
mengunci seed ke `project.seed` (42) dan memakai strategi pemenang ablation, agar
model yang dipakai Fase 3 dan 4 bukan hasil memilih seed terbaik.

`run_phase2.py` menyimpan run `securebert_focal` terbaiknya ke
`models/checkpoints/securebert_focal_best/` **untuk inspeksi saja**, sengaja
bukan ke `models/securebert_ner/`. Memilih seed dengan Vulnerability F1 tertinggi
adalah cherry-picking, terutama karena metrik itu berayun dari 0,00 sampai 0,82
antar seed.

### Detail tiap tahap

**Fase 2** (`scripts/run_phase2.py`) — tiga arm yang sebanding, masing-masing 10 seed:
- `baseline_ce` — bert-base-uncased, cross-entropy biasa (baseline gaya TSTEM)
- `baseline_focal` — bert-base-uncased + Focal Loss, mengisolasi efek fungsi loss
- `securebert_focal` — SecureBERT + Focal Loss, mengisolasi efek encoder
- Ditambah arm `gamma_search` (γ ∈ {1,2,3}) pada seed 42 sebagai uji sensitivitas
  (γ = 2 sudah ditetapkan di config sebelumnya, bukan dipilih dari sweep ini).
- Output: `results/phase2_multiseed.csv` (kolom pembeda arm adalah **`arm`**,
  bukan `model`; `model` hanya mencatat encoder-nya).

**Ablation** (`scripts/run_ablation.py`) — 5 arm × 3 seed pada SecureBERT:
tanpa mitigasi (CE), focal loss, class weighting, augmentasi, dan oversampling.
Augmentasi dan oversampling **hanya** menyentuh train split; validation dan test
mempertahankan distribusi alaminya. Output: `results/ablation.csv`.

**Fase 3** (`scripts/run_phase3.py`) — artikel → NER → graf korelasi:
- Cache entitas di `results/phase3_entities.json`; pakai `--refresh` untuk
  ekstraksi ulang (wajib setelah model berubah).
- Filter kualitas: confidence ≥ 0,7, panjang ≥ 3 karakter, validasi `Indicator`
  terhadap pola IoC kanonik (IP/URL/hash/CVE/email/domain), dan hub-filter
  (`--hub-docfreq`, default 0,20).
- Clustering memakai **community detection Louvain** (`--cluster-method louvain`,
  default). Connected components tersedia via `--cluster-method components` dan
  **selalu ikut dilaporkan** di JSON sebagai pembanding, sehingga pilihan metode
  bisa diaudit penguji.
- Opsi lain: `--no-temporal` (document co-occurrence saja), `--loose-ioc` (tanpa
  validasi IoC, menampilkan hasil naive), `--resolution` (resolusi Louvain).
- Output: `data/correlation/campaign_graph.graphml`, `before_after.png`,
  `results/phase3_casestudy.json`, `results/phase3_campaigns.json`.

**Fase 4** (`scripts/run_phase4.py`) — evaluasi kuantitatif:
- Prediksi model final pada CyNER **test**, per-class P/R/F1 (seqeval), confusion
  matrix antar-tipe entitas, rekap paired t-test dari `phase2_multiseed.csv`, dan
  overhead dari Fase 3.
- Output: `results/phase4_evaluation.json` + `results/confusion_matrix.png`.

**Perbandingan model** (`scripts/compare_models.py`) — hanya membaca hasil tersimpan:
- Uji-t berpasangan BERT vs SecureBERT dengan loss yang sama (`baseline_focal` vs
  `securebert_focal`), vs baseline CE, per kelas (dari `phase2_perclass.csv`), dan
  dengan oversampling (dari `ablation.csv` + `ablation_bert.csv`).
- Bagian `reproducibility` membandingkan run ulang per-kelas dengan
  `phase2_multiseed.csv`. Training di GPU tidak identik bit-per-bit walau seed sama;
  Overall F1 stabil (selisih rata-rata ≤ 0,002), Vulnerability F1 tidak (10 entitas
  di test set, satu entitas ≈ 0,1 F1).
- Output: `results/model_comparison.json`.

---

## C. [MANUAL] Neo4j — opsional, hanya untuk eksplorasi visual

1. Buka **Docker Desktop**, tunggu sampai status "Engine running".
2. Lalu:

```powershell
cd docker
docker compose up -d
```

3. Buka <http://localhost:7474> — login `neo4j` / `ctiresearch`.

Tidak wajib. Bila Neo4j tidak hidup, Fase 3 otomatis fallback ke ekspor
**GraphML** (`data/correlation/campaign_graph.graphml`) yang bisa dibuka di Gephi
atau Cytoscape, plus `before_after.png` yang sudah dihasilkan otomatis.

---

## D. [MANUAL] Fase 1 — crawl dan QC

Crawl penuh sudah dijalankan (80 artikel). Untuk menjalankan ulang:

```powershell
& $py -m src.collection.run_crawl --all
& $py -m src.collection.qc
```

**Status sumber:**

| Sumber | Status | Catatan |
|---|---|---|
| `talos` | bagus | artikel CTI bersih, judul dan tanggal benar |
| `unit42` | bagus | artikel panjang dan relevan, tetapi hanya 4 artikel terambil |
| `bleepingcomputer` | bagus | berita security, bersih |
| `securelist` | bagus | **pengganti Mandiant** (Kaspersky), server-rendered |
| `crowdstrike` | sebagian | konten bersih tetapi banyak post marketing dan `date` kosong |
| `mandiant` | gagal | `cloud.google.com` dirender JavaScript, Scrapy tidak bisa mengekstrak link (perlu JS renderer; **di luar scope**) |

**Dokumentasikan di skripsi** bahwa Mandiant diganti Securelist karena blog
Mandiant kini berbasis JS di Google Cloud — sesuai catatan proposal bahwa
selector dan sumber dapat berubah.

**Yang masih perlu Anda kerjakan:**
- Isi **checklist QC manual** (6 poin) untuk sampel artikel.
- Putuskan status artikel Securelist yang ditandai QC.
- 20 dari 80 artikel tidak punya tanggal publikasi yang bisa dipercaya; artikel
  itu otomatis dikecualikan dari edge temporal agar tidak menimbulkan klaster
  semu. Sebutkan angka ini di Bab III atau IV.
- Untuk `crowdstrike`, jika tanggal dibutuhkan, sesuaikan `date_selectors` di
  `src/collection/spiders/crowdstrike_spider.py` (markup mereka berubah-ubah).

---

## E. [SELESAI] Notebook 01–05 untuk lampiran

Kelima notebook sudah tereksekusi dengan output tersimpan (24 sel kode, 4 gambar,
nol error). Menjalankan ulang:

```powershell
& $py -m ipykernel install --user --name cti --display-name "Python (cti)"
foreach ($n in @('01_data_exploration','02_baseline_replication','03_securebert_training','04_ablation_study','05_correlation_casestudy')) {
  & $py -m jupyter nbconvert --to notebook --execute --inplace `
      --ExecutePreprocessor.kernel_name=cti --ExecutePreprocessor.timeout=3600 "notebooks\$n.ipynb"
}
```

> Jangan set `MPLBACKEND=Agg` saat eksekusi, karena gambar tidak akan terekam ke
> dalam notebook.

**Pembagian peran notebook versus skrip.** Notebook 03, 04, dan 05 **membaca
artefak** yang dihasilkan skrip, tidak melatih ulang model. Ini disengaja: versi
lamanya melatih ulang dan menimpa `results/phase2_multiseed.csv` serta
`models/securebert_ner/`, sehingga angka di naskah bisa berbeda dari angka di
lampiran. Dengan pola sekarang, skrip adalah satu-satunya sumber kebenaran dan
lampiran dijamin konsisten. Notebook asli tersimpan di
`notebooks/_backup_prescript/`.

Notebook 02 tetap melatih baseline sungguhan (~3 menit) karena hanya menulis ke
`models/bert_baseline/` dan tidak mengganggu artefak lain.

---

## F. [MANUAL] Sisa pekerjaan penulisan dan kerapian

- **Kurasi noise NER** sebelum gambar graf masuk Bab IV: `Bitwarden` dan `Log4j`
  tersalah-klasifikasi sebagai Malware, ada duplikat `ABCDoor` versus `ABCDoor.`,
  dan nilai seperti `zero-day vulnerabilities` yang bukan CVE.
- **Format `docs/RESULTS.md`** menjadi BAB IV dan BAB V terpisah sesuai format
  kampus, tambahkan Abstrak dan Daftar Pustaka. Banyak angka di dokumen itu
  sekarang sudah kedaluwarsa.
- **Tulis BAB II Tinjauan Pustaka.**
- **Inisialisasi git + `.gitignore`** untuk mengecualikan bobot model; folder ini
  masih belum berversi sama sekali.

---

## Ringkasan perintah verifikasi

```powershell
& $py -m pytest tests/ -q                     # 21 unit test
& $py scripts/smoke_test_training.py          # pipeline training, cepat
& $py -m src.collection.run_crawl --source talos --limit 3 --timeout 120
& $py -m src.collection.qc
```

## Revisi (permintaan reviewer): eksperimen & analisis tambahan

Semua keluaran baru ditulis ke berkas baru (`results/rev_*.csv`, `results/revision_*.json`); tidak ada hasil lama yang diubah.

```powershell
$py = "C:\Users\user\.venvs\cti\Scripts\python.exe"
# Training tambahan (GPU, beberapa jam): SecureBERT CE (tuned, 10 seed) + diagnostik distribusi label,
# RoBERTa-base + oversampling (tuned, 10 seed), pencarian tuning 4 kandidat untuk BERT-base & RoBERTa-base
powershell -File scripts/run_revision_experiments.ps1
powershell -File scripts/run_revision_experiments2.ps1   # BERT-base CE (tuned, 10 seed)

# Analisis CPU
& $py scripts/revision_stats.py                 # ringkasan + uji berpasangan -> results/revision_stats.json
& $py scripts/revision_bootstrap.py             # bootstrap CI (level seed & level kalimat) -> results/revision_bootstrap.json
& $py scripts/revision_correlation_analysis.py  # ablasi tipe edge, jendela waktu, baseline, korpus -> results/revision_correlation.json,
                                                #   results/revision_corpus.json, data/scraped/article_manifest.csv
& $py scripts/export_split_manifest.py          # results/split_manifest.csv
```

`scripts/run_ablation.py` kini mendukung `--model roberta` dan menulis kolom `diag_*`
(jumlah token Vulnerability yang diprediksi vs gold, probabilitas rata-rata pada token gold).
