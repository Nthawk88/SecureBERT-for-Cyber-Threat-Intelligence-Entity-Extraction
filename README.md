# SecureBERT for Cyber Threat Intelligence Entity Extraction

A matched comparison of SecureBERT, BERT-base, and RoBERTa-base for named-entity recognition on [CyNER](https://github.com/aiforsec/CyNER), plus an exploratory Indicator-of-Compromise (IoC) co-occurrence graph.

The reference system is TSTEM (Balasubramanian et al., *Decision Analytics Journal*, 2025). TSTEM reports NER category accuracy of 98.74%, with per-class F1 of 0.90 (Malware), 0.84 (Indicator and Organization), 0.73 (System), and 0.48 (Vulnerability). This repository covers two parts of that pipeline: entity extraction, and a document-and-temporal co-occurrence graph. It does not reimplement TSTEM's crawler classifiers or claim to be a complete CTI platform.

**Authors.** Aurelio Suhartono, Dwi Arnanta Saputra, Enrico Patras, and Aditya Kurniawan, Bina Nusantara University.

## Results to cite

All NER scores below are means over ten seeds (42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021). The primary comparison gives every encoder the same oversampling and the same search budget. Cite these ten-seed means. The saved seed-42 checkpoint is one run, and its Vulnerability F1 should not be quoted on its own.

| Comparison | Result |
|---|---|
| BERT-base baseline, default schedule, cross-entropy | Vulnerability F1 0.482 ± 0.241 (TSTEM reports 0.48); macro F1 0.588; micro F1 0.645 |
| SecureBERT vs BERT-base, both with oversampling, tuned schedule | Micro F1 0.768 vs 0.683; macro F1 0.759 vs 0.684; higher on all four majority classes on every seed (paired *p* < 10⁻⁴) |
| Vulnerability, same matched setting | 0.783 ± 0.087 (SecureBERT) vs 0.786 ± 0.101 (BERT-base), *p* = 0.94. The test split has about 10 Vulnerability entities |
| RoBERTa-base control, same oversampling and schedule | Micro F1 0.731 (0.744 with RoBERTa's own selected schedule) |
| SecureBERT with no imbalance mitigation | No Vulnerability token predicted under the default 5-epoch schedule. Tuned schedule: 0.682 ± 0.260. Oversampling changes that mean by +0.101 (*p* = 0.26) and mainly reduces the spread |
| TSTEM's own protocol (random re-split, token-level F1) | Above the published F1 on Malware 0.927 vs 0.90, Indicator 0.949 vs 0.84, System 0.842 vs 0.73, Vulnerability 0.642 vs 0.48, and macro F1 0.833 vs 0.758. Below it on Organization, 0.807 vs 0.84. Descriptive only: TSTEM publishes one score per class, and 63% of test entity strings also occur in training under this split, against 35% on the official split |
| Co-occurrence graph, 80 articles | 1,111 nodes, 10 Louvain communities, modularity 0.785, coverage 99.9%, 3.8 ms of graph construction per article |

The graph is an exploratory grouping. Article and outlet baselines reach modularity 0.504 and 0.412, so part of the structure is article membership. There is no gold campaign annotation.

Per-seed tables, bootstrap intervals, and the graph sweep are in `results/`. Start with `oversample_10seeds.csv`, `rev_securebert_ce_tuned.csv`, `rev_baseline_os_best.csv`, `rev_roberta_os_tuned.csv`, `rev_roberta_os_best.csv`, `revision_stats.json`, `revision_bootstrap.json`, `revision_correlation.json`, `tstem_proof_summary.csv`, and `split_manifest.csv`.

## Repository layout

```
config/config.yaml     hyperparameters and paths
src/                   crawler, preprocessing, NER, graph, evaluation
scripts/               phase runners and the revision analyses
tests/                 unit tests
results/               per-seed tables and JSON summaries
data/scraped/article_manifest.csv
data/correlation/before_after.png
docker/                optional Neo4j Compose file
requirements.txt       original pins (torch 2.5.1; see Setup)
```

## Setup

Training was run on an NVIDIA RTX 5060 Laptop (Blackwell, `sm_120`) in fp32. PyTorch builds older than 2.7 have no kernel for that GPU. `requirements.txt` pins `torch==2.5.1` and is the original record; on this machine the working stack is Python 3.13, `torch==2.10.0+cu128`, and `transformers==5.16.1`.

Create the virtual environment outside any synced folder (OneDrive, Dropbox). Checkpoints write multi-gigabyte optimizer states, and a sync client can corrupt them.

```powershell
python -m venv C:\Users\<user>\.venvs\cti
C:\Users\<user>\.venvs\cti\Scripts\python.exe -m pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
C:\Users\<user>\.venvs\cti\Scripts\python.exe -m pip install -r requirements.txt
```

If `requirements-local.txt` is present, install that after the CUDA wheel instead of `requirements.txt`. Confirm that a kernel actually runs:

```powershell
python -c "import torch; a=torch.randn(2048,2048,device='cuda'); print((a@a).sum())"
```

`torch.cuda.is_available()` can be true when no compatible kernel exists. Batch size 16 and a maximum length of 256 use about 5.5 GB of VRAM.

### CyNER

The labelled corpus is not in this repository. Download it from https://github.com/aiforsec/CyNER and place the CoNLL files at:

```
data/cyner/train.conll
data/cyner/dev.conll
data/cyner/test.conll
```

The official split has 4,372 sentences: 2,811 train, 813 development, 748 test. Vulnerability has 48 training entities and 10 in the test split.

### Crawled articles

Full article texts are not released. `data/scraped/article_manifest.csv` lists the 80 articles (URL, outlet, publication date, crawl time, and counts): 20 Talos, 20 CrowdStrike, 19 BleepingComputer, 17 Securelist, and 4 Unit 42. The crawl is used only for the graph. NER metrics come from the CyNER test split.

### Saved model

`models/securebert_ner/` is the seed-42 SecureBERT checkpoint with oversampling (10 epochs, learning rate 5×10⁻⁵). It is about 477 MB, above GitHub's file limit, so it is not in this repository. Per-seed checkpoints are not released.

## Running the pipeline

From the repository root, with `PYTHONPATH` set to that root:

```powershell
python -m src.collection.run_crawl --all
python -m src.collection.qc
python scripts/run_phase2.py --gamma-search
python scripts/run_ablation.py
python scripts/train_final_model.py
python scripts/run_phase3.py --refresh
python scripts/run_phase4.py
```

`train_final_model.py` is what writes `models/securebert_ner/`. The revision scripts (`scripts/revision_stats.py`, `revision_bootstrap.py`, `revision_correlation_analysis.py`, and `run_revision_experiments*.ps1`) reproduce the matched-encoder tables, the bootstrap intervals, and the graph sensitivity sweep.

Neo4j is optional and is only a viewer for the graph:

```powershell
cd docker
docker compose up -d
```

Browser UI: http://localhost:7474. Bolt: `bolt://localhost:7687`. Credentials are in `config/config.yaml` under `correlation.neo4j` (`neo4j` / `ctiresearch`).

Unit tests:

```powershell
python -m pytest tests/ -q
```

## Reading the results

Two data streams are kept apart.

| Stream | What it is | Used for |
|---|---|---|
| CyNER | Labelled IOB corpus, five entity types | Every NER number |
| Clear-web crawl | 80 articles, manifest only in this repo | The co-occurrence graph |

Three evaluation settings are named in the paper.

- **P1-ent** is the official split, scored at entity level with seqeval. Micro F1 pools all entities. Macro F1 is the unweighted mean of the five class scores.
- **P1-tok** is the official split, scored per word with B- and I- tags merged.
- **P2** is TSTEM's protocol: a fresh random 2,811/813/748 split on each seed, scored per word. It is used only for the comparison with TSTEM's published table.

The default training regime is 5 epochs at 2×10⁻⁵. The tuned regime is 10 epochs at 5×10⁻⁵, chosen for SecureBERT with oversampling on development macro F1 (seeds 42 and 123). BERT-base's own best setting is 10 epochs at 3×10⁻⁵. RoBERTa-base's own best setting is 10 epochs at 2×10⁻⁵. The test split was not used to pick these.

## Limitations

- Vulnerability F1 rests on about 10 test entities, so one miss moves recall by 0.1. A non-significant difference on that class is not evidence that two encoders are equivalent.
- P2 lets a model reuse entity strings it saw in training. The official-split numbers are the ones that speak to unseen names.
- SecureBERT and BERT-base differ in architecture, vocabulary, and pre-training data. The RoBERTa-base control separates part of that gap. It does not remove it.
- GPU training is not bit-wise deterministic. Reported figures are ten-seed means.
- The graph has no expert labels. Communities are entities that were reported together. Construction time excludes crawling, NER inference, Louvain, and export, so 3.8 ms per article is not end-to-end latency.
- Louvain uses resolution 1.0 and seed 42. Connected components on the same graph put 99.5% of nodes in one component (modularity 0.0004).
- The released graph is a co-occurrence property graph. It is not STIX 2.1 and it is not mapped to MITRE ATT&CK.

## References

- Balasubramanian et al., "A cognitive platform for collecting cyber threat intelligence and real-time detection using cloud computing," *Decision Analytics Journal*, vol. 14, 2025.
- E. Aghaei, X. Niu, W. Shadid, and E. Al-Shaer, "SecureBERT: A domain-specific language model for cybersecurity," arXiv:2204.02685, 2022.
- Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining approach," arXiv:1907.11692, 2019.
- M. T. Alam, D. Bhusal, Y. Park, and N. Rastogi, "CyNER: A Python library for cybersecurity named entity recognition," arXiv:2204.05754, 2022.
- T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," ICCV, 2017.
