# Extra experiments requested by the reviewers (all new CSVs; no existing result file is modified).
# Usage:  powershell -File scripts/run_revision_experiments.ps1
$py = "C:\Users\rico2\.venvs\cti\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$s = 42,123,456,789,1011,1213,1415,1617,1819,2021
$common = @("--drop-checkpoints", "--resume")

# A. Unmitigated SecureBERT (CE) in the tuned regime, ten seeds (+ label-distribution diagnostics)
& $py scripts/run_ablation.py --model securebert --arms ce --seeds $s --epochs 10 --lr 5e-5 --out rev_securebert_ce_tuned.csv @common

# A2. Unmitigated SecureBERT (CE) in the default regime, three ablation seeds, with diagnostics
& $py scripts/run_ablation.py --model securebert --arms ce --seeds 42 123 456 --out rev_securebert_ce_default_diag.csv @common

# B. RoBERTa-base control with oversampling (k=3), tuned regime, ten seeds
& $py scripts/run_ablation.py --model roberta --arms oversample --seeds $s --epochs 10 --lr 5e-5 --out rev_roberta_os_tuned.csv @common

# C. Equal tuning budget: same four candidates, seeds 42/123, dev macro F1, for BERT-base and RoBERTa-base
foreach ($m in @("baseline", "roberta")) {
    $out = "rev_tuning_$m.csv"
    & $py scripts/run_ablation.py --model $m --arms oversample --seeds 42 123 --epochs 5  --lr 2e-5 --out $out @common
    & $py scripts/run_ablation.py --model $m --arms oversample --seeds 42 123 --epochs 10 --lr 2e-5 --out $out @common
    & $py scripts/run_ablation.py --model $m --arms oversample --seeds 42 123 --epochs 10 --lr 3e-5 --out $out @common
    & $py scripts/run_ablation.py --model $m --arms oversample --seeds 42 123 --epochs 10 --lr 5e-5 --out $out @common
}
Write-Output "ALL_REVISION_RUNS_DONE"
