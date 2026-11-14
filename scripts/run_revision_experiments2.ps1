# Second queue (run after run_revision_experiments.ps1): unmitigated BERT-base in the tuned regime, ten seeds.
$py = "C:\Users\rico2\.venvs\cti\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$s = 42,123,456,789,1011,1213,1415,1617,1819,2021
& $py scripts/run_ablation.py --model baseline --arms ce --seeds $s --epochs 10 --lr 5e-5 --out rev_bert_ce_tuned.csv --drop-checkpoints --resume
Write-Output "QUEUE2_DONE"
