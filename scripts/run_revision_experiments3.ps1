# Third queue: for BERT-base and RoBERTa-base, run oversampling on ten seeds in the regime that wins
# their own 4-candidate search (mean development macro F1, seeds 42/123), unless it is the regime
# already run (10 epochs, 5e-5). Then unmitigated BERT-base in the tuned regime (reference).
$py = "C:\Users\rico2\.venvs\cti\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$s = 42,123,456,789,1011,1213,1415,1617,1819,2021
foreach ($m in @("baseline", "roberta")) {
    $best = & $py -c "
import pandas as pd,re
d=pd.read_csv('results/rev_tuning_$m.csv')
g=d.groupby(['epochs','lr']).val_macro_f1.mean().sort_values(ascending=False)
e,l=g.index[0]; print(int(e), format(float(l),'g'))
"
    $parts = $best.Trim().Split(" ")
    Write-Output "BEST $m : epochs=$($parts[0]) lr=$($parts[1])"
    if (-not ($parts[0] -eq "10" -and [double]$parts[1] -eq 5e-5)) {
        & $py scripts/run_ablation.py --model $m --arms oversample --seeds $s --epochs $parts[0] --lr $parts[1] --out "rev_${m}_os_best.csv" --drop-checkpoints --resume
    }
}
& $py scripts/run_ablation.py --model baseline --arms ce --seeds $s --epochs 10 --lr 5e-5 --out rev_bert_ce_tuned.csv --drop-checkpoints --resume
Write-Output "QUEUE3_DONE"
