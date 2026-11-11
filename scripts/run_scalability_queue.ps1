# Sequential GPU queue: N=10/N=20 IQL+QMIX training under CodeCarbon, then the sealed scalability benchmark.
# Sequential on purpose: each run holds 1.5-2.5 GB of replay in RAM.
param([string]$WaitFor = '')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($WaitFor) { Wait-Process -Id ($WaitFor -split ',' | ForEach-Object { [int]$_ }) -ErrorAction SilentlyContinue }
$env:PYTHONUNBUFFERED = '1'
$py = '.venv\Scripts\python.exe'
$common = '--workload mixed --stage C5 --steps 6000 --horizon 600 --scenario-mix --bc-epochs 8 --dagger-rounds 1 --bc-aux-weight 0.25 --eval-every 3000 --eval-episodes 5 --train-every 8 --epsilon-steps 30000 --epsilon-start 0.3 --device cuda --seed 11'
# The flags above reproduce the published N=5 mixed_stock_v1 recipe; configs/runtime_training.yaml defaults differ.
$runs = @(
    @{ alg = 'qmix'; n = 10; bc = 30 },
    @{ alg = 'iql'; n = 10; bc = 30 },
    @{ alg = 'qmix'; n = 20; bc = 20 },
    @{ alg = 'iql'; n = 20; bc = 20 }
)
foreach ($r in $runs) {
    $artifact = "artifacts\warehouse_v6_research\mixed_stock_v1\$($r.alg)_policy_N$($r.n).pt"
    if (Test-Path $artifact) { "skip $artifact (exists)"; continue }
    $label = "train_$($r.alg)_N$($r.n)"
    "$(Get-Date -Format s) start $label"
    $cmd = "-m scripts.track_emissions $label src.training.runner --algorithm $($r.alg) --num-agents $($r.n) --bc-episodes $($r.bc) $common"
    $p = Start-Process -FilePath $py -ArgumentList $cmd -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "outputs\logs\$label.log" -RedirectStandardError "outputs\logs\$label.err.log"
    "$(Get-Date -Format s) end $label exit=$($p.ExitCode)"
}
"$(Get-Date -Format s) start evaluate_scalability"
$p = Start-Process -FilePath $py -ArgumentList '-m scripts.track_emissions evaluate_scalability scripts.evaluate_scalability' -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput 'outputs\logs\evaluate_scalability.log' -RedirectStandardError 'outputs\logs\evaluate_scalability.err.log'
"$(Get-Date -Format s) end evaluate_scalability exit=$($p.ExitCode)"
