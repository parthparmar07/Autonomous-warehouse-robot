# GPU training lane under CodeCarbon. Runs inside one lane are sequential; start one lane per algorithm to share the GPU.
#   powershell -File scripts\run_scalability_queue.ps1 -Runs "qmix:10,qmix:20"
#   powershell -File scripts\run_scalability_queue.ps1 -Runs "iql:10,iql:20"
#   powershell -File scripts\run_scalability_queue.ps1 -Runs none -WaitFor "<lane pids>" -Evaluate
# Each run holds 1.5-3 GB of RAM/commit, so the lane count is bounded by memory, not by the GPU.
param([string]$Runs = 'qmix:10,iql:10,qmix:20,iql:20', [string]$WaitFor = '', [switch]$Evaluate)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($WaitFor) { Wait-Process -Id ($WaitFor -split ',' | ForEach-Object { [int]$_ }) -ErrorAction SilentlyContinue }
$env:PYTHONUNBUFFERED = '1'
$py = '.venv\Scripts\python.exe'
# These flags reproduce the published N=5 mixed_stock_v1 recipe; configs/runtime_training.yaml defaults differ.
$common = '--workload mixed --stage C5 --steps 6000 --horizon 600 --scenario-mix --bc-epochs 8 --dagger-rounds 1 --bc-aux-weight 0.25 --eval-every 3000 --eval-episodes 5 --train-every 8 --epsilon-steps 30000 --epsilon-start 0.3 --device cuda --seed 11'
foreach ($item in ($Runs -split ',' | Where-Object { $_ -match ':' })) {
    $alg, $n = $item -split ':'
    $bc = if ([int]$n -ge 20) { 20 } else { 30 }
    $artifact = "artifacts\warehouse_v6_research\mixed_stock_v1\$($alg)_policy_N$n.pt"
    if (Test-Path $artifact) { "skip $artifact (exists)"; continue }
    $label = "train_$($alg)_N$n"
    "$(Get-Date -Format s) start $label"
    $cmd = "-m scripts.track_emissions $label src.training.runner --algorithm $alg --num-agents $n --bc-episodes $bc $common"
    $p = Start-Process -FilePath $py -ArgumentList $cmd -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "outputs\logs\$label.log" -RedirectStandardError "outputs\logs\$label.err.log"
    "$(Get-Date -Format s) end $label exit=$($p.ExitCode)"
}
if ($Evaluate) {
    "$(Get-Date -Format s) start evaluate_scalability"
    $p = Start-Process -FilePath $py -ArgumentList '-m scripts.track_emissions evaluate_scalability scripts.evaluate_scalability' -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput 'outputs\logs\evaluate_scalability.log' -RedirectStandardError 'outputs\logs\evaluate_scalability.err.log'
    "$(Get-Date -Format s) end evaluate_scalability exit=$($p.ExitCode)"
}
