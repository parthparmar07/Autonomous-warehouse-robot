# One-command live demo: FastAPI decision service + dashboard, opened in the browser.
#   powershell -ExecutionPolicy Bypass -File scripts\demo.ps1        (Ctrl+C to stop both)
param([int]$ApiPort = 8000, [int]$WebPort = 5500)
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = if (Test-Path '.venv\Scripts\python.exe') { '.venv\Scripts\python.exe' } else { 'python' }
$env:WAREHOUSE_CORS_ORIGINS = "http://127.0.0.1:$WebPort,http://localhost:$WebPort"
$api = Start-Process $py -ArgumentList "-m uvicorn src.deployment.api:app --host 127.0.0.1 --port $ApiPort --workers 1" -NoNewWindow -PassThru
$web = Start-Process $py -ArgumentList "-m http.server $WebPort --bind 127.0.0.1 --directory frontend" -NoNewWindow -PassThru
try {
    Write-Host 'Waiting for the API (loading PyTorch)...'
    for ($i = 0; $i -lt 60; $i++) {
        try { Invoke-RestMethod "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep 1 }
    }
    Start-Process "http://127.0.0.1:$WebPort"
    Write-Host "Dashboard: http://127.0.0.1:$WebPort   API docs: http://127.0.0.1:$ApiPort/docs   (Ctrl+C to stop)"
    Wait-Process -Id $api.Id
} finally {
    Stop-Process -Id $api.Id, $web.Id -ErrorAction SilentlyContinue
}
