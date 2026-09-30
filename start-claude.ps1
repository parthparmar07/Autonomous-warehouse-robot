# Do NOT commit real credentials here -- this file is tracked in git.
# Set your own token in the current shell before running this script, e.g.:
#   $env:ANTHROPIC_AUTH_TOKEN = "sk-..."
# or store it in a local, gitignored .env.ps1 and dot-source it instead.
$env:ANTHROPIC_BASE_URL = "https://agentrouter.org"
if (-not $env:ANTHROPIC_AUTH_TOKEN) {
    Write-Host "ANTHROPIC_AUTH_TOKEN is not set. Set it in your shell before running this script." -ForegroundColor Yellow
    exit 1
}
$env:ANTHROPIC_API_KEY = $env:ANTHROPIC_AUTH_TOKEN
$env:CLAUDE_CODE_USE_AUTH_TOKEN = "true"

Write-Host "Starting Claude Code via AgentRouter proxy..." -ForegroundColor Cyan
npx -y @anthropic-ai/claude-code
