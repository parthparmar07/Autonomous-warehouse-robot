$env:ANTHROPIC_BASE_URL = "https://agentrouter.org"
$env:ANTHROPIC_AUTH_TOKEN = "sk-7mjGe7TCPxaFXyf1FO3dNCfH1MkqB9bqUN5LxhODeR9vmPY7"
$env:ANTHROPIC_API_KEY = "sk-7mjGe7TCPxaFXyf1FO3dNCfH1MkqB9bqUN5LxhODeR9vmPY7"
$env:CLAUDE_CODE_USE_AUTH_TOKEN = "true"

Write-Host "Starting Claude Code via AgentRouter proxy..." -ForegroundColor Cyan
npx -y @anthropic-ai/claude-code
