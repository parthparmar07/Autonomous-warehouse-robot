@echo off
REM Do NOT commit real credentials here -- this file is tracked in git.
REM Set ANTHROPIC_AUTH_TOKEN in your environment before running this script.
set ANTHROPIC_BASE_URL=https://agentrouter.org
if "%ANTHROPIC_AUTH_TOKEN%"=="" (
    echo ANTHROPIC_AUTH_TOKEN is not set. Set it in your environment before running this script.
    exit /b 1
)
set ANTHROPIC_API_KEY=%ANTHROPIC_AUTH_TOKEN%
set CLAUDE_CODE_USE_AUTH_TOKEN=true

echo Starting Claude Code via AgentRouter proxy...
npx -y @anthropic-ai/claude-code
