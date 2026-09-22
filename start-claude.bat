@echo off
set ANTHROPIC_BASE_URL=https://agentrouter.org
set ANTHROPIC_AUTH_TOKEN=sk-7mjGe7TCPxaFXyf1FO3dNCfH1MkqB9bqUN5LxhODeR9vmPY7
set ANTHROPIC_API_KEY=sk-7mjGe7TCPxaFXyf1FO3dNCfH1MkqB9bqUN5LxhODeR9vmPY7
set CLAUDE_CODE_USE_AUTH_TOKEN=true

echo Starting Claude Code via AgentRouter proxy...
npx -y @anthropic-ai/claude-code
