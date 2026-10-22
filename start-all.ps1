<#
.SYNOPSIS
    一键启动 Tuling-Ai 全部服务（PostgreSQL/Redis 容器 + 4 个后端 + 1 个前端）。

.DESCRIPTION
    按依赖顺序启动，每一步都等到健康检查通过才继续：
      1. docker compose 拉起 PostgreSQL(:5432) 与 Redis(:6379)
      2. rag_api_service        :8011
      3. mcp_service            :8010
      4. langgraph_service      :2024
      5. main_service           :8002
      6. 前端 Vite              :5174
    每个服务都会打开一个独立 PowerShell 窗口，可直接看到日志。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\start-all.ps1
#>

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root

# 让所有 Python 子进程统一使用 UTF-8。Windows 控制台默认 GBK，模型输出里的
# emoji 会让调试 print() 抛 UnicodeEncodeError，并中断整轮 SSE。
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { 'pwsh' } else { 'powershell' }

function Wait-Port {
    param([int]$Port, [int]$TimeoutSec = 180)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-NetConnection 127.0.0.1 -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue) { return $true }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Wait-Url {
    param([string]$Url, [int]$TimeoutSec = 180)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest $Url -TimeoutSec 5 -UseBasicParsing
            if ($r.StatusCode -eq 200) { return $true }
        } catch { }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Start-ServiceWindow {
    param([string]$Title, [string]$Command, [string]$WorkDir = $root)
    Start-Process -FilePath $shell -ArgumentList @(
        '-NoExit', '-Command',
        "`$Host.UI.RawUI.WindowTitle='$Title'; Set-Location '$WorkDir'; $Command"
    ) | Out-Null
}

Write-Host '========================================' -ForegroundColor Cyan
Write-Host ' Tuling-Ai 一键启动' -ForegroundColor Cyan
Write-Host '========================================' -ForegroundColor Cyan

# ---------- 0. 依赖容器 ----------
Write-Host '[0/6] 启动 PostgreSQL / Redis 容器 ...' -ForegroundColor Yellow
docker compose up -d
if ($LASTEXITCODE -ne 0) { throw 'docker compose 启动失败，请先确认 Docker Desktop 已启动。' }
if (-not (Wait-Port 5432 60)) { throw 'PostgreSQL 5432 未就绪' }
if (-not (Wait-Port 6379 60)) { throw 'Redis 6379 未就绪' }
Write-Host '      容器就绪' -ForegroundColor Green

# ---------- 1. RAG ----------
Write-Host '[1/6] rag_api_service :8011（首次加载 bge-m3 需要几十秒）...' -ForegroundColor Yellow
Start-ServiceWindow 'Tuling-RAG :8011' 'uv run --project rag_api_service python rag_api_service/run.py'
if (-not (Wait-Url 'http://127.0.0.1:8011/api/docs/documents' 240)) { throw 'RAG 服务启动超时，请查看该窗口日志' }
Write-Host '      RAG 就绪' -ForegroundColor Green

# ---------- 2. MCP ----------
Write-Host '[2/6] mcp_service :8010 ...' -ForegroundColor Yellow
Start-ServiceWindow 'Tuling-MCP :8010' 'uv run --project mcp_service python mcp_service/run.py'
if (-not (Wait-Port 8010 90)) { throw 'MCP 服务启动超时' }
Write-Host '      MCP 就绪' -ForegroundColor Green

# ---------- 3. LangGraph ----------
Write-Host '[3/6] langgraph_service :2024（会连接 PostgreSQL 并建表）...' -ForegroundColor Yellow
Start-ServiceWindow 'Tuling-LangGraph :2024' 'uv run --project langgraph_service python langgraph_service/run.py'
if (-not (Wait-Url 'http://127.0.0.1:2024/health' 240)) { throw 'LangGraph 服务启动超时' }
Write-Host '      LangGraph 就绪' -ForegroundColor Green

# ---------- 4. main_service ----------
Write-Host '[4/6] main_service :8002 ...' -ForegroundColor Yellow
Start-ServiceWindow 'Tuling-Main :8002' 'uv run --project main_service python main_service/run.py'
if (-not (Wait-Url 'http://127.0.0.1:8002/api/health' 120)) { throw '主服务启动超时' }
Write-Host '      主服务就绪' -ForegroundColor Green

# ---------- 5. 前端 ----------
Write-Host '[5/6] 前端 Vite :5174 ...' -ForegroundColor Yellow
$fe = Join-Path $root 'chat-ai-ui-main\chat-ai-ui-main'
Start-ServiceWindow 'Tuling-前端 :5174' 'npm run dev -- --port 5174 --strictPort' $fe
if (-not (Wait-Port 5174 90)) { throw '前端启动超时' }
Write-Host '      前端就绪' -ForegroundColor Green

# ---------- 6. 完成 ----------
Write-Host '[6/6] 全部启动完成，正在打开浏览器 ...' -ForegroundColor Yellow
Start-Process 'http://localhost:5174'
Write-Host ''
Write-Host '前端      http://localhost:5174' -ForegroundColor Green
Write-Host '主服务    http://127.0.0.1:8002/docs' -ForegroundColor Green
Write-Host 'LangGraph http://127.0.0.1:2024/docs' -ForegroundColor Green
Write-Host '默认账号  root / admin123' -ForegroundColor Green

# docker / Start-Process 会向 stderr 写进度，PowerShell 可能把 $LASTEXITCODE 带成 1；
# 这里显式返回 0，避免调用方误判「启动失败」。
exit 0
