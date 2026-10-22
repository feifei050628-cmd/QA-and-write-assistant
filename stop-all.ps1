<#
.SYNOPSIS
    停止 Tuling-Ai 的全部服务。

.DESCRIPTION
    只按端口回收本项目的进程，不会影响 atlas-assistant 等占用 5173/8000 的其他项目。
    默认保留 PostgreSQL/Redis 容器；加 -WithDocker 一并停止容器。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\stop-all.ps1
    powershell -ExecutionPolicy Bypass -File .\stop-all.ps1 -WithDocker
#>

param(
    [switch]$WithDocker
)

$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
Set-Location $root

$ports = @(8011, 8010, 2024, 8002, 5174)

foreach ($port in $ports) {
    $conns = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    if (-not $conns) { Write-Host "端口 $port 无监听，跳过" -ForegroundColor DarkGray; continue }
    foreach ($c in $conns) {
        try {
            Stop-Process -Id $c.OwningProcess -Force -ErrorAction Stop
            Write-Host "已停止端口 $port (PID $($c.OwningProcess))" -ForegroundColor Green
        } catch {
            Write-Host "停止端口 $port 失败: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

# uvicorn reload 模式可能留下父进程，按命令行兜底清理本项目进程
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*Tuling-Ai-main*' } |
    ForEach-Object {
        try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop; Write-Host "已清理残留进程 $($_.ProcessId)" -ForegroundColor Green } catch { }
    }

if ($WithDocker) {
    Write-Host '停止 PostgreSQL / Redis 容器 ...' -ForegroundColor Yellow
    docker compose stop
}

Write-Host '完成。' -ForegroundColor Cyan
