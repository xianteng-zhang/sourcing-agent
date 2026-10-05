# 一键重启本地开发服务。
#
# 为什么需要它：Streamlit 只会**重新执行 `app.py` 自己**，不会重新 import
# 已经加载过的模块。所以只要你改过后端模块（agent / storage / actions /
# jobs / events / compliance / review_miner / prompts ...），光刷新浏览器是不够的，
# 会看到这种东西：
#
#     ImportError: cannot import name 'write_outbox' from 'agent'
#     TypeError: save_analysis() got an unexpected keyword argument 'run_id'
#
# 那些不是代码错，是服务里还跑着改动之前的模块。跑一下这个脚本就好。
#
# 只改 `app.py` 本身的话，直接刷新浏览器即可，不用重启。
#
# 用法：
#     .\dev.ps1              # 默认 8501
#     .\dev.ps1 -Port 8502

param([int]$Port = 8501)

$root = $PSScriptRoot

# 1) 停掉旧的：既按端口找，也按命令行兜底。
#    只按端口找会漏掉 venv 那层启动器进程（它不占端口，但会让端口一直被占）。
$stale = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique

Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'streamlit' -and $_.CommandLine -match 'app\.py' } |
    ForEach-Object { $stale += $_.ProcessId }

$stale = $stale | Sort-Object -Unique
foreach ($procId in $stale) {
    Write-Host "  停掉 PID $procId"
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
}
if ($stale) { Start-Sleep -Seconds 2 }

# 2) 起新的
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "找不到 $python —— 虚拟环境不在？"
    exit 1
}

Write-Host ""
Write-Host "  启动 → http://localhost:$Port"
Write-Host "  Ctrl+C 停止"
Write-Host ""

& $python -m streamlit run (Join-Path $root "app.py") --server.port $Port
