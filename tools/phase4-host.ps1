# Phase 4 (issues #7, #8): run the agent grid on the host. Started by the scheduled task llm-phase4 (created with a trigger that can never fire
# by itself and started with `schtasks /Run`), so an SSH drop does not kill it.
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\arthu\llm\phase4\phase4-host.ps1 -Stages "O+T@1" -Tag full
# Log: C:\Users\arthu\llm\phase4\phase4-<Tag>.log (ends with an `exit=` line). Results: C:\Users\arthu\llm\phase4\results-<Tag>\<arm>.jsonl
# Host rules: waits until no benchmark holds the GPU (sweep-c.log ends with `exit=`) and no llama process runs; the grid unloads Ollama,
# starts `forge serve` itself and kills only the process tree it started.
param([string]$Stages = 'O+T@1', [string]$Tag = 'full', [string]$Tasks = '', [double]$BudgetMin = 0, [int]$LimitS = 900, [int]$WaitMinutes = 60)
$ErrorActionPreference = 'Continue'
$d = 'C:\Users\arthu\llm'
$p4 = "$d\phase4"
$log = "$p4\phase4-$Tag.log"
$env:FORGE_LLAMA_DIR = "$d\llama-b12207"
$env:FORGE_MODEL = 'C:\Users\arthu\.ollama\models\blobs\sha256-55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312'
$env:FORGE_HOME = "$d\forge\cache"
$env:PYTHONIOENCODING = 'utf-8'
function Say($m) { Add-Content -Path $log -Value "$(Get-Date -Format s) $m" -Encoding ascii }
function LastLine($f) { if (Test-Path $f) { (Get-Content $f -Tail 1) } else { '' } }
New-Item -ItemType Directory -Force $p4 | Out-Null
Say "wrapper start stages=$Stages tag=$Tag tasks=$Tasks budget=$BudgetMin limit=$LimitS"
$deadline = (Get-Date).AddMinutes($WaitMinutes)
while ($true) {
  $gpu = LastLine "$d\sweep-c.log"
  $busy = Get-Process llama-server, llama-bench -ErrorAction SilentlyContinue
  if ($gpu -like 'exit=*' -and -not $busy) { break }
  if ((Get-Date) -gt $deadline) { Say "ABORT: host still busy after $WaitMinutes min (gpu='$gpu' llama=$([bool]$busy))"; Add-Content $log "exit=5 $(Get-Date -Format s)" -Encoding ascii; exit 5 }
  Say "waiting: gpu='$gpu' llama=$([bool]$busy)"
  Start-Sleep 60
}
Say 'host is free'
Set-Location "$d\forge\repo"
$extra = @()
if ($Tasks -ne '') { $extra += @('--tasks', $Tasks) }
if ($BudgetMin -gt 0) { $extra += @('--budget-min', "$BudgetMin") }
& C:\Python312\python.exe -u 03-agent\harness\grid.py --stages $Stages --out-dir "$p4\results-$Tag" --work-dir "$p4\work" --repo "$d\forge\repo" `
  --python C:\Python312\python.exe --node 'C:\Program Files\nodejs\node.exe' --opencode 'C:\Users\arthu\.opencode\bin\opencode.exe' `
  --oc-cache-seed "$p4\ocseed" --time-limit $LimitS --stop-file "$p4\STOP" --log $log @extra | Out-Null
# grid.py wrote the exit= line itself
