# Phase 5 (issue #22): run the 30-minute agent arms on the host. Started by the scheduled task llm-phase5 (created with a trigger that can never
# fire by itself, started with `schtasks /Run`, deleted afterwards), so an SSH drop does not kill it.
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\arthu\llm\phase5\phase5-host.ps1 -Stages "O30@1;O30-ngram@1;O30-tool@1" -Tag full -BudgetMin 400
# Log: C:\Users\arthu\llm\phase5\phase5-<Tag>.log (ends with an `exit=` line). Results: C:\Users\arthu\llm\phase5\results-<Tag>\<arm>.jsonl
# Host rules: waits until no benchmark holds the GPU (sweep-c.log ends with `exit=`) and no llama process runs; the grid unloads Ollama,
# starts `forge serve` itself and kills only the process tree it started. -LimitS 0 (default) leaves every arm under its own limit
# (O30 arms: 1800 s); a number forces one limit on all arms (smoke runs). Create C:\Users\arthu\llm\phase5\STOP to stop launching new runs.
param([string]$Stages = 'O30@1;O30-ngram@1;O30-tool@1', [string]$Tag = 'full', [string]$Tasks = '', [double]$BudgetMin = 0, [int]$LimitS = 0, [int]$WaitMinutes = 60)
$ErrorActionPreference = 'Continue'
$d = 'C:\Users\arthu\llm'
$p5 = "$d\phase5"
$log = "$p5\phase5-$Tag.log"
$env:FORGE_LLAMA_DIR = "$d\llama-b12207"
$env:FORGE_MODEL = 'C:\Users\arthu\.ollama\models\blobs\sha256-55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312'
$env:FORGE_HOME = "$d\forge\cache"
$env:PYTHONIOENCODING = 'utf-8'
function Say($m) { Add-Content -Path $log -Value "$(Get-Date -Format s) $m" -Encoding ascii }
function LastLine($f) { if (Test-Path $f) { (Get-Content $f -Tail 1) } else { '' } }
New-Item -ItemType Directory -Force $p5 | Out-Null
Say "wrapper start stages=$Stages tag=$Tag tasks=$Tasks budget=$BudgetMin limit=$LimitS"
# control from docs/PROTOCOL.md: every llm-* task and its trigger, so that a task that can fire by itself is visible in the log
Get-ScheduledTask | Where-Object { $_.TaskName -like 'llm*' } | ForEach-Object { Say ("scheduled task " + $_.TaskName + " state=" + $_.State + " triggers=" + (($_.Triggers | ForEach-Object { $_.StartBoundary }) -join ',')) }
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
if ($LimitS -gt 0) { $extra += @('--time-limit', "$LimitS") }
& C:\Python312\python.exe -u 03-agent\harness\grid.py --stages $Stages --out-dir "$p5\results-$Tag" --work-dir "$p5\work" --repo "$d\forge\repo" `
  --python C:\Python312\python.exe --node 'C:\Program Files\nodejs\node.exe' --opencode 'C:\Users\arthu\.opencode\bin\opencode.exe' `
  --oc-cache-seed "$p5\ocseed" --stop-file "$p5\STOP" --log $log @extra | Out-Null
# grid.py wrote the exit= line itself
