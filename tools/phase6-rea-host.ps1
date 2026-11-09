# Phase 6 part C (issue #23): the REA tool-use benchmark on the host (arm R = fixed recipe, arm O = OpenCode with REA tools), model on config A.
# Started by a scheduled task whose trigger cannot fire on its own (schtasks /Create /SC ONCE /SD 01/01/2030 /ST 00:00), run with
# `schtasks /Run`, deleted when done:
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\arthu\llm\phase6\phase6-rea-host.ps1 -Arms R,O -Tag rea
# Log: C:\Users\arthu\llm\phase6\phase6-<Tag>.log (ends with an `exit=` line). Results: C:\Users\arthu\llm\phase6\results-<Tag>\ (STOP file = stop between tasks).
# Host rules as phase6-host.ps1: waits for a free GPU (sweep-c.log `exit=`, no llama process), kills only its own process tree, writes only below
# C:\Users\arthu\llm\phase6\ (REA itself only reads C:\Users\arthu\llm\rea\).
param([string]$Arms = 'R,O', [string]$Tag = 'rea', [string]$Tasks = '', [int]$Runs = 1, [int]$TimeLimit = 600, [int]$WaitMinutes = 120)
$ErrorActionPreference = 'Continue'
$d = 'C:\Users\arthu\llm'
$p6 = "$d\phase6"
$log = "$p6\phase6-$Tag.log"
$env:FORGE_LLAMA_DIR = "$d\llama-b12207"
$env:FORGE_MODEL = 'C:\Users\arthu\.ollama\models\blobs\sha256-55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312'
$env:FORGE_HOME = "$d\forge\cache"
$env:PYTHONIOENCODING = 'utf-8'
function Say($m) { Add-Content -Path $log -Value "$(Get-Date -Format s) $m" -Encoding ascii }
function LastLine($f) { if (Test-Path $f) { (Get-Content $f -Tail 1) } else { '' } }
New-Item -ItemType Directory -Force $p6 | Out-Null
Say "wrapper start arms=$Arms tag=$Tag tasks=$Tasks runs=$Runs"
$deadline = (Get-Date).AddMinutes($WaitMinutes)
while ($true) {
  $gpu = LastLine "$d\sweep-c.log"
  $busy = Get-Process llama-server, llama-bench -ErrorAction SilentlyContinue
  if ($gpu -like 'exit=*' -and -not $busy) { break }
  if ((Get-Date) -gt $deadline) { Say "ABORT: host still busy after $WaitMinutes min (gpu='$gpu' llama=$([bool]$busy))"; Add-Content $log "exit=5 $(Get-Date -Format s)" -Encoding ascii; exit 5 }
  Say "waiting: gpu='$gpu' llama=$([bool]$busy)"
  Start-Sleep 120
}
Say 'host is free'
Set-Location "$d\forge\repo"
$extra = @()
if ($Tasks -ne '') { $extra += @('--tasks', $Tasks) }
& C:\Python312\python.exe -u 03-agent\rea\rea_grid.py --arms $Arms --out "$p6\results-$Tag" --repo "$d\forge\repo" --python C:\Python312\python.exe `
  --runs $Runs --time-limit $TimeLimit --binary "$d\forge\repo\03-agent\rea\bin\bench.exe" --gguf $env:FORGE_MODEL --rea-root "$d\rea" `
  --work-dir "$p6\work-rea" --log $log @extra | Out-Null
# rea_grid.py wrote the exit= line itself
