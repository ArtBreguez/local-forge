# Phase 6 (issue #23): best-of-N candidates on the host. Started by a scheduled task whose trigger cannot fire on its own
# (schtasks /Create /SC ONCE /SD 01/01/2030 /ST 00:00), run with `schtasks /Run`, and deleted when the job is done.
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\arthu\llm\phase6\phase6-host.ps1 -Arms S10 -Tag bon
# Log: C:\Users\arthu\llm\phase6\phase6-<Tag>.log (ends with an `exit=` line). Results: C:\Users\arthu\llm\phase6\results-<Tag>\
# Stop cleanly between two requests by creating C:\Users\arthu\llm\phase6\results-<Tag>\STOP (a rerun resumes the file).
# Host rules: waits until no benchmark holds the GPU (sweep-c.log `exit=`) and no llama process runs; the grid driver unloads Ollama,
# starts a fresh `forge serve` per arm and kills only its own process tree. Writes only below C:\Users\arthu\llm\phase6\.
param([string]$Arms = 'S10', [string]$Tag = 'bon', [int]$MaxProblems = 0, [string]$Problems = '', [int]$WaitMinutes = 120)
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
Say "wrapper start arms=$Arms tag=$Tag maxProblems=$MaxProblems"
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
if ($MaxProblems -gt 0) { $extra += @('--max-problems', "$MaxProblems") }
if ($Problems -ne '') { $extra += @('--problems', $Problems) }
& C:\Python312\python.exe -u 03-agent\quality\qgrid.py --arms $Arms --out "$p6\results-$Tag" --repo "$d\forge\repo" --python C:\Python312\python.exe `
  --node 'C:\Program Files\nodejs\node.exe' --log $log @extra | Out-Null
# qgrid.py wrote the exit= line itself
