# Phase 3 (issue #6): run the answer-quality arms on the host. Started by the scheduled task llm-phase3, so an SSH drop does not kill it.
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\arthu\llm\phase3\phase3-host.ps1 -Arms A,B,C,D,E -Tag full
# Log: C:\Users\arthu\llm\phase3\phase3-<Tag>.log (ends with an `exit=` line). Results: C:\Users\arthu\llm\phase3\results-<Tag>\
# Host rules: waits until the Ghidra install job has written its `exit=` line and no benchmark holds the GPU (sweep-c.log `exit=`),
# refuses to start over a running llama process, unloads Ollama; the grid driver restarts the server per arm and kills it afterwards.
param([string]$Arms = 'A,B,C,D,E', [string]$Tag = 'full', [int]$MaxProblems = 0, [string]$Problems = '', [int]$WaitMinutes = 180)
$ErrorActionPreference = 'Continue'
$d = 'C:\Users\arthu\llm'
$p3 = "$d\phase3"
$log = "$p3\phase3-$Tag.log"
$env:FORGE_LLAMA_DIR = "$d\llama-b12207"
$env:FORGE_MODEL = 'C:\Users\arthu\.ollama\models\blobs\sha256-55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312'
$env:FORGE_HOME = "$d\forge\cache"
$env:PYTHONIOENCODING = 'utf-8'
function Say($m) { Add-Content -Path $log -Value "$(Get-Date -Format s) $m" -Encoding ascii }
function LastLine($f) { if (Test-Path $f) { (Get-Content $f -Tail 1) } else { '' } }
New-Item -ItemType Directory -Force $p3 | Out-Null
Say "wrapper start arms=$Arms tag=$Tag maxProblems=$MaxProblems"
$deadline = (Get-Date).AddMinutes($WaitMinutes)
while ($true) {
  $ghidra = LastLine "$d\rea\ghidra.log"
  $gpu = LastLine "$d\sweep-c.log"
  $busy = Get-Process llama-server, llama-bench -ErrorAction SilentlyContinue
  if ($ghidra -like 'exit=*' -and $gpu -like 'exit=*' -and -not $busy) { break }
  if ((Get-Date) -gt $deadline) { Say "ABORT: host still busy after $WaitMinutes min (ghidra='$ghidra' gpu='$gpu' llama=$([bool]$busy))"; Add-Content $log "exit=5 $(Get-Date -Format s)" -Encoding ascii; exit 5 }
  Say "waiting: ghidra='$ghidra' gpu='$gpu' llama=$([bool]$busy)"
  Start-Sleep 120
}
Say 'host is free'
Set-Location "$d\forge\repo"
$extra = @()
if ($MaxProblems -gt 0) { $extra += @('--max-problems', "$MaxProblems") }
if ($Problems -ne '') { $extra += @('--problems', $Problems) }
& C:\Python312\python.exe -u 03-agent\quality\qgrid.py --arms $Arms --out "$p3\results-$Tag" --repo "$d\forge\repo" --python C:\Python312\python.exe `
  --node 'C:\Program Files\nodejs\node.exe' --log $log @extra | Out-Null
# qgrid.py wrote the exit= line itself
