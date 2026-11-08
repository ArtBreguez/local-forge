# Post-fix host check of `forge serve` (branch feat/hw-adapt, fix of the server plan). Writes C:\Users\arthu\llm\forge\serve-fixed.log. Only writes under C:\Users\arthu\llm\forge\.
# A  serve --once, plan path (real cache dir, which holds only the inert entry of the buggy first tune)
# B  serve --once with the REAL result of the first tune (15 layers by llama-bench) replayed into a scratch cache: must be re-checked and clamped
# C  long-running serve, plan path, + one /v1/chat/completions at temperature 0
# D  long-running serve with a SYNTHETIC 16 layers / ubatch 256 entry (out-of-sample prediction check), + the same completion
$ErrorActionPreference = 'Continue'
$d = 'C:\Users\arthu\llm'; $f = "$d\forge"; $log = "$f\serve-fixed.log"; $repo = "$f\repo"
$py = 'C:\Python312\python.exe'
$env:FORGE_LLAMA_DIR = "$d\llama-b12207"
$env:FORGE_MODEL = 'C:\Users\arthu\.ollama\models\blobs\sha256-55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:FORGE_HOME = "$f\cache"      # forge's default cache dir is under %LOCALAPPDATA%, outside the folder this job may write to
function Log($s) { $s | Add-Content -Path $log -Encoding utf8 }
"start $(Get-Date -Format s)" | Out-File $log -Encoding utf8
$last = (Get-Content "$d\sweep-c.log" -Tail 1)
Log "preflight: sweep-c.log last line: $last"
$procs = @(Get-Process | Where-Object { $_.ProcessName -match '^llama-' })
Log "preflight: llama processes running: $($procs.Count)"
if (($last -notmatch '^exit=') -or $procs.Count -gt 0) { Log 'ABORT: the GPU is not free'; Log "exit=1 $(Get-Date -Format s)"; exit 1 }
try {
  $ps = (Invoke-RestMethod http://127.0.0.1:11434/api/ps).models
  Log "preflight: ollama models loaded: $(@($ps).Count)"
  $ps | ForEach-Object { Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -Body (@{ model = $_.name; keep_alive = 0 } | ConvertTo-Json) | Out-Null; Log "unloaded $($_.name)" }
} catch { Log "ollama check: $_" }
Start-Sleep 2
$dev = & "$d\llama-b12207\llama-bench.exe" --list-devices 2>&1 | Where-Object { "$_" -match 'Vulkan0:' }
Log "preflight: $dev"
Set-Location $repo
Log "git-tree files: $((Get-ChildItem $repo\forge -Filter *.py).Count) py in forge, tests: $((Get-ChildItem $repo\tests -Filter test_*.py).Count)"

function Stop-All($pid_) {
  if ($pid_) { & taskkill /PID $pid_ /T /F 2>&1 | Out-Null }
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
  Start-Sleep 3
}
function Completion($port, $tag) {
  $body = @{ model = 'local'; temperature = 0; max_tokens = 160; messages = @(@{ role = 'user'; content = 'Write a JavaScript function isLeapYear(y) with no comments. Code only.' }) } | ConvertTo-Json -Depth 5
  $t0 = Get-Date
  try {
    $r = Invoke-RestMethod "http://127.0.0.1:$port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 300
    Log "[$tag] completion wall=$([int]((Get-Date)-$t0).TotalSeconds)s completion_tokens=$($r.usage.completion_tokens) prompt_tokens=$($r.usage.prompt_tokens) finish=$($r.choices[0].finish_reason)"
    Log "[$tag] timings predicted_per_second=$([math]::Round($r.timings.predicted_per_second,2)) prompt_per_second=$([math]::Round($r.timings.prompt_per_second,2)) predicted_n=$($r.timings.predicted_n) prompt_n=$($r.timings.prompt_n)"
    Log "[$tag] --- content"; Log $r.choices[0].message.content; Log "[$tag] --- end content"
  } catch { Log "[$tag] completion FAILED: $_" }
}
function LongServe($tag, $extra) {
  $out = "$f\serve-$tag.out.txt"; $err = "$f\serve-$tag.err.txt"
  $args_ = @('-m', 'forge', 'serve', '--port', '8088') + $extra
  $p = Start-Process $py -ArgumentList $args_ -WorkingDirectory $repo -PassThru -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err
  $ok = $false; $t0 = Get-Date
  for ($i = 0; $i -lt 100; $i++) {
    try { $h = Invoke-RestMethod http://127.0.0.1:8088/health -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; break } } catch {}
    Start-Sleep 2
  }
  Log "[$tag] healthy=$ok after $([int]((Get-Date)-$t0).TotalSeconds)s"
  if ($ok) { Completion 8088 $tag }
  Log "[$tag] server cmdline:"
  Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'llama-server.exe' } | ForEach-Object { Log $_.CommandLine }
  Stop-All $p.Id
  Log "[$tag] serve stdout:"; Get-Content $out -ErrorAction SilentlyContinue | ForEach-Object { Log "  $_" }
  Log "[$tag] serve stderr:"; Get-Content $err -ErrorAction SilentlyContinue | ForEach-Object { Log "  $_" }
  Log "[$tag] llama-server processes left: $(@(Get-Process llama-server -ErrorAction SilentlyContinue).Count)"
}

Log '=== A: forge serve --once (plan path, real cache dir)'
& $py -m forge serve --once --port 8088 2>&1 | ForEach-Object { Log "$_" }
Log "A rc=$LASTEXITCODE"; Stop-All $null

Log '=== B: forge serve --once with the REAL first-tune result (15 layers, llama-bench) replayed'
& $py "$f\replay_tune.py" "$f\cache-replay" 2>&1 | ForEach-Object { Log "$_" }
& $py -m forge serve --once --port 8088 --cache-dir "$f\cache-replay" 2>&1 | ForEach-Object { Log "$_" }
Log "B rc=$LASTEXITCODE"; Stop-All $null

Log '=== C: long-running serve, plan path, one completion'
LongServe 'C' @()

Log '=== D: long-running serve, SYNTHETIC 16 layers / ubatch 256 entry, one completion'
& $py "$f\replay_tune.py" "$f\cache-replay-d" 16 256 2>&1 | ForEach-Object { Log "$_" }
LongServe 'D' @('--cache-dir', "$f\cache-replay-d")

Log "final: llama processes left: $(@(Get-Process | Where-Object { $_.ProcessName -match '^llama-' }).Count)"
Log "exit=0 $(Get-Date -Format s)"
