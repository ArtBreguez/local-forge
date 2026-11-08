# First real `forge tune` with the merged code (ubatch axis, server bound), then `forge serve --once` with the tuned
# result and one completion. Log: C:\Users\arthu\llm\forge\tune2.log
$ErrorActionPreference = 'Continue'
$d = 'C:\Users\arthu\llm'
$log = "$d\forge\tune2.log"
$env:FORGE_LLAMA_DIR = "$d\llama-b12207"
$env:FORGE_MODEL = 'C:\Users\arthu\.ollama\models\blobs\sha256-55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312'
$env:FORGE_HOME = "$d\forge\cache"
"start $(Get-Date -Format s)" | Out-File $log -Encoding ascii
if (Get-Process llama-server, llama-bench -ErrorAction SilentlyContinue) { 'ABORT: llama process running'; 'exit=1' | Add-Content $log; exit 1 }
try {
  (Invoke-RestMethod http://127.0.0.1:11434/api/ps).models | ForEach-Object {
    Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -Body (@{ model = $_.name; keep_alive = 0 } | ConvertTo-Json) | Out-Null
  }
} catch {}
Start-Sleep 5
Set-Location "$d\forge\repo"
'--- tune' | Add-Content $log
$t0 = Get-Date
& C:\Python312\python.exe -m forge tune --budget 2400 --reps 2 *>> $log
"tune rc=$LASTEXITCODE secs=$([int]((Get-Date)-$t0).TotalSeconds)" | Add-Content $log
'--- serve --once' | Add-Content $log
& C:\Python312\python.exe -m forge serve --once --port 8088 *>> $log
"serve rc=$LASTEXITCODE" | Add-Content $log
'--- long serve + completion' | Add-Content $log
$p = Start-Process C:\Python312\python.exe -ArgumentList '-m', 'forge', 'serve', '--port', '8088' -WorkingDirectory "$d\forge\repo" -PassThru -WindowStyle Hidden -RedirectStandardOutput "$d\forge\tune2-serve.out" -RedirectStandardError "$d\forge\tune2-serve.err"
$ok = $false
for ($i = 0; $i -lt 120; $i++) { try { if ((Invoke-RestMethod http://127.0.0.1:8088/health -TimeoutSec 3).status -eq 'ok') { $ok = $true; break } } catch {}; Start-Sleep 2 }
"healthy=$ok" | Add-Content $log
if ($ok) {
  foreach ($k in 1..3) {
    $body = @{ model = 'local'; temperature = 0; seed = 1; max_tokens = 200; messages = @(@{ role = 'user'; content = 'Write a Python function that merges two sorted lists into one sorted list. Code only.' }) } | ConvertTo-Json -Depth 5
    $r = Invoke-RestMethod http://127.0.0.1:8088/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 300
    "completion $k tokens=$($r.timings.predicted_n) gen_tps=$([math]::Round($r.timings.predicted_per_second,2)) prompt_tps=$([math]::Round($r.timings.prompt_per_second,1))" | Add-Content $log
  }
}
Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'llama-server.exe' } | ForEach-Object { 'cmd: ' + $_.CommandLine } | Add-Content $log
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
"exit=0 $(Get-Date -Format s)" | Add-Content $log
