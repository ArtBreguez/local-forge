# Smoke test of `forge serve --once` on the host with the branch code (feat/hw-adapt), then a real completion through the
# OpenAI-compatible endpoint with a long-running server, then stop. Output: C:\Users\arthu\llm\serve-smoke.log
$ErrorActionPreference = 'Continue'
$d = 'C:\Users\arthu\llm'
$log = "$d\serve-smoke.log"
$repo = "$d\forge\repo"
$env:FORGE_LLAMA_DIR = "$d\llama-b12207"
$env:FORGE_MODEL = 'C:\Users\arthu\.ollama\models\blobs\sha256-55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312'
$env:FORGE_HOME = "$d\forge\cache"
"start $(Get-Date -Format s)" | Out-File $log -Encoding ascii
try {
  (Invoke-RestMethod http://127.0.0.1:11434/api/ps).models | ForEach-Object {
    Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -Body (@{ model = $_.name; keep_alive = 0 } | ConvertTo-Json) | Out-Null
  }
} catch {}
Set-Location $repo
'--- serve --once' | Add-Content $log
& C:\Python312\python.exe -m forge serve --once --port 8088 2>&1 | Add-Content $log
"once rc=$LASTEXITCODE" | Add-Content $log

'--- long-running serve + one completion' | Add-Content $log
$p = Start-Process C:\Python312\python.exe -ArgumentList '-m', 'forge', 'serve', '--port', '8088' -WorkingDirectory $repo -PassThru -WindowStyle Hidden -RedirectStandardOutput "$d\serve-out.txt" -RedirectStandardError "$d\serve-err.txt"
$ok = $false
for ($i = 0; $i -lt 90; $i++) {
  try { $h = Invoke-RestMethod http://127.0.0.1:8088/health -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; break } } catch {}
  Start-Sleep 2
}
"healthy=$ok after $($i*2)s" | Add-Content $log
if ($ok) {
  $body = @{ model = 'local'; temperature = 0; max_tokens = 160; messages = @(@{ role = 'user'; content = 'Write a JavaScript function isLeapYear(y) with no comments. Code only.' }) } | ConvertTo-Json -Depth 5
  $t0 = Get-Date
  $r = Invoke-RestMethod http://127.0.0.1:8088/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 300
  "completion secs=$([int]((Get-Date)-$t0).TotalSeconds) tokens=$($r.usage.completion_tokens) timings=$($r.timings | ConvertTo-Json -Compress)" | Add-Content $log
  '--- content' | Add-Content $log
  $r.choices[0].message.content | Add-Content $log
  '--- props' | Add-Content $log
  try { (Invoke-RestMethod http://127.0.0.1:8088/props).default_generation_settings | ConvertTo-Json -Compress | ForEach-Object { $_.Substring(0, [Math]::Min(400, $_.Length)) } | Add-Content $log } catch { "props: $_" | Add-Content $log }
}
'--- server cmdline' | Add-Content $log
Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'llama-server.exe' } | ForEach-Object { $_.CommandLine } | Add-Content $log
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
'--- serve stdout tail' | Add-Content $log
Get-Content "$d\serve-out.txt" -Tail 15 -ErrorAction SilentlyContinue | Add-Content $log
Get-Content "$d\serve-err.txt" -Tail 8 -ErrorAction SilentlyContinue | Add-Content $log
"exit=0 $(Get-Date -Format s)" | Add-Content $log
