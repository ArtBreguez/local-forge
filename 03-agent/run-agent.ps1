$ErrorActionPreference = 'Continue'
$root = 'C:\Users\arthu\llm\agent-test'
$log = 'C:\Users\arthu\llm\agent-run.log'
$oll = 'C:\Users\arthu\AppData\Local\Programs\Ollama\ollama.exe'
$oc = Join-Path $env:APPDATA 'npm\opencode.cmd'
Set-Location $root
"start $(Get-Date -Format s)" | Out-File $log -Encoding ascii
'--- capabilities' | Add-Content $log
& $oll show dsv4-9b-heretic 2>$null | Select-String -Pattern 'capabilit|completion|tools|thinking|vision|context' | ForEach-Object { $_.Line } | Add-Content $log
'--- tests before' | Add-Content $log
(& node --test 2>&1 | Select-String -Pattern '^# (pass|fail)') -join ' ' | Add-Content $log
$prompt = 'The tests in test/duration.test.js fail. Implement parseDuration in src/duration.js (read the comment at the top of that file and the tests) so that `node --test` passes. Do not modify the tests. Run `node --test` yourself to check your work, and fix failures until every test passes. When done, reply with a one-line summary.'
$t0 = Get-Date
$p = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', 'C:\Users\arthu\llm\agent.cmd' -WorkingDirectory $root -PassThru -WindowStyle Hidden
$limit = 45 * 60
$ps = @()
while (-not $p.HasExited -and ((Get-Date) - $t0).TotalSeconds -lt $limit) {
  Start-Sleep -Seconds 30
  $line = (& $oll ps 2>$null | Select-Object -Skip 1 | Select-Object -First 1)
  if ($line) { $ps += ('{0:0}s {1}' -f ((Get-Date) - $t0).TotalSeconds, ($line -replace '\s{2,}', ' | ')) }
}
$timedOut = -not $p.HasExited
if ($timedOut) { & taskkill /PID $p.Id /T /F | Out-Null }
$secs = [int]((Get-Date) - $t0).TotalSeconds
"agent seconds=$secs timed_out=$timedOut" | Add-Content $log
'--- ollama ps samples (first, last)' | Add-Content $log
if ($ps.Count) { $ps[0] | Add-Content $log; $ps[-1] | Add-Content $log }
'--- tests after' | Add-Content $log
(& node --test 2>&1 | Select-String -Pattern '^# (tests|pass|fail)') -join ' ' | Add-Content $log
'--- test file unchanged?' | Add-Content $log
(Get-FileHash test\duration.test.js).Hash | Add-Content $log
'--- speed probe skipped (measured in run 1: 7.90 tok/s)' | Add-Content $log
& $oll run --verbose dsv4-9b-heretic 'Write a JavaScript function that reverses a string. Code only.' 2> C:\Users\arthu\llm\speed.txt | Out-Null
"exit=0" | Add-Content $log
