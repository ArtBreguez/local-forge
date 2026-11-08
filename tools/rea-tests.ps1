# REA smoke tests on real local files (offline). Log: C:\Users\arthu\llm\rea\rea-tests.log
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\arthu\llm\rea'
$env:PATH = "$root\node-v24.22.2-win-x64;$env:PATH"
$rea = "$root\app\node_modules\.bin\rea.cmd"
$log = "$root\rea-tests.log"
$out = "$root\out"; New-Item -ItemType Directory -Force $out | Out-Null
"start $(Get-Date -Format s)" | Out-File $log -Encoding ascii
'--- commands matching our targets' | Add-Content $log
(& $rea --help 2>&1) | Where-Object { $_ -match 'gguf|model|hardening|pe-imports|pe imports|inspect-pe|javascript|strings|doctor|go-|rust' } | ForEach-Object { $_.Trim().Substring(0, [Math]::Min(150, $_.Trim().Length)) } | Add-Content $log
function Run($name, $cmdArgs) {
  $t0 = Get-Date
  $o = & $rea @cmdArgs 2>&1 | Out-String
  $rc = $LASTEXITCODE
  $o | Out-File "$out\$name.txt" -Encoding utf8
  "=== $name rc=$rc secs=$([int]((Get-Date)-$t0).TotalSeconds) bytes=$($o.Length) :: rea $($cmdArgs -join ' ')" | Add-Content $log
  ($o -split "`n" | Select-Object -First 25) -join "`n" | ForEach-Object { $_.Substring(0, [Math]::Min(3000, $_.Length)) } | Add-Content $log
}
$gguf = 'C:\Users\arthu\.ollama\models\blobs\sha256-55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312'
$exe = 'C:\Users\arthu\llm\llama-b12207\llama-server.exe'
Run 'doctor' @('doctor', '--json')
foreach ($c in 'inspect-model-artifact', 'inspect-model', 'inspect-gguf') {
  if ((& $rea --help 2>&1 | Out-String) -match "\b$c\b") { Run 'gguf' @($c, $gguf, '--json'); break }
}
foreach ($c in 'inspect-binary-hardening', 'binary-hardening', 'inspect-hardening') {
  if ((& $rea --help 2>&1 | Out-String) -match "\b$c\b") { Run 'hardening' @($c, $exe, '--json'); break }
}
foreach ($c in 'inspect-pe-imports', 'pe-imports') {
  if ((& $rea --help 2>&1 | Out-String) -match "\b$c\b") { Run 'peimports' @($c, $exe, '--json'); break }
}
"exit=0 $(Get-Date -Format s)" | Add-Content $log
