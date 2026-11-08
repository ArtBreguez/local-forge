# Build a known-source binary on the host and have REA + Ghidra decompile one function. Log: C:\Users\arthu\llm\rea\decompile.log
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\arthu\llm\rea'
$log = "$root\decompile.log"
$env:JAVA_HOME = "$root\jdk-25.0.5+8"
$env:GHIDRA_INSTALL_DIR = "$root\ghidra_12.1.4_PUBLIC"
$env:PATH = "$root\node-v24.22.2-win-x64;$env:JAVA_HOME\bin;$env:PATH"
$rea = "$root\app\node_modules\.bin\rea.cmd"
"start $(Get-Date -Format s)" | Out-File $log -Encoding ascii
Set-Location "$root\target"
& 'C:\TDM-GCC-64\bin\gcc.exe' -O1 -fno-lto -o checksum.exe checksum.c 2>&1 | Add-Content $log
"build rc=$LASTEXITCODE; run: $(& .\checksum.exe local-forge)" | Add-Content $log
$t0 = Get-Date
$o = & $rea decompile --procedure forge_checksum "$root\target\checksum.exe" --format json --provider ghidra 2>&1 | Out-String
"decompile by name rc=$LASTEXITCODE secs=$([int]((Get-Date)-$t0).TotalSeconds) bytes=$($o.Length)" | Add-Content $log
$o | Out-File "$root\target\decompile-by-name.json" -Encoding utf8
($o -split "`n" | Select-Object -First 15) -join "`n" | Add-Content $log
if ($LASTEXITCODE -ne 0) {
  '--- list functions instead (stripped binary)' | Add-Content $log
  $t0 = Get-Date
  $l = & $rea --help 2>&1 | Out-String
  ($l -split "`n" | Where-Object { $_ -match 'function|procedure|symbols|list-' } | Select-Object -First 12) -join "`n" | Add-Content $log
}
"exit=0 $(Get-Date -Format s)" | Add-Content $log
