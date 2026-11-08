# Install Ghidra 12.1.4 + Temurin JDK 25 (the versions REA 6.8.0 accepts: Ghidra 12.1.x, JDK 25+) into C:\Users\arthu\llm\rea\ (isolated, checksum-verified), then make REA
# decompile one function of a known binary. Log: C:\Users\arthu\llm\rea\ghidra.log
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\arthu\llm\rea'
$log = "$root\ghidra.log"
"start $(Get-Date -Format s)" | Out-File $log -Encoding ascii
foreach ($old in "$root\ghidra_12.2_PUBLIC", "$root\jdk-21.0.13+9") { if (Test-Path $old) { Remove-Item $old -Recurse -Force; "removed $old" | Add-Content $log } }
function Get-Verified($url, $sha, $dest) {
  if (-not (Test-Path $dest)) { Invoke-WebRequest $url -OutFile $dest -UseBasicParsing }
  $h = (Get-FileHash $dest -Algorithm SHA256).Hash.ToLower()
  "$(Split-Path $dest -Leaf) sha256 match=$($h -eq $sha)" | Add-Content $log
  if ($h -ne $sha) { Remove-Item $dest; 'ABORT checksum'; 'exit=1' | Add-Content $log; exit 1 }
}
$jdkZip = "$root\jdk25.zip"
$ghZip = "$root\ghidra.zip"
if (-not (Test-Path "$root\jdk-25.0.5+8\bin\java.exe")) {
  Get-Verified 'https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.5%2B8/OpenJDK25U-jdk_x64_windows_hotspot_25.0.5_8.zip' '636bdc13dc649ddc7d5921c5c7d23cca6e44fd19d662b45cc9c02135a6c96c42' $jdkZip
  Expand-Archive $jdkZip -DestinationPath $root -Force; Remove-Item $jdkZip
}
if (-not (Test-Path "$root\ghidra_12.1.4_PUBLIC\support\analyzeHeadless.bat")) {
  Get-Verified 'https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_12.1.4_build/ghidra_12.1.4_PUBLIC_20260921.zip' 'ddac49f903da9d5bac833e5cc79395098b9c33cfd3279be5f31bd00387d2d4db' $ghZip
  Expand-Archive $ghZip -DestinationPath $root -Force; Remove-Item $ghZip
}
$env:JAVA_HOME = "$root\jdk-25.0.5+8"
$env:GHIDRA_INSTALL_DIR = "$root\ghidra_12.1.4_PUBLIC"
$env:PATH = "$root\node-v24.22.2-win-x64;$env:JAVA_HOME\bin;$env:PATH"
"java: $((& "$env:JAVA_HOME\bin\java.exe" -version 2>&1 | Select-Object -First 1))" | Add-Content $log
"C free GB: $([math]::Round((Get-PSDrive C).Free/1GB,1))" | Add-Content $log
$rea = "$root\app\node_modules\.bin\rea.cmd"
'--- doctor (ghidra lines)' | Add-Content $log
(& $rea doctor --json 2>&1 | Out-String) -split '\},\{' | Where-Object { $_ -match '"name":"ghidra' } | ForEach-Object { $_.Substring(0, [Math]::Min(400, $_.Length)) } | Add-Content $log
'--- decompile help' | Add-Content $log
& $rea decompile --help 2>&1 | Select-Object -First 30 | Add-Content $log
"exit=0 $(Get-Date -Format s)" | Add-Content $log
