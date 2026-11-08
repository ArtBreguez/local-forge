# Install REA (morluto/rea, npm rea-agents 6.8.0) into an isolated folder on the host, with a portable Node 24 LTS,
# then run offline smoke tests against local files. Nothing global: no npm -g, no `rea setup`, no agent config changes.
# Log: C:\Users\arthu\llm\rea\rea-install.log
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\arthu\llm\rea'
$log = "$root\rea-install.log"
New-Item -ItemType Directory -Force $root | Out-Null
"start $(Get-Date -Format s)" | Out-File $log -Encoding ascii

# 1. portable Node 24 LTS, checksum-verified
$nv = 'v24.22.2'
$zipSha = 'ca1bff5d8ead13c9f2ea404a176db46bd430704d5b5113734049fa365e7799e2'
$nodeDir = "$root\node-$nv-win-x64"
if (-not (Test-Path "$nodeDir\node.exe")) {
  $zip = "$root\node.zip"
  Invoke-WebRequest "https://nodejs.org/dist/$nv/node-$nv-win-x64.zip" -OutFile $zip -UseBasicParsing
  $h = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
  "node zip sha256 $h expected $zipSha match=$($h -eq $zipSha)" | Add-Content $log
  if ($h -ne $zipSha) { 'ABORT checksum'; 'exit=1' | Add-Content $log; exit 1 }
  Expand-Archive $zip -DestinationPath $root -Force
  Remove-Item $zip
}
$env:PATH = "$nodeDir;$env:PATH"
"node $(& "$nodeDir\node.exe" -v) npm $(& "$nodeDir\npm.cmd" -v)" | Add-Content $log

# 2. REA into a local prefix, exact version, scripts disabled, integrity from the registry lockfile
$app = "$root\app"
New-Item -ItemType Directory -Force $app | Out-Null
Set-Location $app
if (-not (Test-Path "$app\package.json")) { '{"name":"rea-local","private":true}' | Out-File "$app\package.json" -Encoding ascii }
$t0 = Get-Date
& "$nodeDir\npm.cmd" install --no-audit --no-fund --ignore-scripts --omit=optional rea-agents@6.8.0 *> "$root\npm-install.txt"
"npm install rc=$LASTEXITCODE secs=$([int]((Get-Date)-$t0).TotalSeconds)" | Add-Content $log
Get-Content "$root\npm-install.txt" -Tail 4 | Add-Content $log
$rea = "$app\node_modules\.bin\rea.cmd"
"installed size MB: $([math]::Round(((Get-ChildItem $app\node_modules -Recurse -File | Measure-Object Length -Sum).Sum)/1MB,1))" | Add-Content $log
"integrity: $((Get-Content $app\package-lock.json | Select-String -Pattern '\"node_modules/rea-agents\"' -Context 0,4 | Out-String) -replace '\s+',' ')" | Add-Content $log

# 3. smoke tests, all offline targets that exist on this PC
'--- rea --version' | Add-Content $log
& $rea --version *>&1 | Select-Object -First 3 | Add-Content $log
'--- rea --help (head)' | Add-Content $log
& $rea --help *>&1 | Select-Object -First 40 | Add-Content $log
"exit=0 $(Get-Date -Format s)" | Add-Content $log
