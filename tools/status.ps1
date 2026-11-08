$d = 'C:\Users\arthu\llm'
'--- setup.log'
if (Test-Path "$d\setup.log") { Get-Content "$d\setup.log" } else { 'no setup.log' }
'--- pull tail'
if (Test-Path "$d\pull.log") { ((Get-Content "$d\pull.log" -Raw) -split "`r|`n" | Where-Object { $_ -match '\S' } | Select-Object -Last 2) -join ' | ' }
'--- npm tail'
if (Test-Path "$d\npm.log") { Get-Content "$d\npm.log" -Tail 3 }
'--- task'
schtasks /Query /TN llm-setup /FO LIST | Select-String 'Status|Estado|Ultimo|Last'
'C free GB: ' + [math]::Round((Get-PSDrive C).Free/1GB,1)
'RAM free GB: ' + [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)
