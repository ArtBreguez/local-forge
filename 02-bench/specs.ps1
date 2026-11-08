$c = Get-CimInstance Win32_Processor
'CPU: ' + $c.Name + ' | cores ' + $c.NumberOfCores + ' | threads ' + $c.NumberOfLogicalProcessors
$m = Get-CimInstance Win32_PhysicalMemory
'RAM: ' + [math]::Round(($m | Measure-Object Capacity -Sum).Sum/1GB) + ' GB | ' + ($m | ForEach-Object { "$($_.Speed)MT/s" }) -join ','
$g = Get-CimInstance Win32_VideoController
$g | ForEach-Object { 'GPU: ' + $_.Name + ' | driver ' + $_.DriverVersion }
'Ollama: ' + (& C:\Users\arthu\AppData\Local\Programs\Ollama\ollama.exe --version 2>$null)
$log = Join-Path $env:LOCALAPPDATA 'Ollama\server.log'
if (Test-Path $log) { Select-String -Path $log -Pattern 'inference compute|library=|offloaded|vulkan|rocm|ROCm|Vulkan' | Select-Object -Last 6 | ForEach-Object { $_.Line.Substring(0, [Math]::Min(220, $_.Line.Length)) } }
'Disks:'; Get-PSDrive -PSProvider FileSystem | ForEach-Object { '{0}: free {1} GB of {2} GB' -f $_.Name, [math]::Round($_.Free/1GB,1), [math]::Round(($_.Used+$_.Free)/1GB) }
'OS: ' + (Get-CimInstance Win32_OperatingSystem).Caption
