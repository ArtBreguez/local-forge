$oll = 'C:\Users\arthu\AppData\Local\Programs\Ollama\ollama.exe'
& $oll show dsv4-9b-heretic --modelfile 2>$null | Select-String -Pattern '^FROM|^PARAMETER' | ForEach-Object { $_.Line }
& $oll show dsv4-9b-heretic 2>$null | Select-String -Pattern 'architecture|parameters|quantization|embedding|context' | ForEach-Object { $_.Line.Trim() }
$log = Join-Path $env:LOCALAPPDATA 'Ollama\server.log'
Select-String -Path $log -Pattern 'block_count|n_layer|offloaded \d+/\d+|layers.model|layers.offload|flash_attn|type_k|n_threads|nextn|mtp' | Select-Object -Last 10 | ForEach-Object { $_.Line.Substring(0, [Math]::Min(200, $_.Line.Length)) }
'--- ollama procs'
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'ollama' } | ForEach-Object { '{0} {1}' -f $_.ProcessId, $_.CommandLine }
'--- env'
[Environment]::GetEnvironmentVariables('User').GetEnumerator() | Where-Object { $_.Key -match 'OLLAMA' } | ForEach-Object { "$($_.Key)=$($_.Value)" }
'--- mem'
$os = Get-CimInstance Win32_OperatingSystem
'free RAM GB ' + [math]::Round($os.FreePhysicalMemory/1MB,1) + ' / ' + [math]::Round($os.TotalVisibleMemorySize/1MB,1)
Get-CimInstance Win32_PageFileUsage | ForEach-Object { 'pagefile ' + $_.Name + ' alloc MB ' + $_.AllocatedBaseSize + ' used ' + $_.CurrentUsage }
'python: ' + ((Get-Command python -ErrorAction SilentlyContinue).Source)
