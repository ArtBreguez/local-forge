# Sweep B: flash attention + q8_0 KV cache at the server level, then re-run the layer grid.
$d = 'C:\Users\arthu\llm'
$log = "$d\bench-b-setup.log"
"start $(Get-Date -Format s)" | Out-File $log -Encoding ascii
[Environment]::SetEnvironmentVariable('OLLAMA_FLASH_ATTENTION', '1', 'User')
[Environment]::SetEnvironmentVariable('OLLAMA_KV_CACHE_TYPE', 'q8_0', 'User')
$env:OLLAMA_FLASH_ATTENTION = '1'; $env:OLLAMA_KV_CACHE_TYPE = 'q8_0'
Get-Process 'ollama app', ollama -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 3
Start-Process 'C:\Users\arthu\AppData\Local\Programs\Ollama\ollama app.exe'
for ($i = 0; $i -lt 30; $i++) {
  try { Invoke-RestMethod http://127.0.0.1:11434/api/version -TimeoutSec 3 | Out-Null; break } catch { Start-Sleep 2 }
}
'server up: ' + (Invoke-RestMethod http://127.0.0.1:11434/api/version).version | Add-Content $log
& C:\Python312\python.exe "$d\bench.py" b
$srv = Join-Path $env:LOCALAPPDATA 'Ollama\server.log'
Select-String -Path $srv -Pattern 'flash_attn|type_k|type_v|OLLAMA_FLASH|OLLAMA_KV' | Select-Object -Last 6 | ForEach-Object { $_.Line.Substring(0, [Math]::Min(200, $_.Line.Length)) } | Add-Content $log
"exit=0" | Add-Content $log
