$logs = Get-ChildItem C:\Users\arthu\llm\forge\cache\logs\llama-server-20261108-14*.log | Sort-Object LastWriteTime
foreach ($f in $logs) {
  "== $($f.Name)"
  Select-String -Path $f.FullName -Pattern 'Vulkan0 model buffer|Vulkan0 KV buffer|Vulkan0 compute buffer|Vulkan0 RS buffer|offloaded|n_ctx  |n_ctx_seq|failed|OutOfDeviceMemory|unable to allocate' | Select-Object -First 14 | ForEach-Object { $_.Line.Substring(0, [Math]::Min(170, $_.Line.Length)) }
}
