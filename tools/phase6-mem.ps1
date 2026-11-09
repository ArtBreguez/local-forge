# Phase 6 (issue #23): a light memory sampler for the host, next to a benchmark. One line every -Every seconds into -Out:
#   <time> avail_mb pages_s faults_s pagefile_used_mb llama_ws_mb java_ws_mb vm_ws_mb
# WMI instead of typeperf: this host is pt-BR, the performance counters have localized names (`\Memory\Pages/sec` does not exist there).
# Ends by itself after -Count samples. Started as a scheduled task whose trigger cannot fire on its own, run with `schtasks /Run`, deleted when done.
param([string]$Out = 'C:\Users\arthu\llm\phase6\mem.log', [int]$Every = 60, [int]$Count = 200)
function Sum-Ws($name) { $p = Get-Process $name -ErrorAction SilentlyContinue; if ($p) { [int](($p | Measure-Object WorkingSet64 -Sum).Sum / 1MB) } else { 0 } }
for ($i = 0; $i -lt $Count; $i++) {
  try {
    $m = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $pf = Get-CimInstance Win32_PageFileUsage
    $line = "{0} avail_mb={1} pages_s={2} faults_s={3} pagefile_used_mb={4} llama_ws_mb={5} java_ws_mb={6} vm_ws_mb={7}" -f (Get-Date -Format s), $m.AvailableMBytes, $m.PagesPersec, $m.PageFaultsPersec, $pf.CurrentUsage, (Sum-Ws 'llama-server'), (Sum-Ws 'java'), (Sum-Ws 'VirtualBoxVM')
  } catch { $line = "{0} sample failed: {1}" -f (Get-Date -Format s), $_.Exception.Message }
  Add-Content -Path $Out -Value $line -Encoding ascii
  Start-Sleep -Seconds $Every
}
