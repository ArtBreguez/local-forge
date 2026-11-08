# Phase 1, fine grid around the sweep-C edge: 16..20 GPU layers, flash attention off (sweep C: FA on halves prompt speed
# at depth on this Vulkan build), physical batch 512 vs 256 (a smaller ubatch shrinks the compute buffer, which may let
# 20 layers fit), plus FA on + q8_0 KV at 18/20 to see whether the smaller KV buys a layer.
# Holds the GPU mutex: sweep-c.log loses its exit= line while this runs and gets it back at the end, so any other job that
# waits for that line (the hw-adapt CLI's tune step) does not load a model at the same time.
$ErrorActionPreference = 'Continue'
$d = 'C:\Users\arthu\llm'
$log = "$d\sweep-d.log"
$out = "$d\bench-d.jsonl"
$mutex = "$d\sweep-c.log"
$saved = Get-Content $mutex
$saved | Where-Object { $_ -notmatch '^exit=' } | Set-Content $mutex -Encoding ascii
"GPU held by sweep D since $(Get-Date -Format s)" | Add-Content $mutex
"start $(Get-Date -Format s)" | Out-File $log -Encoding ascii
$bench = (Get-ChildItem "$d\llama-b12207" -Recurse -Filter llama-bench.exe | Select-Object -First 1).FullName
$model = 'C:\Users\arthu\.ollama\models\blobs\sha256-55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312'
try {
  (Invoke-RestMethod http://127.0.0.1:11434/api/ps).models | ForEach-Object {
    Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -Body (@{ model = $_.name; keep_alive = 0 } | ConvertTo-Json) | Out-Null
  }
} catch {}
Start-Sleep 5
'' | Out-File $out -Encoding ascii
$points = @()
foreach ($ngl in 16, 17, 18, 19, 20) { foreach ($ub in 512, 256) { $points += , @($ngl, 'off', 'f16', $ub) } }
foreach ($ngl in 18, 20) { $points += , @($ngl, 'on', 'q8_0', 256) }
foreach ($p in $points) {
  $ngl, $fa, $kv, $ub = $p
  $t0 = Get-Date
  $res = & $bench -m $model -ngl $ngl -fa $fa -ctk $kv -ctv $kv -ub $ub -t 6 -p 512 -n 128 -d '0,8192' -r 3 -o jsonl 2> "$d\bench-d-last.err"
  $rc = $LASTEXITCODE
  if ($res) { $res | Add-Content $out }
  $tail = (Get-Content "$d\bench-d-last.err" -Tail 2 -ErrorAction SilentlyContinue) -join ' | '
  "ngl=$ngl fa=$fa kv=$kv ub=$ub rc=$rc secs=$([int]((Get-Date) - $t0).TotalSeconds) lines=$(@($res).Count) $(if ($rc -ne 0) { $tail.Substring(0, [Math]::Min(160, $tail.Length)) })" | Add-Content $log
}
"exit=0 $(Get-Date -Format s)" | Add-Content $log
$saved | Set-Content $mutex -Encoding ascii
