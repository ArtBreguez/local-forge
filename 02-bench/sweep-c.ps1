# Phase 1 on llama.cpp: install the Vulkan build, then sweep GPU layers x attention/KV x depth with llama-bench.
# Waits for any running Ollama sweep to finish first, unloads every Ollama model, and logs each llama-bench call separately
# so an out-of-memory crash at one point cannot lose the others.
$ErrorActionPreference = 'Continue'
$d = 'C:\Users\arthu\llm'
$log = "$d\sweep-c.log"
$out = "$d\bench-c.jsonl"
"start $(Get-Date -Format s)" | Out-File $log -Encoding ascii

# wait for sweep B
for ($i = 0; $i -lt 240; $i++) {
  if ((Test-Path "$d\bench-b-setup.log") -and (Select-String -Path "$d\bench-b-setup.log" -Pattern '^exit=' -Quiet)) { break }
  if (-not (Test-Path "$d\bench-b-setup.log")) { break }
  Start-Sleep 15
}
"sweep B finished or absent at $(Get-Date -Format s)" | Add-Content $log

# install llama.cpp b12207 Vulkan
$lc = "$d\llama-b12207"
if (-not (Test-Path "$lc\llama-bench.exe")) {
  $zip = "$d\llama-b12207-vulkan.zip"
  Invoke-WebRequest 'https://github.com/ggml-org/llama.cpp/releases/download/b12207/llama-b12207-bin-win-vulkan-x64.zip' -OutFile $zip -UseBasicParsing
  Expand-Archive $zip -DestinationPath $lc -Force
  Remove-Item $zip
}
$bench = Get-ChildItem $lc -Recurse -Filter llama-bench.exe | Select-Object -First 1
"llama-bench: $($bench.FullName)" | Add-Content $log
& $bench.FullName --list-devices 2>&1 | Add-Content $log

# the same GGUF Ollama uses
$model = 'C:\Users\arthu\.ollama\models\blobs\sha256-55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312'

# unload Ollama models so the GPU is free
try {
  (Invoke-RestMethod http://127.0.0.1:11434/api/ps).models | ForEach-Object {
    Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -Body (@{ model = $_.name; keep_alive = 0 } | ConvertTo-Json) | Out-Null
  }
} catch {}
Start-Sleep 5

$os = Get-CimInstance Win32_OperatingSystem
"free RAM GB before: $([math]::Round($os.FreePhysicalMemory/1MB,1))" | Add-Content $log

'' | Out-File $out -Encoding ascii
$attn = @(@('off', 'f16'), @('on', 'f16'), @('on', 'q8_0'))
foreach ($ngl in 0, 8, 12, 16, 20, 24, 28, 99) {
  foreach ($a in $attn) {
    $fa = $a[0]; $kv = $a[1]
    $t0 = Get-Date
    $res = & $bench.FullName -m $model -ngl $ngl -fa $fa -ctk $kv -ctv $kv -t 6 -p 512 -n 128 -d '0,8192' -r 3 -o jsonl 2> "$d\bench-c-last.err"
    $rc = $LASTEXITCODE
    $secs = [int]((Get-Date) - $t0).TotalSeconds
    if ($res) { $res | Add-Content $out }
    $tail = (Get-Content "$d\bench-c-last.err" -Tail 2 -ErrorAction SilentlyContinue) -join ' | '
    "ngl=$ngl fa=$fa kv=$kv rc=$rc secs=$secs lines=$(@($res).Count) $(if ($rc -ne 0) { $tail })" | Add-Content $log
  }
}
"exit=0 $(Get-Date -Format s)" | Add-Content $log
