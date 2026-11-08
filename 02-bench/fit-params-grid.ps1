# llama-fit-params grid on the host: llama.cpp's own estimate of model / context / compute buffer per device for a grid of
# (ngl, ctx, ubatch, flash attention, KV type). `-fit off -fitp on` prints the estimate and exits: no model is loaded (0.5 s per point),
# so the GPU is not held, but still check C:\Users\arthu\llm\sweep-c.log ends with `exit=` before running it.
# Output: C:\Users\arthu\llm\forge\cache\fit\grid.txt  (copied to results/fit-params-grid-host-20261108.txt)
# NB: `fit-params` prints whole MiB, truncated.
$ErrorActionPreference = 'Continue'
$d = 'C:\Users\arthu\llm\forge\cache\fit'
$m = 'C:\Users\arthu\.ollama\models\blobs\sha256-55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312'
$exe = 'C:\Users\arthu\llm\llama-b12207\llama-fit-params.exe'
$rows = New-Object System.Collections.Generic.List[string]
$rows.Add("# llama-fit-params b12207, -fit off -fitp on -np 1; columns: ngl ctx ub fa kv | Vulkan0 model context compute | Host model context compute")
function Run($ngl,$ctx,$ub,$fa,$kv) {
  $o = & $exe -m $m -ngl $ngl -c $ctx -ub $ub -fa $fa -ctk $kv -ctv $kv -np 1 -fit off -fitp on 2>&1 | ForEach-Object { "$_" }
  $v = ($o | Where-Object { $_ -match '^Vulkan0 ' }) -join ''
  $h = ($o | Where-Object { $_ -match '^Host ' }) -join ''
  if (-not $v) { $v = 'Vulkan0 - - -' }
  $rows.Add("$ngl $ctx $ub $fa $kv | $v | $h")
}
# A: compute/context vs ctx x ub x settings at ngl 16
foreach ($c in @(('off','f16'),('on','f16'),('on','q8_0'))) {
  foreach ($ctx in 640,2048,4096,8320,8704,12288,16384,24576,32768) {
    foreach ($ub in 128,256,512,1024) { Run 16 $ctx $ub $c[0] $c[1] }
  }
}
# B: dependence on ngl at the server config and at the bench config
foreach ($ngl in 0,1,2,4,8,12,13,14,15,16,17,18,19,20,24,28,32,33,99) {
  Run $ngl 16384 512 off f16
  Run $ngl 8704 512 off f16
  Run $ngl 8704 256 off f16
}
$rows | Set-Content -Encoding utf8 "$d\grid.txt"
"rows=$($rows.Count)"
