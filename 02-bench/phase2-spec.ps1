# Phase 2: speculative decoding on llama-server.
# Step 1: find the most GPU layers that load at 16K context in llama-server (its compute buffer is sized for the full
#         context, unlike llama-bench's, so the bench limit of 16 layers does not carry over).
# Step 2: on that point, compare --spec-type none / ngram-simple / ngram-mod / ngram-cache / draft-mtp on fixed coding
#         prompts at temperature 0, recording tokens/s, draft acceptance, and whether the text is identical to `none`.
$ErrorActionPreference = 'Continue'
$d = 'C:\Users\arthu\llm'
$log = "$d\phase2.log"
$out = "$d\phase2.jsonl"
$srv = "$d\llama-b12207\llama-server.exe"
$model = 'C:\Users\arthu\.ollama\models\blobs\sha256-55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312'
$port = 8089
"start $(Get-Date -Format s)" | Out-File $log -Encoding ascii
'' | Out-File $out -Encoding ascii
try {
  (Invoke-RestMethod http://127.0.0.1:11434/api/ps).models | ForEach-Object {
    Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -Body (@{ model = $_.name; keep_alive = 0 } | ConvertTo-Json) | Out-Null
  }
} catch {}

function Start-Srv($ngl, $extra) {
  $args = @('-m', $model, '--host', '127.0.0.1', '--port', $port, '-ngl', $ngl, '-c', 16384, '-t', 6, '-tb', 6,
            '-fa', 'off', '-ctk', 'f16', '-ctv', 'f16', '-ub', 512, '-np', 1, '--fit', 'off') + $extra
  $p = Start-Process $srv -ArgumentList $args -PassThru -WindowStyle Hidden -RedirectStandardError "$d\phase2-srv.err" -RedirectStandardOutput "$d\phase2-srv.out"
  for ($i = 0; $i -lt 60; $i++) {
    if ($p.HasExited) { return $null }
    try { if ((Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 3).status -eq 'ok') { return $p } } catch {}
    Start-Sleep 2
  }
  Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
  return $null
}
function Stop-Srv($p) { if ($p) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Start-Sleep 3 } }

# step 1
$best = $null
foreach ($ngl in 15, 14, 13, 12, 11) {
  $p = Start-Srv $ngl @()
  $ok = [bool]$p
  "load ngl=$ngl ctx=16384 ok=$ok" | Add-Content $log
  Stop-Srv $p
  if ($ok) { $best = $ngl; break }
}
if (-not $best) { 'no layer count loads' | Add-Content $log; 'exit=1' | Add-Content $log; exit 1 }

# step 2
$prompts = @(
  'Write a Python function parse_duration(text) that turns strings like "1h30m", "90s" or "2h 5s" into seconds, rejecting repeated units with ValueError. Code only.',
  'Here is a function:\n\ndef total(items):\n    t = 0\n    for i in items:\n        t = t + i["price"] * i["qty"]\n    return t\n\nRewrite it to skip items whose qty is missing or negative, and add type hints. Return the whole function.',
  'Write a JavaScript module that exports debounce(fn, ms) and throttle(fn, ms), with JSDoc comments. Code only.',
  'Explain in five short bullet points how a hash map handles collisions.'
)
$specs = @(
  @{ name = 'none'; args = @() },
  @{ name = 'ngram-simple'; args = @('--spec-type', 'ngram-simple') },
  @{ name = 'ngram-mod'; args = @('--spec-type', 'ngram-mod') },
  @{ name = 'ngram-cache'; args = @('--spec-type', 'ngram-cache') },
  @{ name = 'draft-mtp'; args = @('--spec-type', 'draft-mtp') }
)
foreach ($s in $specs) {
  $p = Start-Srv $best $s.args
  if (-not $p) {
    $err = (Get-Content "$d\phase2-srv.err" -Tail 3 -ErrorAction SilentlyContinue) -join ' | '
    "spec=$($s.name) failed to start: $($err.Substring(0, [Math]::Min(220, $err.Length)))" | Add-Content $log
    @{ spec = $s.name; ngl = $best; error = 'failed to start' } | ConvertTo-Json -Compress | Add-Content $out
    continue
  }
  for ($rep = 1; $rep -le 2; $rep++) {
    for ($k = 0; $k -lt $prompts.Count; $k++) {
      $body = @{ model = 'local'; temperature = 0; seed = 1; max_tokens = 320; cache_prompt = $false;
                 messages = @(@{ role = 'user'; content = $prompts[$k] }) } | ConvertTo-Json -Depth 5
      try {
        $r = Invoke-RestMethod "http://127.0.0.1:$port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
        $txt = $r.choices[0].message.content
        $hash = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($txt))).Replace('-', '').Substring(0, 16)
        @{ spec = $s.name; ngl = $best; rep = $rep; prompt = $k; gen_tokens = $r.timings.predicted_n; gen_tps = $r.timings.predicted_per_second;
           prompt_tps = $r.timings.prompt_per_second; draft_n = $r.timings.draft_n; draft_accepted = $r.timings.draft_n_accepted; text_sha = $hash } |
          ConvertTo-Json -Compress | Add-Content $out
      } catch {
        @{ spec = $s.name; ngl = $best; rep = $rep; prompt = $k; error = "$_".Substring(0, 160) } | ConvertTo-Json -Compress | Add-Content $out
      }
    }
  }
  "spec=$($s.name) done" | Add-Content $log
  Stop-Srv $p
}
"exit=0 $(Get-Date -Format s)" | Add-Content $log
