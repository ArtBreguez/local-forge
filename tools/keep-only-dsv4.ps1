# Remove every Ollama model except the DeepSeek-V4-Pro-Qwen3.5-9B heretic (both tags share one blob).
$o = 'C:\Users\arthu\AppData\Local\Programs\Ollama\ollama.exe'
'C free GB before: ' + [math]::Round((Get-PSDrive C).Free/1GB,1)
$names = & $o list | Select-Object -Skip 1 | ForEach-Object { ($_ -split '\s+')[0] } | Where-Object { $_ }
foreach ($m in $names) {
  if ($m -match 'dsv4-9b-heretic|DeepSeek-V4-Pro-Qwen3\.5-9B-UNCENSORED-heretic') { "keep $m" }
  else { "remove $m"; & $o rm $m }
}
'--- after'
& $o list
'C free GB after: ' + [math]::Round((Get-PSDrive C).Free/1GB,1)
