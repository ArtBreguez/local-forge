$d = 'C:\Users\arthu\llm'
'now: ' + (Get-Date -Format s)
'--- run log'; Get-Content "$d\agent-run.log" -ErrorAction SilentlyContinue | Select-Object -Last 12
'--- task: ' + (Get-ScheduledTask -TaskName llm-agent).State
'--- err last write: ' + (Get-Item "$d\agent-err.txt").LastWriteTime.ToString('s') + ' size ' + (Get-Item "$d\agent-err.txt").Length
'--- steps / tools'
Select-String -Path "$d\agent-err.txt" -Pattern 'step=|Read |Edit |Write |Bash |Shell |level=ERROR|error=' | Select-Object -Last 30 | ForEach-Object { $_.Line.Substring(0, [Math]::Min(200, $_.Line.Length)) }
'--- out tail'; Get-Content "$d\agent-out.txt" -Tail 25 -ErrorAction SilentlyContinue
'--- src'; Get-Content "$d\agent-test\src\duration.js"
'--- tests now'; Set-Location "$d\agent-test"; (& node --test --test-reporter=tap 2>&1 | Select-String -Pattern '^# (pass|fail)') -join ' '
'--- test hash'; (Get-FileHash test\duration.test.js).Hash
'--- ollama ps'; & C:\Users\arthu\AppData\Local\Programs\Ollama\ollama.exe ps
