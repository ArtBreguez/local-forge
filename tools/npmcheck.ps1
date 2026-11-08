Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'node|npm' } | ForEach-Object { '{0} {1} {2}' -f $_.ProcessId, $_.CreationDate, ($_.CommandLine -replace '\s+',' ').Substring(0, [Math]::Min(160, ($_.CommandLine -replace '\s+',' ').Length)) }
'npm.log bytes: ' + (Get-Item C:\Users\arthu\llm\npm.log).Length
'npm cache _logs newest:'
Get-ChildItem "$env:LOCALAPPDATA\npm-cache\_logs" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Desc | Select-Object -First 1 | ForEach-Object { $_.FullName; Get-Content $_.FullName -Tail 4 }
Test-Path "$env:APPDATA\npm\opencode.cmd"
