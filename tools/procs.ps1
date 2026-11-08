Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'opencode|run-agent' } | ForEach-Object { '{0} {1} {2}' -f $_.ProcessId, $_.CreationDate.ToString('s'), $_.Name }
