'npm prefix: ' + (& 'C:\Program Files\nodejs\npm.cmd' prefix -g 2>$null)
Get-ChildItem -Path "$env:APPDATA\npm", 'C:\Program Files\nodejs', "$env:LOCALAPPDATA\hermes\node" -Filter 'opencode*' -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
(Get-Command opencode -ErrorAction SilentlyContinue).Source
