foreach ($d in "$env:USERPROFILE\.local\share\opencode", "$env:USERPROFILE\.config\opencode", "$env:LOCALAPPDATA\opencode", "$env:APPDATA\opencode") {
  if (Test-Path $d) { "== $d"; Get-ChildItem $d -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 15 | ForEach-Object { '{0} {1} {2}' -f $_.LastWriteTime.ToString('s'), $_.Length, $_.FullName.Replace($d,'') } }
}
