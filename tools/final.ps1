Set-Location C:\Users\arthu\llm\agent-test
$out = & node --test --test-reporter=spec 2>&1 | Out-String
($out -split "`n" | Where-Object { $_ -match 'RangeError for|Expected values|actual|expected|^\s*[x✖]|failing tests' } | Select-Object -First 12) -join "`n"
$e = 'C:\Users\arthu\llm\agent-err.txt'
'steps: ' + (Select-String -Path $e -Pattern 'step=').Count
'edits: ' + (Select-String -Path $e -Pattern 'Edit src').Count
'test runs: ' + (Select-String -Path $e -Pattern 'permission=bash').Count
'reads: ' + (Select-String -Path $e -Pattern ' Read ').Count
