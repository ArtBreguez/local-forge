@echo off
cd /d C:\Users\arthu\llm\agent-test
set XDG_DATA_HOME=C:\Users\arthu\llm\ocdata\data
set XDG_CONFIG_HOME=C:\Users\arthu\llm\ocdata\config
set XDG_CACHE_HOME=C:\Users\arthu\llm\ocdata\cache
set XDG_STATE_HOME=C:\Users\arthu\llm\ocdata\state
call "C:\Users\arthu\AppData\Roaming\npm\opencode.cmd" run --print-logs --model ollama/dsv4-9b-heretic "The tests in test/duration.test.js fail. Implement parseDuration in src/duration.js (read the comment at the top of that file and the tests) so that node --test passes. Do not modify the tests. Run node --test yourself to check your work, and fix failures until every test passes. When done, reply with a one-line summary." > C:\Users\arthu\llm\agent-out.txt 2> C:\Users\arthu\llm\agent-err.txt
echo agent-exit=%ERRORLEVEL% >> C:\Users\arthu\llm\agent-out.txt
