@echo off
rem Pull the GGUF, build a 32K-context variant, install opencode. Logs to C:\Users\arthu\llm\setup.log
cd /d C:\Users\arthu\llm
set OLL=C:\Users\arthu\AppData\Local\Programs\Ollama\ollama.exe
echo [%date% %time%] pull start > setup.log
"%OLL%" pull hf.co/mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF:i1-IQ4_XS > pull.log 2>&1
echo pull exit=%ERRORLEVEL% >> setup.log
"%OLL%" create dsv4-9b-heretic -f C:\Users\arthu\llm\Modelfile > create.log 2>&1
echo create exit=%ERRORLEVEL% >> setup.log
"%OLL%" list >> setup.log 2>&1
call "C:\Program Files\nodejs\npm.cmd" i -g opencode-ai@latest > npm.log 2>&1
echo npm exit=%ERRORLEVEL% >> setup.log
call "%APPDATA%\npm\opencode.cmd" --version >> setup.log 2>&1
echo opencode-version exit=%ERRORLEVEL% >> setup.log
powershell -NoProfile -Command "[math]::Round((Get-PSDrive C).Free/1GB,1)" >> setup.log
echo exit=0 >> setup.log
