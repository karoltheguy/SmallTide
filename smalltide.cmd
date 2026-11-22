@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0smalltide.ps1" %*
exit /b %ERRORLEVEL%
