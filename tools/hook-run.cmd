@echo off
rem ===========================================================================
rem hook-run.cmd -- Windows launcher for the MCode project Hook.
rem
rem PURPOSE: pick the PowerShell to run tools/net-auto-recover.ps1 with.
rem
rem   pwsh              = PowerShell 7   (on many machines it is a per-user
rem                       Microsoft Store app-execution alias: a 0-byte
rem                       reparse point under WindowsApps, i.e. NOT a
rem                       guaranteed system binary -> may be absent)
rem   powershell.exe    = PowerShell 5.1 (C:\Windows\System32\... ->
rem                       always present)
rem   => prefer pwsh, fall back to powershell.exe.
rem
rem WHY A SEPARATE FILE (the reasoning in Chinese lives in the .ps1):
rem   1) "which interpreter" must be decided BEFORE the script starts, so it
rem      cannot live inside the .ps1.
rem   2) A "where pwsh >nul 2>&1 && pwsh ... || powershell.exe ..." chain
rem      written directly as the hook command BREAKS when the project path
rem      contains spaces (verified: exit 1, "The filename, directory name, or
rem      volume label syntax is incorrect"), because Node runs the hook as
rem      cmd.exe /d /s /c "<command>" and /s strips the outer quotes.
rem      The hook file is committed to the repo and may run on other people's
rem      machines, whose project path may contain spaces.
rem   3) %~dp0 derives the script path from THIS file, so the path itself
rem      never has to be spelled out (and spaces are fine).
rem
rem IMPORTANT: keep this file PURE ASCII. cmd.exe decodes .cmd files with the
rem OEM/ANSI code page, so UTF-8 non-ASCII bytes corrupt the parse and can
rem swallow the following command line entirely (verified: the whole file
rem stopped parsing when it contained Chinese comments).
rem
rem NOTE: cmd's "call" emits one spurious "The system cannot find the file
rem specified." on stderr when the invoked path contains spaces; that only
rem happens on space-containing project paths (normal paths are silent), and
rem the script still runs. The .ps1 passes its self-test on BOTH 5.1 and 7.
rem ===========================================================================
setlocal
set "SCRIPT=%~dp0net-auto-recover.ps1"

if not exist "%SCRIPT%" (
    echo hook-run: script not found: "%SCRIPT%" 1>&2
    exit /b 3
)

rem Branch with goto, NOT a parenthesized if-block: inside a block cmd expands
rem %errorlevel% at PARSE time, so the exit code would be wrong.
where pwsh >nul 2>&1
if errorlevel 1 goto :use_ps51

pwsh -NoProfile -File "%SCRIPT%" %*
exit /b %errorlevel%

:use_ps51
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
exit /b %errorlevel%
