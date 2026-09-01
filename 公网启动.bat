@echo off
title Medical KB - Public Access (cpolar)
cd /d "%~dp0.."
set "KB=%CD%"
echo ==================================================
echo   Starting Medical KB Web + cpolar public tunnel
echo   Keep this window open while sharing.
echo ==================================================
echo [1/3] Releasing port 5000 if occupied by stale process...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
timeout /t 1 /nobreak >nul
start "MedicalKB-Web" cmd /k ""%KB%\med_venv\Scripts\python.exe" "%KB%\web_app.py""
echo [2/3] Waiting for web server to be ready (up to 90s)...
set /a tries=0
:waitloop
timeout /t 3 /nobreak >nul
powershell -NoProfile -Command "try{(Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:5000/' -TimeoutSec 2)|Out-Null; exit 0}catch{exit 1}" >nul 2>&1
if not errorlevel 1 goto ready
set /a tries+=1
if %tries% GEQ 30 goto ready
goto waitloop
:ready
echo [3/3] Web server is ready. Looking for cpolar...

set "CPOLAR_EXE="
if exist "D:\cpolar\cpolar.exe" set "CPOLAR_EXE=D:\cpolar\cpolar.exe"
if not defined CPOLAR_EXE (
    for %%i in (cpolar.exe) do if not "%%~$PATH:i"=="" set "CPOLAR_EXE=cpolar"
)
if not defined CPOLAR_EXE (
    echo.
    echo   [ERROR] cpolar not found!
    echo   请先安装 cpolar：
    echo     1. 下载 Windows 版:  https://www.cpolar.cn/download
    echo     2. 解压后把 cpolar.exe 放到 D:\cpolar\  (或加入系统 PATH)
    echo     3. 配置账号:  cpolar authtoken 你的Token
    echo        (Token 在 cpolar 官网-验证-authtoken 页面查看)
    echo     4. 重新运行本脚本
    pause & exit /b 1
)
echo   Using cpolar: %CPOLAR_EXE%
"%CPOLAR_EXE%" http 127.0.0.1:5000 --log stdout
pause
