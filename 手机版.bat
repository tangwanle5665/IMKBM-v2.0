@echo off
title Medical KB - Mobile Web
cd /d "D:\python work\医学知识库"
echo ================================================
echo   Medical KB Web Server (for mobile access)
echo ================================================
echo [1/4] Releasing port 5000 if occupied by stale process...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
timeout /t 1 /nobreak >nul
echo [2/4] Allowing port 5000 in Windows firewall...
netsh advfirewall firewall delete rule name="MedicalKB5000" >nul 2>&1
netsh advfirewall firewall add rule name="MedicalKB5000" dir=in action=allow protocol=TCP localport=5000 >nul 2>&1
echo [3/4] Your computer LAN IP address:
ipconfig | findstr /i "IPv4"
echo [4/4] On your PHONE browser (same WiFi) open:
echo        http://THIS_IP:5000
echo ------------------------------------------------
echo Starting server... first start loads model, wait ~30s
"D:\python work\med_venv\Scripts\python.exe" web_app.py
pause
