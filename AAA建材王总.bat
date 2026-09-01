@echo off
cd /d "D:\python work\医学知识库"
echo 当前路径: %cd%
echo.
"D:\python work\med_venv\Scripts\python.exe" --version
echo.
echo 正在启动 GUI (虚拟环境 med_venv)...
"D:\python work\med_venv\Scripts\python.exe" gui_app.py
pause