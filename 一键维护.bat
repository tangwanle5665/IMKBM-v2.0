@echo off
title Medical KB Maintenance
cd /d "D:\python work\医学知识库"
set "PY=D:\python work\med_venv\Scripts\python.exe"

:menu
cls
echo ==================================================
echo   Medical Knowledge Base - Maintenance
echo ==================================================
echo   [1] Start GUI (桌面版)
echo   [2] Parse docs  (PDF/Word/PPT 解析为文本)
echo   [3] Rebuild vector DB (重建向量库)
echo   [4] Update question bank (更新题库)
echo   [5] Download BGE model (下载模型)
echo   [6] Health check (健康检查)
echo   [0] Exit
echo ==================================================
set /p choice=Choose [0-6]: 
if "%choice%"=="1" goto gui
if "%choice%"=="2" goto parse
if "%choice%"=="3" goto build
if "%choice%"=="4" goto bank
if "%choice%"=="5" goto model
if "%choice%"=="6" goto check
if "%choice%"=="0" exit /b
goto menu

:gui
"%PY%" gui_app.py
pause & goto menu

:parse
"%PY%" -c "from pdf_parser import parse_files_to_txt; from ppt_parser import parse_all_ppt; parse_files_to_txt(); parse_all_ppt(); print('DONE')"
pause & goto menu

:build
"%PY%" build_vector_db.py
pause & goto menu

:bank
"%PY%" update_question_bank.py
pause & goto menu

:model
"%PY%" download_model.py
pause & goto menu

:check
"%PY%" -c "import sys; sys.path.insert(0,r'D:\python work\医学知识库'); import med_qa; print('med_qa import OK')"
pause & goto menu
