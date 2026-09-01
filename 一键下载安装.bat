@echo off
title Medical KB - Full Download & Install
cd /d "D:\python work\医学知识库"

set "MINICONDA_DIR=D:\miniconda3"
set "MC_PY=%MINICONDA_DIR%\python.exe"
set "VENV_PY=D:\python work\med_venv\Scripts\python.exe"

echo ==================================================
echo   [0/5] Checking miniconda...
if exist "%MC_PY%" goto have_miniconda
echo   miniconda NOT found, will download and install automatically.

echo   Downloading Miniconda3 (Tsinghua mirror, ~90MB)...
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile '%TEMP%\miniconda3.exe' -TimeoutSec 600 } catch { exit 1 }"
if errorlevel 1 (
    echo   Tsinghua mirror failed, try official source...
    curl -L -o "%TEMP%\miniconda3.exe" https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
)
if not exist "%TEMP%\miniconda3.exe" (
    echo   [ERROR] Download failed. Please download it manually and put it to:
    echo     https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe
    pause & exit /b 1
)
echo   Installing miniconda silently to %MINICONDA_DIR% ...
start /wait "" "%TEMP%\miniconda3.exe" /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D=%MINICONDA_DIR%
if not exist "%MC_PY%" (
    echo   [ERROR] Miniconda install failed. Please install manually to D:\miniconda3
    echo   (download: https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/ )
    pause & exit /b 1
)
del "%TEMP%\miniconda3.exe" >nul 2>&1
:have_miniconda
echo   miniconda found: %MC_PY%
"%MC_PY%" --version

echo ==================================================
echo   [1/5] Creating virtual env (med_venv) with miniconda python...
"%MC_PY%" -m venv "D:\python work\med_venv" --clear
if not exist "%VENV_PY%" (
    echo   [ERROR] venv creation failed.
    pause & exit /b 1
)

set "PY=%VENV_PY%"
echo [2/5] Set pip mirror (Aliyun) and install dependencies...
"%PY%" -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install numpy scipy scikit-learn pandas torch faiss-cpu onnxruntime sentence-transformers transformers pymupdf python-docx python-pptx langchain-text-splitters langchain-community tqdm joblib matplotlib networkx Pillow flask requests python-dotenv comtypes onnx onnxscript

echo [3/5] Downloading BGE model via hf-mirror...
set "HF_ENDPOINT=https://hf-mirror.com"
"%PY%" download_model.py

echo [4/5] Verify Python version...
"%PY%" --version
echo [5/5] All done! Run "汤臣一品\AAA建材王总.bat" to start GUI.
pause
