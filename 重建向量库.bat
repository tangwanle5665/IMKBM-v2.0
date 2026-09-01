@echo off
title 重建向量库（单进程·省内存）
cd /d "D:\python work\医学知识库"
echo ==================================================
echo  重建向量库（fastembed v1.5 新模型）
echo  单进程流式编码，内存占用最小
echo  预计 2~3 小时，期间请勿关闭窗口
echo  进度条显示当前编码进度
echo ==================================================
"D:\python work\med_venv\Scripts\python.exe" "D:\python work\医学知识库\汤臣一品\_reencode.py"
echo.
echo 完成！可关闭窗口，启动知识库正常使用。
pause
