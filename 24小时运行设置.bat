@echo off
title 24小时在线运行设置
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在请求管理员权限...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)
echo ==================================================
echo  24小时在线运行设置（无人值守）
echo ==================================================
echo.
echo [1/3] 禁用睡眠（接通电源时从不睡眠）...
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
echo  完成：连接电源时电脑永不睡眠
echo.
echo [2/3] 关闭休眠文件（释放 C 盘空间）...
powercfg /h off
echo  完成
echo.
echo [3/3] 显示器可在数分钟后关闭（不影响服务运行）...
powercfg /change monitor-timeout-ac 10
echo  完成：显示器 10 分钟无操作自动关闭，主机保持运行
echo.
echo ==================================================
echo  防小孩乱动小贴士：
echo   1. 设置 Windows 登录密码：设置-账户-登录选项-密码
echo   2. 无人时按 Win+L 锁屏，小孩打不开
echo   3. 可把公网启动.bat 加入开机自启（见说明）
echo ==================================================
echo.
echo 全部设置完成！电脑将保持唤醒，公网服务可 24 小时访问。
pause
