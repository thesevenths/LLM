@echo off
:: 设置UTF-8编码，防止中文提示乱码
chcp 65001 >nul 

echo ==========================================
echo [1/3] 正在执行 pendulum 配置...
echo ==========================================
python run_all.py --config configs/pendulum.yaml

echo.
echo ==========================================
echo [2/3] 正在执行 newton 配置...
echo ==========================================
python run_all.py --config configs/newton.yaml

echo.
echo ==========================================
echo [3/3] 正在执行 double_pendulum 配置...
echo ==========================================
python run_all.py --config configs/double_pendulum.yaml

echo.
echo ==========================================
echo 所有任务执行完毕！
echo ==========================================

:: 暂停，防止执行完后窗口直接关闭
pause