@echo off
:: 设置UTF-8编码，防止中文提示乱码
chcp 65001 >nul 

echo ==========================================
echo [1/3] 正在执行 pendulum 配置...
echo ==========================================
:: python run_all.py --config configs/pendulum.yaml
:: python train.py --config configs/pendulum.yaml
python analyze.py --config configs/pendulum.yaml
python symbolic.py --config configs/pendulum.yaml
python evaluate.py --config configs/pendulum.yaml
python tta.py --config configs/pendulum.yaml --noise-std 0.05

echo.
echo ==========================================
echo [2/3] 正在执行 newton 配置...
echo ==========================================
python train.py --config configs/newton.yaml
python analyze.py --config configs/newton.yaml
python symbolic.py --config configs/newton.yaml
python evaluate.py --config configs/newton.yaml
python tta.py --config configs/newton.yaml --noise-std 0.05

echo.
echo ==========================================
echo [3/3] 正在执行 double_pendulum 配置...
echo ==========================================
python train.py --config configs/double_pendulum.yaml
python analyze.py --config configs/double_pendulum.yaml
python symbolic.py --config configs/double_pendulum.yaml
python evaluate.py --config configs/double_pendulum.yaml
python tta.py --config configs/double_pendulum.yaml --noise-std 0.05

echo.
echo ==========================================
echo 所有任务执行完毕！
echo ==========================================

:: 暂停，防止执行完后窗口直接关闭
pause