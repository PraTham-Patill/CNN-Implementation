@echo off
echo ===============================================
echo Quick Test - 5 Epochs (12 experiments)
echo MNIST and CIFAR-10 datasets - 6 models each
echo ===============================================
echo.

REM Clear previous results
if exist results\train_report.txt del results\train_report.txt
echo Starting quick test at %date% %time%

echo.
echo [1/6] Testing Baseline Model (5 epochs)...
python 06_final_model.py --dataset mnist --epochs 5 --optimizer adam
if %errorlevel% neq 0 (
    echo ERROR: Quick test failed!
    pause
    exit /b 1
)

python 06_final_model.py --dataset cifar10 --epochs 5 --optimizer adam
if %errorlevel% neq 0 (
    echo ERROR: Quick test failed!
    pause
    exit /b 1
)

echo.
echo [2/6] Testing L2 Regularization (5 epochs)...
python 06_final_model.py --dataset mnist --epochs 5 --weight_decay 0.01
if %errorlevel% neq 0 (
    echo ERROR: Quick test failed!
    pause
    exit /b 1
)

python 06_final_model.py --dataset cifar10 --epochs 5 --weight_decay 0.01
if %errorlevel% neq 0 (
    echo ERROR: Quick test failed!
    pause
    exit /b 1
)

echo.
echo [3/6] Testing Dropout (5 epochs)...
python 06_final_model.py --dataset mnist --epochs 5 --dropout 0.5
if %errorlevel% neq 0 (
    echo ERROR: Quick test failed!
    pause
    exit /b 1
)

python 06_final_model.py --dataset cifar10 --epochs 5 --dropout 0.5
if %errorlevel% neq 0 (
    echo ERROR: Quick test failed!
    pause
    exit /b 1
)

echo.
echo [4/6] Testing SGD + Momentum (5 epochs)...
python 06_final_model.py --dataset mnist --epochs 5 --optimizer sgd --lr 0.01 --momentum 0.9
if %errorlevel% neq 0 (
    echo ERROR: Quick test failed!
    pause
    exit /b 1
)

python 06_final_model.py --dataset cifar10 --epochs 5 --optimizer sgd --lr 0.01 --momentum 0.9
if %errorlevel% neq 0 (
    echo ERROR: Quick test failed!
    pause
    exit /b 1
)

echo.
echo [5/6] Testing All Regularizations (5 epochs)...
python 06_final_model.py --dataset mnist --epochs 5 --optimizer sgd --lr 0.01 --momentum 0.9 --dropout 0.5 --weight_decay 0.01
if %errorlevel% neq 0 (
    echo ERROR: Quick test failed!
    pause
    exit /b 1
)

python 06_final_model.py --dataset cifar10 --epochs 5 --optimizer sgd --lr 0.01 --momentum 0.9 --dropout 0.5 --weight_decay 0.01
if %errorlevel% neq 0 (
    echo ERROR: Quick test failed!
    pause
    exit /b 1
)

echo.
echo [6/6] Testing Early Stopping (5 epochs)...
python 06_final_model.py --dataset mnist --epochs 5 --early_stopping --patience 3
if %errorlevel% neq 0 (
    echo ERROR: Quick test failed!
    pause
    exit /b 1
)

python 06_final_model.py --dataset cifar10 --epochs 5 --early_stopping --patience 3
if %errorlevel% neq 0 (
    echo ERROR: Quick test failed!
    pause
    exit /b 1
)

echo.
echo ===============================================
echo Quick test completed successfully!
echo Total experiments: 12 (5 epochs each)
echo ===============================================
echo.
echo Results saved to:
echo - results\train_report.txt (comprehensive log)
echo - results\*.png (training plots)
echo.
echo Finished at %date% %time%
pause
