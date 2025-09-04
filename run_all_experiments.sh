#!/bin/bash

echo "==============================================="
echo "Running All LeNet Experiments (12 total)"
echo "MNIST and CIFAR-10 datasets - 6 models each"
echo "==============================================="
echo

# Clear previous results
if [ -f "results/train_report.txt" ]; then
    rm results/train_report.txt
fi

echo "Starting experiments at $(date)"

echo
echo "[1/6] Running Baseline Model..."
echo "Running: 01_baseline_model.py"
python 01_baseline_model.py
if [ $? -ne 0 ]; then
    echo "ERROR: Baseline model failed!"
    exit 1
fi

echo
echo "[2/6] Running L2 Regularization Model..."
echo "Running: 02_l2_regularization_model.py"
python 02_l2_regularization_model.py
if [ $? -ne 0 ]; then
    echo "ERROR: L2 regularization model failed!"
    exit 1
fi

echo
echo "[3/6] Running Dropout Model..."
echo "Running: 03_dropout_model.py"
python 03_dropout_model.py
if [ $? -ne 0 ]; then
    echo "ERROR: Dropout model failed!"
    exit 1
fi

echo
echo "[4/6] Running SGD + Momentum Model..."
echo "Running: 04_sgd_momentum_model.py"
python 04_sgd_momentum_model.py
if [ $? -ne 0 ]; then
    echo "ERROR: SGD momentum model failed!"
    exit 1
fi

echo
echo "[5/6] Running All Regularizations Model..."
echo "Running: 05_all_regularizations_model.py"
python 05_all_regularizations_model.py
if [ $? -ne 0 ]; then
    echo "ERROR: All regularizations model failed!"
    exit 1
fi

echo
echo "[6/6] Running Final Configurable Model (both datasets)..."
echo "Running: 06_final_model.py --dataset mnist"
python 06_final_model.py --dataset mnist
if [ $? -ne 0 ]; then
    echo "ERROR: Final model (MNIST) failed!"
    exit 1
fi

echo "Running: 06_final_model.py --dataset cifar10"
python 06_final_model.py --dataset cifar10
if [ $? -ne 0 ]; then
    echo "ERROR: Final model (CIFAR-10) failed!"
    exit 1
fi

echo
echo "==============================================="
echo "All experiments completed successfully!"
echo "Total experiments run: 12 (6 models x 2 datasets)"
echo "==============================================="
echo
echo "Results saved to:"
echo "- results/train_report.txt (comprehensive log)"
echo "- results/*.png (training plots)"
echo "- results/*.pth (model checkpoints)"
echo
echo "Finished at $(date)"
