#!/bin/bash

echo "==============================================="
echo "Quick Test - 5 Epochs (12 experiments)"
echo "MNIST and CIFAR-10 datasets - 6 models each"
echo "==============================================="
echo

# Clear previous results
if [ -f "results/train_report.txt" ]; then
    rm results/train_report.txt
fi

echo "Starting quick test at $(date)"

echo
echo "[1/6] Testing Baseline Model (5 epochs)..."
python 06_final_model.py --dataset mnist --epochs 5 --optimizer adam
if [ $? -ne 0 ]; then
    echo "ERROR: Quick test failed!"
    exit 1
fi

python 06_final_model.py --dataset cifar10 --epochs 5 --optimizer adam
if [ $? -ne 0 ]; then
    echo "ERROR: Quick test failed!"
    exit 1
fi

echo
echo "[2/6] Testing L2 Regularization (5 epochs)..."
python 06_final_model.py --dataset mnist --epochs 5 --weight_decay 0.01
if [ $? -ne 0 ]; then
    echo "ERROR: Quick test failed!"
    exit 1
fi

python 06_final_model.py --dataset cifar10 --epochs 5 --weight_decay 0.01
if [ $? -ne 0 ]; then
    echo "ERROR: Quick test failed!"
    exit 1
fi

echo
echo "[3/6] Testing Dropout (5 epochs)..."
python 06_final_model.py --dataset mnist --epochs 5 --dropout 0.5
if [ $? -ne 0 ]; then
    echo "ERROR: Quick test failed!"
    exit 1
fi

python 06_final_model.py --dataset cifar10 --epochs 5 --dropout 0.5
if [ $? -ne 0 ]; then
    echo "ERROR: Quick test failed!"
    exit 1
fi

echo
echo "[4/6] Testing SGD + Momentum (5 epochs)..."
python 06_final_model.py --dataset mnist --epochs 5 --optimizer sgd --lr 0.01 --momentum 0.9
if [ $? -ne 0 ]; then
    echo "ERROR: Quick test failed!"
    exit 1
fi

python 06_final_model.py --dataset cifar10 --epochs 5 --optimizer sgd --lr 0.01 --momentum 0.9
if [ $? -ne 0 ]; then
    echo "ERROR: Quick test failed!"
    exit 1
fi

echo
echo "[5/6] Testing All Regularizations (5 epochs)..."
python 06_final_model.py --dataset mnist --epochs 5 --optimizer sgd --lr 0.01 --momentum 0.9 --dropout 0.5 --weight_decay 0.01
if [ $? -ne 0 ]; then
    echo "ERROR: Quick test failed!"
    exit 1
fi

python 06_final_model.py --dataset cifar10 --epochs 5 --optimizer sgd --lr 0.01 --momentum 0.9 --dropout 0.5 --weight_decay 0.01
if [ $? -ne 0 ]; then
    echo "ERROR: Quick test failed!"
    exit 1
fi

echo
echo "[6/6] Testing Early Stopping (5 epochs)..."
python 06_final_model.py --dataset mnist --epochs 5 --early_stopping --patience 3
if [ $? -ne 0 ]; then
    echo "ERROR: Quick test failed!"
    exit 1
fi

python 06_final_model.py --dataset cifar10 --epochs 5 --early_stopping --patience 3
if [ $? -ne 0 ]; then
    echo "ERROR: Quick test failed!"
    exit 1
fi

echo
echo "==============================================="
echo "Quick test completed successfully!"
echo "Total experiments: 12 (5 epochs each)"
echo "==============================================="
echo
echo "Results saved to:"
echo "- results/train_report.txt (comprehensive log)"
echo "- results/*.png (training plots)"
echo
echo "Finished at $(date)"
