# 🧠 CNN Implementation: Enhanced LeNet for MNIST & CIFAR-10

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python&logoColor=white)](https://python.org/)

A comprehensive implementation of **Enhanced LeNet** architecture achieving **74.93% accuracy on CIFAR-10** and **99.17% on MNIST**. This repository contains 6 different training configurations exploring various regularization techniques, optimizers, and architectural improvements.

## 📌 Project Overview

This project implements and compares different CNN training strategies using LeNet architecture:

- **6 Training Scripts**: Each exploring different regularization and optimization techniques
- **Dual Dataset Support**: Automatic training and evaluation on both MNIST and CIFAR-10
- **Enhanced Architecture**: Improved LeNet achieving state-of-the-art performance for the model class
- **Comprehensive Analysis**: Detailed comparison of all approaches with timing and accuracy metrics

## 🏆 Key Results

| Model | MNIST Accuracy | CIFAR-10 Accuracy | Key Features |
|-------|----------------|-------------------|--------------|
| **Enhanced LeNet (06)** | **99.17%** | **74.93%** | Improved architecture + data augmentation |
| Dropout Model (03) | 99.29% | 58.33% | Strong regularization |
| Baseline Model (01) | 99.00% | 59.74% | Standard LeNet |
| SGD + Momentum (04) | 98.81% | - | Stable optimization |
| L2 Regularization (02) | 97.26% | 52.55% | Weight decay |
| All Techniques (05) | 97.41% | - | Combined regularization |

## 📂 Project Structure

```
CNN-Implementation/
├── 01_baseline_model.py              # 🔸 Standard LeNet (no regularization)
├── 02_l2_regularization_model.py     # 🔸 LeNet + L2 weight decay
├── 03_dropout_model.py               # 🔸 LeNet + Dropout layers
├── 04_sgd_momentum_model.py          # 🔸 LeNet + SGD with momentum
├── 05_all_regularizations_model.py   # 🔸 Combined techniques + early stopping
├── 06_final_model_improved.py        # 🔥 Enhanced LeNet (best performance)
├── run_all_experiments.bat           # � Run all 12 experiments (6 models × 2 datasets)
├── run_quick_test.bat                # ⚡ Quick 10-epoch test
├── requirements.txt                  # 📦 Python dependencies
├── README.md                         # 📖 This file
├── models/                           # 💾 Best model checkpoints
│   ├── best_improved_lenet_mnist.pth
│   └── best_improved_lenet_cifar10.pth
└── results/                          # 📊 Training outputs
    ├── train_report.txt              # Comprehensive experiment logs
    ├── comparison_table.txt          # Performance comparison table
    └── ImprovedLeNet_*.png           # Training plots
```

## ⚙️ Training Details

### Enhanced LeNet Architecture (Script 06):
```
Input (32×32) → Conv2d(32,5×5,pad=2) → MaxPool(2×2) → ReLU →
Conv2d(64,5×5,pad=2) → MaxPool(2×2) → ReLU →
Conv2d(128,3×3,pad=1) → ReLU → Flatten →
Linear(8192→256) → ReLU → Dropout(0.5) →
Linear(256→128) → ReLU → Dropout(0.5) →
Linear(128→10) → Output
```

### Training Configuration:
- **Epochs**: 50 (configurable)
- **Batch Size**: 128 (enhanced model), 64 (standard models)
- **Optimizers**: Adam (lr=0.001) or SGD (lr=0.01, momentum=0.9)
- **Regularization**: Dropout (0.5), L2 weight decay (0.0005-0.01)
- **Data Augmentation**: RandomCrop + RandomHorizontalFlip (CIFAR-10)
- **LR Scheduling**: StepLR (reduce by 0.1 every 15 epochs)

### Key Improvements in Enhanced Model:
- **8x more filters** in first layer (32 vs 4)
- **Additional convolutional layer** for better feature extraction
- **MaxPooling** instead of AvgPooling
- **Larger FC layers** with more capacity (~2.2M vs 60K parameters)
- **Data augmentation** for CIFAR-10 generalization

## 📊 Output Structure

### Training Report (`results/train_report.txt`):
Each experiment includes:
```
=============================
Experiment: Enhanced LeNet Model
Dataset: CIFAR10
Hyperparameters:
  Learning Rate: 0.01
  Batch Size: 128
  Epochs: 15
  Optimizer: SGD
  Dropout: 0.5
  Weight Decay: 0.0005
Training Logs:
  Epoch 1: Train Loss=2.0518, Train Acc=22.44%, Val Loss=1.8749, Val Acc=29.11%
  ...
  Epoch 15: Train Loss=0.8189, Train Acc=72.37%, Val Loss=0.7671, Val Acc=73.58%
Best Validation Accuracy: 73.58% (Epoch 15)
Final Test Accuracy: 74.93%
Total Training Time: 16.33 minutes
=============================
```

### Performance Plots:
- **Enhanced naming**: `ImprovedLeNet_cifar10_sgd_dropout0.5_wd5e-04_15epochs.png`
- **Triple layout**: Training loss, validation accuracy, combined plot
- **High quality**: 300 DPI with comprehensive legends

## 🚀 How to Run

### Prerequisites:
```bash
# Clone the repository
git clone https://github.com/PraTham-Patill/CNN-Implementation.git
cd CNN-Implementation

# Install dependencies
pip install -r requirements.txt
```

### Run All Experiments (Recommended):
```bash
# Windows
.\run_all_experiments.bat

# This will run all 6 models on both MNIST and CIFAR-10 (12 total experiments)
# Expected time: ~2-3 hours on CPU
```

### Quick Test (10 epochs):
```bash
# Windows
.\run_quick_test.bat

# Fast testing of all models (~30 minutes)
```

### Individual Model Testing:
```bash
# Test enhanced model on CIFAR-10
python 06_final_model_improved.py --dataset cifar10 --epochs 15

# Test baseline model on MNIST  
python 01_baseline_model.py

# Test with custom parameters
python 06_final_model_improved.py --dataset mnist --epochs 20 --lr 0.001
```

### Available Parameters (06_final_model_improved.py):
```bash
python 06_final_model_improved.py --help

Options:
  --dataset: 'mnist' or 'cifar10' (default: mnist)
  --batch_size: Batch size (default: 128)
  --lr: Learning rate (default: 0.01)
  --epochs: Number of epochs (default: 50)
  --early_stopping: Enable early stopping
  --weight_decay: L2 regularization (default: 0.0005)
  --dropout: Dropout rate (default: 0.5)
```

## 📦 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
numpy>=1.20.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🔬 Technical Highlights

### Performance Achievements:
- **CIFAR-10**: 74.93% accuracy (25% improvement over basic LeNet ~60%)
- **MNIST**: 99.17% accuracy with enhanced architecture
- **Training Efficiency**: 16.3 min for CIFAR-10, 11.8 min for MNIST

### Research Quality Features:
- **Reproducible Results**: Fixed random seeds (42) across all experiments
- **Comprehensive Logging**: Training metrics, validation tracking, timing analysis
- **Professional Visualization**: High-quality plots with descriptive naming
- **Statistical Analysis**: Automated comparison table generation

### Production-Ready Code:
- **Error Handling**: Robust dataset loading and model checkpointing
- **Memory Efficient**: Proper batch processing and gradient management
- **Configurable**: Command-line interface for hyperparameter tuning
- **Well Documented**: Comprehensive code comments and type hints

## � Use Cases

### Educational:
- Learn CNN fundamentals with LeNet
- Compare different regularization techniques
- Understand the impact of architectural improvements

### Research:
- Baseline for advanced CNN architectures
- Regularization technique comparison
- Hyperparameter optimization studies

### Practical:
- Quick prototyping for image classification
- Performance benchmarking
- Transfer learning starting point

## 📈 Expected Training Times

| Configuration | MNIST (50 epochs) | CIFAR-10 (50 epochs) |
|---------------|-------------------|----------------------|
| **CPU (Intel i5)** | ~15-20 minutes | ~25-35 minutes |
| **GPU (GTX 1660)** | ~3-5 minutes | ~8-12 minutes |
| **Quick Test (10 epochs)** | ~3-4 minutes | ~5-7 minutes |

## 🏅 Key Insights

1. **Architecture Matters**: Enhanced LeNet achieves 25% improvement on CIFAR-10
2. **Data Augmentation**: Critical for CIFAR-10 generalization
3. **Regularization**: Dropout shows excellent performance across datasets
4. **Optimization**: SGD with momentum + LR scheduling outperforms Adam for this architecture
5. **Scale**: MNIST benefits from any reasonable approach, CIFAR-10 requires careful tuning

## 🚫 What's Excluded (Gitignore)

- **Datasets** (`data/`): Auto-downloaded on first run
- **Most Model Checkpoints** (`*.pth`): Reproducible via training
- **Old Plots**: Only latest enhanced model plots included
- **Cache Files**: Python `__pycache__/`, system files

## 🎉 Getting Started

1. **Clone the repo** and install requirements
2. **Run quick test**: `.\run_quick_test.bat` (Windows)
3. **Check results**: Open `results/train_report.txt` for detailed analysis
4. **Run full experiments**: `.\run_all_experiments.bat` for complete comparison
5. **Experiment**: Try different parameters with `06_final_model_improved.py`

## 🤝 Contributing

This project serves as an educational and research baseline. Feel free to:
- Extend with more advanced architectures (ResNet, DenseNet)
- Add more datasets (Fashion-MNIST, SVHN)
- Implement additional regularization techniques
- Optimize for mobile/edge deployment

