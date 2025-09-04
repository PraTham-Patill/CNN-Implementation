import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import argparse
import os
import time
import numpy as np
import random
from datetime import datetime

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImprovedLeNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5, input_channels=1):
        super(ImprovedLeNet, self).__init__()
        # Enhanced architecture for better CIFAR-10 performance
        self.conv1 = nn.Conv2d(input_channels, 32, 5, padding=2)  # 32x32 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)             # 32x32 -> 32x32
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)            # 16x16 -> 16x16
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the size for FC layer based on input size
        # After 2 pooling operations: 32x32 -> 16x16 -> 8x8
        # MNIST is resized to 32x32, so same calculation applies
        fc_input_size = 128 * 8 * 8
            
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # First conv block
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 16x16
        # Second conv block  
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 -> 8x8
        # Third conv block (no pooling)
        x = self.relu(self.conv3(x))             # Keep spatial dimensions at 8x8
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers with dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_transforms(dataset):
    """Get appropriate transforms for each dataset"""
    if dataset == 'mnist':
        # Simple transforms for MNIST
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to 32x32 for consistency
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transform_train
    else:  # CIFAR-10
        # Data augmentation for CIFAR-10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    return transform_train, transform_test

def get_dataset(dataset_name, train_transform, test_transform):
    """Load and return train/val/test datasets"""
    if dataset_name == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=test_transform
        )
        input_channels = 1
    else:  # CIFAR-10
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )
        input_channels = 3
    
    # Split training data into train and validation (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    return train_subset, val_subset, test_dataset, input_channels

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(val_loader)
    return avg_loss, accuracy

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_accuracy = 100. * correct / total
    test_loss = test_loss / len(test_loader)
    return test_loss, test_accuracy

def save_plots(train_losses, val_losses, train_accs, val_accs, model_name, dataset, args):
    """Save training plots with descriptive names"""
    # Create filename
    dropout_str = f"dropout{args.dropout:.1f}" if args.dropout > 0 else "nodropout"
    wd_str = f"wd{args.weight_decay:.0e}" if args.weight_decay > 0 else "nowd"
    filename = f"{model_name}_{dataset}_{args.optimizer}_{dropout_str}_{wd_str}_{args.epochs}epochs.png"
    
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title(f'Training & Validation Loss\n{model_name} - {dataset.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc', color='blue')
    plt.plot(val_accs, label='Val Acc', color='red')
    plt.title(f'Training & Validation Accuracy\n{model_name} - {dataset.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Combined plot
    plt.subplot(1, 3, 3)
    plt.plot(train_losses, label='Train Loss', color='blue', linestyle='-')
    plt.plot(val_losses, label='Val Loss', color='red', linestyle='-')
    ax2 = plt.gca().twinx()
    ax2.plot(train_accs, label='Train Acc', color='blue', linestyle='--', alpha=0.7)
    ax2.plot(val_accs, label='Val Acc', color='red', linestyle='--', alpha=0.7)
    plt.title(f'Combined Loss & Accuracy\n{model_name} - {dataset.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax2.set_ylabel('Accuracy (%)')
    
    # Combine legends
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join('results', filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def run_experiment(args):
    set_seed(42)  # Set seed for reproducibility
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get transforms and datasets
    train_transform, test_transform = get_transforms(args.dataset)
    train_dataset, val_dataset, test_dataset, input_channels = get_dataset(
        args.dataset, train_transform, test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    # Create model
    model = ImprovedLeNet(num_classes=10, dropout_rate=args.dropout, input_channels=input_channels)
    model = model.to(device)
    
    print(f"Model parameters: {count_parameters(model)}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            os.makedirs('results', exist_ok=True)
            torch.save(model.state_dict(), f'results/best_improved_lenet_{args.dataset}.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Update learning rate
        scheduler.step()
    
    training_time = time.time() - start_time
    
    # Load best model for testing
    model.load_state_dict(torch.load(f'results/best_improved_lenet_{args.dataset}.pth'))
    test_loss, test_acc = test_model(model, test_loader, criterion, device)
    
    # Save plots
    plot_filename = save_plots(train_losses, val_losses, train_accs, val_accs, 
                              "ImprovedLeNet", args.dataset, args)
    
    # Determine regularization techniques used
    reg_techniques = []
    if args.dropout > 0:
        reg_techniques.append(f"Dropout (rate = {args.dropout})")
    if args.weight_decay > 0:
        reg_techniques.append(f"L2 (weight_decay = {args.weight_decay})")
    if args.optimizer == 'sgd':
        reg_techniques.append(f"SGD with Momentum (momentum = {args.momentum})")
    if args.early_stopping:
        reg_techniques.append(f"Early Stopping (patience = {args.patience})")
    
    reg_used = ", ".join(reg_techniques) if reg_techniques else "None"
    
    # Log results
    log_results(args, train_losses, val_losses, train_accs, val_accs, 
               best_val_acc, best_epoch, test_loss, test_acc, training_time, 
               reg_used, plot_filename)
    
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Total Training Time: {training_time/60:.2f} minutes")
    print("Training completed and results saved!")

def log_results(args, train_losses, val_losses, train_accs, val_accs, 
               best_val_acc, best_epoch, test_loss, test_acc, training_time, 
               reg_used, plot_filename):
    """Log detailed results to train_report.txt"""
    os.makedirs('results', exist_ok=True)
    
    with open('results/train_report.txt', 'a') as f:
        f.write("=============================\n")
        f.write("Experiment: Improved LeNet Model\n")
        f.write(f"Dataset: {args.dataset.upper()}\n")
        f.write("Hyperparameters:\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Optimizer: {args.optimizer.upper()}\n")
        f.write(f"  Dropout: {args.dropout}\n")
        f.write(f"  Weight Decay: {args.weight_decay}\n")
        f.write(f"  Momentum: {args.momentum if args.optimizer == 'sgd' else 'N/A'}\n")
        f.write(f"  Early Stopping: {args.early_stopping}\n")
        f.write(f"  Patience: {args.patience if args.early_stopping else 'N/A'}\n")
        f.write(f"  LR Scheduler: StepLR (step=15, gamma=0.1)\n")
        f.write("Training Logs:\n")
        
        # Write epoch-by-epoch logs
        for i in range(len(train_losses)):
            f.write(f"  Epoch {i+1}: Train Loss={train_losses[i]:.4f}, "
                   f"Train Acc={train_accs[i]:.2f}%, Val Loss={val_losses[i]:.4f}, "
                   f"Val Acc={val_accs[i]:.2f}%\n")
        
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})\n")
        f.write(f"Final Test Loss: {test_loss:.4f}\n")
        f.write(f"Final Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Total Training Time: {training_time/60:.2f} minutes\n")
        f.write(f"Regularization Used: {reg_used}\n")
        f.write(f"Plot: {plot_filename}\n")
        f.write("=============================\n")

def main():
    parser = argparse.ArgumentParser(description='Improved LeNet Training')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'], default='mnist',
                       help='Dataset to use (mnist or cifar10)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='sgd',
                       help='Optimizer to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate (0.0 to disable)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay for L2 regularization')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    
    args = parser.parse_args()
    
    print("Starting Improved LeNet Model Training...")
    print(f"Configuration: {args}")
    
    run_experiment(args)

if __name__ == '__main__':
    main()
