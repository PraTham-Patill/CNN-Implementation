import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from datetime import datetime

class LeNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0, input_channels=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.AvgPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

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
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def plot_results(train_losses, val_losses, train_accs, val_accs, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')
    
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.set_title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(f'results/{filename}')
    plt.close()

def append_to_report(content):
    os.makedirs('results', exist_ok=True)
    with open('results/train_report.txt', 'a', encoding='utf-8') as f:
        f.write(content + '\n')

def get_dataset(dataset_name, batch_size):
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError("Dataset must be 'mnist' or 'cifar10'")
    
    # Split train into train and validation
    torch.manual_seed(42)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, test_loader, num_classes

def main():
    parser = argparse.ArgumentParser(description='Configurable LeNet Training')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='Dataset to use (mnist or cifar10)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer to use')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (0.0 to disable)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for L2 regularization')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loading
    train_loader, val_loader, test_loader, num_classes = get_dataset(args.dataset, args.batch_size)
    
    # Model setup
    input_channels = 1 if args.dataset == 'mnist' else 3
    model = LeNet(num_classes=num_classes, dropout_rate=args.dropout, input_channels=input_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    print(f"Model parameters: {count_parameters(model)}")
    
    # Training
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')
        
        # Early stopping
        if args.early_stopping:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), 'results/best_configurable_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model if early stopping was used
    if args.early_stopping:
        model.load_state_dict(torch.load('results/best_configurable_model.pth'))
    
    # Test evaluation
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f'Final Test Accuracy: {test_acc:.2f}%')
    
    # Save plot
    plot_filename = f'configurable_{args.dataset}_{args.optimizer}_results.png'
    plot_results(train_losses, val_losses, train_accs, val_accs, plot_filename)
    
    # Prepare regularization description
    regularizations = []
    if args.dropout > 0:
        regularizations.append(f"Dropout (rate = {args.dropout})")
    if args.weight_decay > 0:
        regularizations.append(f"L2 (weight_decay = {args.weight_decay})")
    if args.optimizer == 'sgd':
        regularizations.append(f"SGD with Momentum (momentum = {args.momentum})")
    if args.early_stopping:
        regularizations.append(f"Early Stopping (patience = {args.patience})")
    
    reg_used = ", ".join(regularizations) if regularizations else "None"
    
    # Append detailed report
    report_content = f"""=============================
Experiment: LeNet Configurable Model
Dataset: {args.dataset.upper()}
Hyperparameters:
  Learning Rate: {args.lr}
  Batch Size: {args.batch_size}
  Epochs: {len(train_losses)}
  Optimizer: {args.optimizer.upper()}
  Dropout: {args.dropout}
  Weight Decay: {args.weight_decay}
  Momentum: {args.momentum if args.optimizer == 'sgd' else 'N/A'}
  Early Stopping: {args.early_stopping}
  Patience: {args.patience if args.early_stopping else 'N/A'}
Training Logs:"""
    
    for i in range(len(train_losses)):
        report_content += f"\n  Epoch {i+1}: Train Loss={train_losses[i]:.4f}, Train Acc={train_accs[i]:.2f}%, Val Loss={val_losses[i]:.4f}, Val Acc={val_accs[i]:.2f}%"
    
    report_content += f"""
Final Test Accuracy: {test_acc:.2f}%
Regularization Used: {reg_used}
Plot: {plot_filename}
============================="""
    
    append_to_report(report_content)
    print("Training completed and results saved!")

if __name__ == "__main__":
    main()
