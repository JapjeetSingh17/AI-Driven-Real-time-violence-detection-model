import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

from dataset import VideoDataset
from model import ViolenceDetectionModel

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)
    
    return epoch_loss, acc, precision, recall, f1, cm

def train_model(data_dir, num_epochs=10, batch_size=8, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset & DataLoader
    full_dataset = VideoDataset(root_dir=data_dir, num_frames=16)
    
    if len(full_dataset) == 0:
        print("Dataset is empty. Please add videos to violence/ and non_violence/ folders.")
        return
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 2. Model, Loss, Optimizer
    model = ViolenceDetectionModel(num_classes=2, pretrained=True)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # 3. Fine-tuning strategy: First freeze backbone, train only classifier
    print("Phase 1: Training classifier head only (frozen backbone)")
    model.freeze_backbone()
    optimizer = optim.Adam(model.model.fc.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs // 2):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        train_loss = running_loss / len(train_dataset)
        val_loss, val_acc, val_p, val_r, val_f1, val_cm = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs//2} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - F1: {val_f1:.4f}")
        
    # Phase 2: Unfreeze backbone for full fine-tuning
    print("Phase 2: Full network fine-tuning")
    model.unfreeze_all()
    # Lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=learning_rate * 0.1)
    
    for epoch in range(num_epochs // 2, num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        train_loss = running_loss / len(train_dataset)
        val_loss, val_acc, val_p, val_r, val_f1, val_cm = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - F1: {val_f1:.4f}")

    print("\n--- Final Evaluation ---")
    print(f"Accuracy: {val_acc:.4f}")
    print(f"Precision: {val_p:.4f}")
    print(f"Recall: {val_r:.4f}")
    print(f"F1-Score: {val_f1:.4f}")
    print("Confusion Matrix:")
    print(val_cm)
    
    # Save the fine-tuned model
    torch.save(model.state_dict(), "violence_detection_model.pth")
    print("Model saved to violence_detection_model.pth")

if __name__ == "__main__":
    # Ensure dataset directory structure exists
    os.makedirs("dataset/violence", exist_ok=True)
    os.makedirs("dataset/non_violence", exist_ok=True)
    
    # Start training (you can adjust batch size and epochs)
    train_model(data_dir="dataset", num_epochs=10, batch_size=8, learning_rate=1e-4)
