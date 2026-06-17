import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from isl_dataset import ISLDataset
from model import SignLSTM
import os
import gc  # Garbage Collector to free up RAM

# 1. Setup Device
device = torch.device("cpu")  # Using CPU as discussed
print(f"Using device: {device}")

# 2. Load and Split Dataset
DATASET_PATH = r"D:\signtosound\data\INDIAN_SIGN_LANGUAGE_NUMPY_ARRAY_SKELETAL_POINT_DATASET\data"
full_dataset = ISLDataset(DATASET_PATH)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# CRITICAL: Batch size reduced to 8 to prevent RAM freeze
# num_workers=0 is necessary for Windows stability
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

# 3. Initialize Model, Loss, and Optimizer
model = SignLSTM(num_classes=len(full_dataset.labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
num_epochs = 50 
print(f"Starting Training for {num_epochs} epochs...")
print("Tip: Close Google Chrome and other apps to free up RAM.")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    
    for i, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Update progress every 20 batches
        if i % 20 == 0:
            print(f"Epoch {epoch+1:2d} | Batch {i:3d}/{len(train_loader)} | Loss: {loss.item():.4f}", end='\r')

    # 5. Validation Phase (Every 2nd Epoch to save CPU/RAM)
    if (epoch + 1) % 2 == 0 or epoch == 0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                _, predicted = torch.max(outputs.data, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        
        accuracy = 100 * correct / total
        print(f"\n--- Epoch {epoch+1} Results: Loss: {train_loss/len(train_loader):.4f} | Val Acc: {accuracy:.2f}% ---")
        
        # SAVE PROGRESS: If the computer crashes later, you have the latest model
        torch.save(model.state_dict(), "isl_model.pth")
        
    # Clear memory cache after each epoch
    gc.collect()

print("\n✅ Training Complete. Final model saved as isl_model.pth")
