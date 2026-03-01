# 7-train-model.py
# Model Training: 1D CNN for EMG classification
# Part of the AlterEgo ML Pipeline
# UPDATED: Dynamic Channels, Dynamic Labels, N-Channel Adaptive

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import pickle
from torch.utils.data import TensorDataset, DataLoader
from utils.model import EMG_CNN  # Shared model (adaptive pooling, N-channel)

# ==========================================
# CONFIGURATION
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 10  # Early stopping: stop if no improvement for this many epochs

def train_model(model, loader, criterion, optimizer, epochs, save_path):
    print(f"\n🚀 Training for up to {epochs} epochs (early stopping after {PATIENCE} without improvement)...")

    best_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        acc = 100 * correct / total
        avg_loss = total_loss / len(loader)

        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))

        # Check for improvement
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
            patience_counter = 0
            marker = " ⭐ NEW BEST"
        else:
            patience_counter += 1
            marker = ""

        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: [{bar}] {acc:.1f}%{marker}")

        # Early stopping check
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping! No improvement for {PATIENCE} epochs.")
            print(f"   Best accuracy: {best_acc:.1f}%")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n✅ Restored best model (Acc: {best_acc:.1f}%)")

    return best_acc


# MAIN
# ==========================================
def main():
    print("╔════════════════════════════════════════╗")
    print("║      AlterEgo Model Trainer            ║")
    print("║      (Dynamic Architecture)            ║")
    print("╚════════════════════════════════════════╝")

    parser = argparse.ArgumentParser(description="Train EMG model.")
    parser.add_argument("--features", default="features", help="Directory with X.npy, y.npy")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--output", default="models", help="Output directory")
    args = parser.parse_args()

    # Load Data
    try:
        X = np.load(os.path.join(args.features, "X.npy"))
        y = np.load(os.path.join(args.features, "y.npy"))

        # Load Label Encoder to know class names
        with open(os.path.join(args.features, "label_encoder.pkl"), "rb") as f:
            le = pickle.load(f)

    except Exception as e:
        print(f"❌ Error loading features: {e}")
        return

    # Determine Architecture Parameters
    # X shape: (Samples, Timesteps, Features)
    num_input_features = X.shape[2]
    num_classes = len(np.unique(y))

    print(f"\n✅ Data Loaded:")
    print(f"   Samples: {len(X)}")
    print(f"   Input Channels: {num_input_features} (MFCCs * Physical Ch)")
    print(f"   Classes: {num_classes} ({list(le.classes_)})")

    # Tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = EMG_CNN(input_channels=num_input_features, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    model_path = os.path.join(args.output, "model.pth")
    train_model(model, loader, criterion, optimizer, args.epochs, model_path)

    # Save
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Save Model Weights
    model_path = os.path.join(args.output, "model.pth")
    torch.save(model.state_dict(), model_path)

    # Save Metadata needed for inference
    meta = {
        'input_channels': num_input_features,
        'num_classes': num_classes,
        'classes': list(le.classes_)
    }
    with open(os.path.join(args.output, "model_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"\n✅ Training complete!")
    print(f"   Model: {model_path}")
    print(f"   Meta:  {os.path.join(args.output, 'model_meta.pkl')}")

if __name__ == "__main__":
    main()
