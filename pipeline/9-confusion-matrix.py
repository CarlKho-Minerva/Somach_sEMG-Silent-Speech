# 9-confusion-matrix.py
# Model Evaluation: Generate confusion matrix to diagnose classification errors
# Part of the AlterEgo ML Pipeline

import torch
import numpy as np
import os
import argparse
from sklearn.metrics import confusion_matrix, classification_report
from utils.model import EMG_CNN  # Shared model (adaptive pooling, N-channel)


# ==========================================
# VISUALIZATION
# ==========================================
def print_confusion_matrix(cm, labels):
    """
    Print a nicely formatted confusion matrix to the terminal.

    Args:
        cm: 2D numpy array (confusion matrix)
        labels: List of class label strings
    """
    n = len(labels)

    # Header
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("Rows = True Label | Columns = Predicted")
    print("=" * 60)

    # Column headers
    header = "      " + "  ".join([f"{l[:4]:>4}" for l in labels])
    print(header)
    print("-" * len(header))

    # Rows
    for i, row in enumerate(cm):
        row_str = f"{labels[i][:4]:>4} |"
        for j, val in enumerate(row):
            if i == j:
                # Highlight diagonal (correct predictions)
                row_str += f" \033[92m{val:4d}\033[0m"  # Green
            elif val > 0:
                # Highlight errors
                row_str += f" \033[91m{val:4d}\033[0m"  # Red
            else:
                row_str += f" {val:4d}"
        print(row_str)

    print("-" * len(header))


def find_problem_pairs(cm, labels, threshold=5):
    """
    Find pairs of classes that are frequently confused.

    Args:
        cm: Confusion matrix
        labels: Class labels
        threshold: Minimum errors to report
    """
    print("\n⚠️  PROBLEM PAIRS (most confused):")
    problems = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] >= threshold:
                problems.append((labels[i], labels[j], cm[i, j]))

    if not problems:
        print("   None! All classes are well-separated.")
        return

    problems.sort(key=lambda x: -x[2])
    for true_label, pred_label, count in problems[:5]:
        print(f"   {true_label} → {pred_label}: {count} errors")
        print(f"      💡 Record more samples of '{true_label}' with clear articulation.")


# ==========================================
# MAIN
# ==========================================
def main():
    print("╔════════════════════════════════════════╗")
    print("║      AlterEgo Model Evaluator          ║")
    print("╚════════════════════════════════════════╝")

    parser = argparse.ArgumentParser(description="Generate confusion matrix for trained model.")
    parser.add_argument("--model", default="models/model.pth", help="Path to trained model (.pth)")
    parser.add_argument("--meta", default="models/model_meta.pkl", help="Path to model metadata (.pkl)")
    parser.add_argument("--features", default="features/", help="Directory containing X.npy, y.npy")
    args = parser.parse_args()

    # 1. Load Model Metadata (for dynamic architecture)
    print(f"\n📂 Loading model metadata from {args.meta}...")
    try:
        import pickle
        with open(args.meta, "rb") as f:
            meta = pickle.load(f)
        input_channels = meta['input_channels']
        num_classes = meta['num_classes']
        labels = meta['classes']
        print(f"   Input channels: {input_channels}")
        print(f"   Classes: {labels}")
    except FileNotFoundError:
        print("❌ Model metadata not found. Run 7-train-model.py first.")
        return

    # 2. Load Data
    print(f"\n📂 Loading test data from {args.features}...")
    try:
        X = np.load(os.path.join(args.features, "X.npy"))
        y = np.load(os.path.join(args.features, "y.npy"))
    except FileNotFoundError:
        print("❌ Features not found. Run 6-extract-features.py first.")
        return

    print(f"   Samples: {len(X)}")

    # 3. Load Model
    print(f"📂 Loading model from {args.model}...")
    try:
        model = EMG_CNN(input_channels=input_channels, num_classes=num_classes)
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
        model.eval()
    except FileNotFoundError:
        print("❌ Model not found.")
        return

    # 4. Run Predictions
    print("🔮 Running predictions...")
    X_tensor = torch.FloatTensor(X)

    with torch.no_grad():
        outputs = model(X_tensor)
        _, predictions = torch.max(outputs, 1)

    y_pred = predictions.numpy()
    y_true = y

    # 5. Generate Confusion Matrix
    # Filter to only include classes present in the data
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    present_labels = [labels[i] for i in unique_labels if i < len(labels)]

    cm = confusion_matrix(y_true, y_pred)

    # 6. Display Results
    print_confusion_matrix(cm, present_labels)

    # 7. Classification Report
    print("\n📊 CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=present_labels, zero_division=0))

    # 8. Actionable Insights
    find_problem_pairs(cm, present_labels)

    # 9. Summary
    accuracy = np.trace(cm) / np.sum(cm) * 100
    print(f"\n✅ Overall Accuracy: {accuracy:.1f}%")

    if accuracy < 60:
        print("   ⚠️  Low accuracy. Collect more training data.")
    elif accuracy < 80:
        print("   💡 Good start! Focus on problem pairs above.")
    else:
        print("   🎉 Excellent! Ready for transfer learning.")


if __name__ == "__main__":
    main()
