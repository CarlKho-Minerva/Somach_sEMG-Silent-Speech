#!/usr/bin/env python3
"""
10-curriculum-eval.py
=====================
Curriculum Learning Evaluation: Train on ALL phases, evaluate EACH phase separately.
Shows how classification accuracy degrades from Overt → Whispered → Mouthing → Covert.

Usage:
    python3 10-curriculum-eval.py --features-dir features/

Expects directory structure:
    features/
    ├── Phase_1_Overt/       X.npy, y.npy, label_encoder.pkl
    ├── Phase_2_Whispered/   X.npy, y.npy, label_encoder.pkl
    ├── Phase_3_Mouthing/    X.npy, y.npy, label_encoder.pkl
    └── Phase_6_Covert/      X.npy, y.npy, label_encoder.pkl

Output:
    - Per-phase confusion matrix
    - Accuracy degradation summary table
    - Combined overall confusion matrix
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import pickle
import sys
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from utils.model import EMG_CNN

# ==========================================
# CONFIGURATION
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 10
TEST_SPLIT = 0.2

# ANSI Colors
C_RESET  = "\033[0m"
C_ORANGE = "\033[38;5;208m"
C_GREY   = "\033[38;5;240m"
C_WHITE  = "\033[37m"
C_RED    = "\033[31m"
C_GREEN  = "\033[32m"
C_BOLD   = "\033[1m"
C_CYAN   = "\033[36m"
C_YELLOW = "\033[33m"

# Phase display order (curriculum order)
PHASE_ORDER = [
    "Phase_1_Overt",
    "Phase_2_Whispered",
    "Phase_3_Mouthing",
    "Phase_5_Exaggerated",
    "Phase_6_Covert",
]

PHASE_LABELS = {
    "Phase_1_Overt":       "Overt (Aloud)",
    "Phase_2_Whispered":   "Whispered",
    "Phase_3_Mouthing":    "Mouthing",
    "Phase_5_Exaggerated": "Exaggerated",
    "Phase_6_Covert":      "Covert (Subvocal)",
}

# ==========================================
# LOADING
# ==========================================
def discover_phases(features_dir):
    """Find all Phase_* subdirectories that contain X.npy and y.npy."""
    phases = {}
    for entry in sorted(os.listdir(features_dir)):
        subdir = os.path.join(features_dir, entry)
        if not os.path.isdir(subdir):
            continue
        if not entry.startswith("Phase_"):
            continue
        x_path = os.path.join(subdir, "X.npy")
        y_path = os.path.join(subdir, "y.npy")
        le_path = os.path.join(subdir, "label_encoder.pkl")
        if os.path.exists(x_path) and os.path.exists(y_path):
            phases[entry] = {
                "X": np.load(x_path),
                "y": np.load(y_path),
                "le": None,
            }
            if os.path.exists(le_path):
                with open(le_path, "rb") as f:
                    phases[entry]["le"] = pickle.load(f)
            print(f"   {C_GREEN}✓{C_RESET} {entry}: {len(phases[entry]['X'])} samples")
        else:
            print(f"   {C_YELLOW}⚠{C_RESET} {entry}: missing X.npy or y.npy, skipping")

    return phases


def unify_labels(phases):
    """
    Ensure all phases use the same label encoding.
    Returns unified label list and re-encoded phase data.
    """
    # Collect all unique class names across phases
    all_classes = set()
    for name, data in phases.items():
        if data["le"] is not None:
            all_classes.update(data["le"].classes_)
        else:
            all_classes.update(str(v) for v in np.unique(data["y"]))

    all_classes = sorted(all_classes)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    # Re-encode each phase
    for name, data in phases.items():
        if data["le"] is not None:
            # Map original indices → class names → unified indices
            original_classes = list(data["le"].classes_)
            new_y = np.array([class_to_idx[original_classes[v]] for v in data["y"]])
            data["y"] = new_y
        # If no label encoder, assume y values are already consistent

    return all_classes, phases


# ==========================================
# TRAINING
# ==========================================
def train_model(X_train, y_train, input_channels, num_classes):
    """Train a model on the given data, return the trained model."""
    model = EMG_CNN(input_channels=input_channels, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

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
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
            patience_counter = 0
            marker = " ⭐"
        else:
            patience_counter += 1
            marker = ""

        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"   Epoch {epoch+1:3d}/{EPOCHS} | [{bar}] {acc:.1f}%{marker}", end="\r")

        if patience_counter >= PATIENCE:
            break

    print(f"\n   {C_GREEN}Best training accuracy: {best_acc:.1f}%{C_RESET}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ==========================================
# EVALUATION
# ==========================================
def evaluate_phase(model, X, y, labels, phase_name):
    """Run inference on one phase's data, return accuracy and confusion matrix."""
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X))
        _, preds = torch.max(outputs, 1)

    y_pred = preds.numpy()
    y_true = y

    # Filter to labels that actually appear
    present = sorted(list(set(y_true) | set(y_pred)))
    present_labels = [labels[i] for i in present if i < len(labels)]

    cm = confusion_matrix(y_true, y_pred, labels=present)
    accuracy = np.trace(cm) / np.sum(cm) * 100 if np.sum(cm) > 0 else 0

    return accuracy, cm, present_labels, y_true, y_pred


def print_phase_matrix(cm, labels, phase_name, accuracy):
    """Print a single phase's confusion matrix."""
    display = PHASE_LABELS.get(phase_name, phase_name)
    print(f"\n{C_BOLD}{C_ORANGE}{'─'*60}{C_RESET}")
    print(f"{C_BOLD}{C_ORANGE}  {display}  —  Accuracy: {accuracy:.1f}%{C_RESET}")
    print(f"{C_ORANGE}{'─'*60}{C_RESET}")

    # Header
    header = "          " + "  ".join([f"{l[:5]:>5}" for l in labels])
    print(f"{C_GREY}{header}{C_RESET}")
    print(f"{C_GREY}{'─' * len(header)}{C_RESET}")

    for i, row in enumerate(cm):
        row_str = f"  {labels[i][:5]:>5} │"
        for j, val in enumerate(row):
            if i == j:
                row_str += f" {C_GREEN}{val:5d}{C_RESET}"
            elif val > 0:
                row_str += f" {C_RED}{val:5d}{C_RESET}"
            else:
                row_str += f" {val:5d}"
        print(row_str)


def print_degradation_table(results):
    """Print the summary table showing accuracy drop across phases."""
    print(f"\n\n{C_BOLD}{'═'*60}{C_RESET}")
    print(f"{C_BOLD}{C_ORANGE}  CURRICULUM DEGRADATION SUMMARY{C_RESET}")
    print(f"{C_BOLD}{'═'*60}{C_RESET}")
    print(f"  {'Phase':<25} {'Accuracy':>10} {'Samples':>10}  {'Bar'}")
    print(f"  {'─'*25} {'─'*10} {'─'*10}  {'─'*22}")

    for phase_name, acc, n_samples in results:
        display = PHASE_LABELS.get(phase_name, phase_name)
        bar_len = int(acc / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        if acc >= 80:
            c = C_GREEN
        elif acc >= 60:
            c = C_YELLOW
        else:
            c = C_RED

        print(f"  {display:<25} {c}{acc:>9.1f}%{C_RESET} {n_samples:>10}  {c}{bar}{C_RESET}")

    # Delta
    if len(results) >= 2:
        drop = results[0][1] - results[-1][1]
        print(f"\n  {C_CYAN}Δ (Overt → weakest): {C_BOLD}{drop:+.1f}%{C_RESET}")

    print(f"{'═'*60}\n")


# ==========================================
# MAIN
# ==========================================
def main():
    print(f"{C_GREY}╔{'═'*58}╗{C_RESET}")
    print(f"{C_GREY}║ {C_ORANGE}{C_BOLD}{'CURRICULUM LEARNING EVALUATION':^56}{C_RESET} {C_GREY}║{C_RESET}")
    print(f"{C_GREY}║ {C_GREY}{'Train on ALL phases • Evaluate EACH phase':^56}{C_RESET} {C_GREY}║{C_RESET}")
    print(f"{C_GREY}╚{'═'*58}╝{C_RESET}")

    parser = argparse.ArgumentParser(description="Curriculum learning cross-phase evaluation.")
    parser.add_argument("--features-dir", default="features",
                        help="Directory containing Phase_* subdirs with X.npy, y.npy")
    parser.add_argument("--save-model", default=None,
                        help="Optional: save trained model to this path")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    # 1. Discover phases
    print(f"\n{C_WHITE}📂 Scanning {args.features_dir} for phases...{C_RESET}")
    phases = discover_phases(args.features_dir)

    if len(phases) == 0:
        print(f"\n{C_RED}❌ No phases found. Run 6-extract-features.py --per-phase first.{C_RESET}")
        print(f"{C_GREY}   Expected: {args.features_dir}/Phase_1_Overt/X.npy etc.{C_RESET}")
        return

    if len(phases) == 1:
        print(f"\n{C_YELLOW}⚠️  Only 1 phase found. Need at least 2 for curriculum comparison.{C_RESET}")
        print(f"{C_GREY}   Record more phases and extract features with --per-phase.{C_RESET}")
        return

    # 2. Unify labels across phases
    print(f"\n{C_WHITE}🏷️  Unifying labels across {len(phases)} phases...{C_RESET}")
    all_classes, phases = unify_labels(phases)
    print(f"   Classes: {all_classes}")

    # 3. Combine ALL data for training
    print(f"\n{C_WHITE}🔗 Combining all phases for training...{C_RESET}")
    X_all = np.concatenate([d["X"] for d in phases.values()])
    y_all = np.concatenate([d["y"] for d in phases.values()])
    print(f"   Total: {len(X_all)} samples, {X_all.shape[2]} features")

    input_channels = X_all.shape[2]
    num_classes = len(all_classes)

    # 4. Train on combined data
    print(f"\n{C_WHITE}🚀 Training unified model on ALL {len(X_all)} samples...{C_RESET}")
    model = train_model(X_all, y_all, input_channels, num_classes)

    # 5. Evaluate each phase separately
    print(f"\n{C_WHITE}📊 Evaluating per-phase...{C_RESET}")
    results = []
    all_y_true = []
    all_y_pred = []

    # Sort by curriculum order
    phase_names = [p for p in PHASE_ORDER if p in phases]
    # Add any phases not in PHASE_ORDER
    for p in phases:
        if p not in phase_names:
            phase_names.append(p)

    for phase_name in phase_names:
        data = phases[phase_name]
        acc, cm, present_labels, y_true, y_pred = evaluate_phase(
            model, data["X"], data["y"], all_classes, phase_name
        )
        print_phase_matrix(cm, present_labels, phase_name, acc)
        results.append((phase_name, acc, len(data["X"])))
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    # 6. Degradation summary
    print_degradation_table(results)

    # 7. Combined confusion matrix
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    combined_cm = confusion_matrix(all_y_true, all_y_pred)
    present = sorted(list(set(all_y_true) | set(all_y_pred)))
    combined_labels = [all_classes[i] for i in present if i < len(all_classes)]

    print(f"{C_BOLD}COMBINED CONFUSION MATRIX (all phases):{C_RESET}")
    print_phase_matrix(combined_cm, combined_labels, "ALL PHASES COMBINED",
                       np.trace(combined_cm) / np.sum(combined_cm) * 100)

    print(f"\n{C_WHITE}📊 Full Classification Report (combined):{C_RESET}")
    print(classification_report(all_y_true, all_y_pred,
                                target_names=combined_labels, zero_division=0))

    # 8. Optional: save model
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        meta = {
            "input_channels": input_channels,
            "num_classes": num_classes,
            "classes": all_classes,
        }
        meta_path = args.save_model.replace(".pth", "_meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        print(f"\n{C_GREEN}💾 Model saved: {args.save_model}{C_RESET}")
        print(f"{C_GREEN}   Meta saved: {meta_path}{C_RESET}")


if __name__ == "__main__":
    main()
