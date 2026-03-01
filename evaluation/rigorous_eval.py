#!/usr/bin/env python3
"""
rigorous_eval.py — Comprehensive ML Evaluation for Paper 1 (Study B)
=====================================================================
Runs 7 evaluation protocols on the Study B sEMG dataset and generates
a results_summary.md with all findings.

Usage:
    python rigorous_eval.py

Requires: torch, numpy, scikit-learn, librosa, scipy, pandas
"""

import os
import sys

# Workaround: numba caching fails on Python 3.14
# Must be set BEFORE importing librosa (which imports numba)
os.environ["NUMBA_DISABLE_JIT"] = "1"

import glob
import time
import warnings
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
# NOTE: librosa is broken on Python 3.14 due to numba.
# Using pure scipy/numpy MFCC implementation instead.
import scipy.signal
import scipy.fft
from datetime import datetime

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ===========================================================
# PATHS (relative to project root)
# ===========================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INSTRUCTABLES = os.path.join(os.path.dirname(PROJECT_ROOT),
    "020526_INSTRUCTABLES_GuideForMeEveryone", "code")

# Study B full 5-phase (1,500 samples, Feb 11)
STUDYB_RAW = os.path.join(INSTRUCTABLES, "python", "data_collection", "carl-subvocalization")
# Supplemental covert sessions
COVERT_A = os.path.join(INSTRUCTABLES, "python", "data_collection", "carl", "Phase_6_Covert")
COVERT_B = os.path.join(INSTRUCTABLES, "python", "data_collection", "carl_022526_sessionb", "Phase_6_Covert")
COVERT_C = os.path.join(INSTRUCTABLES, "python", "data_collection", "carl_sessionc2", "Phase_6_Covert")

# ===========================================================
# SIGNAL PROCESSING & FEATURE EXTRACTION (Pure scipy/numpy)
# ===========================================================
SAMPLE_RATE = 250
LOW_CUT = 1.0
HIGH_CUT = 50.0
NOTCH_FREQ = 60.0
N_MFCC = 13
N_FFT = 128
HOP_LENGTH = 25
N_MELS = 26
TARGET_TIMESTEPS = 100


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a


def notch_filter(freq, fs, q=30):
    b, a = scipy.signal.iirnotch(freq / (0.5 * fs), q)
    return b, a


def preprocess_signal(data, fs=SAMPLE_RATE):
    # Guard: signal must be longer than filter padlen (3 * max(len(b), len(a)))
    min_len = 28  # safe minimum for 4th-order Butterworth
    if len(data) < min_len:
        vmin, vmax = np.min(data), np.max(data)
        if vmax - vmin == 0:
            return data.astype(float)
        return ((data - vmin) / (vmax - vmin)).astype(float)

    b_band, a_band = butter_bandpass(LOW_CUT, HIGH_CUT, fs)
    y = scipy.signal.filtfilt(b_band, a_band, data)
    b_notch, a_notch = notch_filter(NOTCH_FREQ, fs)
    y = scipy.signal.filtfilt(b_notch, a_notch, y)
    vmin, vmax = np.min(y), np.max(y)
    if vmax - vmin == 0:
        return y
    return (y - vmin) / (vmax - vmin)


def _mel_filterbank(sr, n_fft, n_mels):
    """Create a mel filterbank (pure numpy, no librosa)."""
    fmin, fmax = 0.0, sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        for j in range(left, center):
            if center != left:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                fb[i, j] = (right - j) / (right - center)
    return fb


def extract_mfcc(signal, sr=SAMPLE_RATE):
    """Extract MFCCs using pure scipy/numpy (no librosa dependency)."""
    signal = signal.astype(float)
    fb = _mel_filterbank(sr, N_FFT, N_MELS)

    # STFT
    frames = []
    for start in range(0, len(signal) - N_FFT + 1, HOP_LENGTH):
        frame = signal[start:start + N_FFT]
        frame = frame * np.hanning(N_FFT)
        spectrum = np.abs(np.fft.rfft(frame)) ** 2
        frames.append(spectrum)

    if not frames:
        frames = [np.zeros(N_FFT // 2 + 1)]

    power_spec = np.array(frames)  # (time, freq)
    mel_spec = np.dot(power_spec, fb.T)  # (time, n_mels)
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec)

    # DCT-II to get MFCCs
    mfcc = scipy.fft.dct(log_mel, type=2, axis=1, norm='ortho')[:, :N_MFCC]
    return mfcc  # (time, n_mfcc)


def pad_or_truncate(features, target_len=TARGET_TIMESTEPS):
    if len(features) < target_len:
        features = np.pad(features, ((0, target_len - len(features)), (0, 0)), mode='constant')
    else:
        features = features[:target_len, :]
    return features


def load_csv_to_features(filepath):
    """Load a single CSV, preprocess, extract MFCCs, return feature matrix."""
    df = pd.read_csv(filepath)
    ch_cols = [c for c in df.columns if c.startswith('CH')]
    if not ch_cols:
        return None

    all_feats = []
    for ch in ch_cols:
        sig = preprocess_signal(df[ch].values)
        feats = extract_mfcc(sig)
        feats = pad_or_truncate(feats)
        all_feats.append(feats)
    return np.hstack(all_feats)  # (time, n_mfcc * n_channels)


def load_dataset_from_dir(directory, phase_label=None):
    """Load all CSVs from a directory. Returns X, y_labels, phase_labels."""
    X_list, y_list, p_list = [], [], []
    csv_files = sorted(glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True))

    for f in csv_files:
        feats = load_csv_to_features(f)
        if feats is None:
            continue

        # Label from filename
        label = os.path.basename(f).split('_')[0].upper()
        X_list.append(feats)
        y_list.append(label)

        # Phase from directory
        if phase_label:
            p_list.append(phase_label)
        else:
            parts = f.replace("\\", "/").split("/")
            phase = "Unknown"
            for part in parts:
                if part.startswith("Phase_"):
                    phase = part
                    break
            p_list.append(phase)

    if not X_list:
        return np.array([]), np.array([]), np.array([])

    return np.array(X_list), np.array(y_list), np.array(p_list)


# ===========================================================
# MODEL
# ===========================================================
class EMG_CNN(nn.Module):
    def __init__(self, input_channels=26, num_classes=6, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.adaptive_pool(x).squeeze(-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def train_and_evaluate(X_train, y_train, X_test, y_test, num_classes,
                       lr=0.001, dropout=0.5, weight_decay=0, hidden_dim=128,
                       epochs=100, patience=10, batch_size=32, verbose=False):
    """Train a model, return test accuracy and predictions."""
    input_channels = X_train.shape[2]

    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.LongTensor(y_train)
    X_te = torch.FloatTensor(X_test)
    y_te = torch.LongTensor(y_test)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    model = EMG_CNN(input_channels, num_classes, hidden_dim, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    best_state = None
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(out, 1)
            total += by.size(0)
            correct += (pred == by).sum().item()

        train_acc = correct / total
        if train_acc > best_acc:
            best_acc = train_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    with torch.no_grad():
        out = model(X_te)
        probs = torch.softmax(out, dim=1)
        _, preds = torch.max(out, 1)

    y_pred = preds.numpy()
    y_prob = probs.numpy()
    acc = accuracy_score(y_te.numpy(), y_pred)

    return acc, y_pred, y_prob, best_acc


# ===========================================================
# EVALUATION PROTOCOLS
# ===========================================================
def run_kfold_cv(X, y, phases, le, n_splits=5, **hparams):
    """Protocol 1: Stratified K-Fold CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accs = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        acc, y_pred, _, train_acc = train_and_evaluate(
            X[train_idx], y[train_idx], X[test_idx], y[test_idx],
            num_classes=len(le.classes_), **hparams
        )
        fold_accs.append(acc)
        all_y_true.extend(y[test_idx])
        all_y_pred.extend(y_pred)

    return {
        "fold_accs": fold_accs,
        "mean": np.mean(fold_accs),
        "std": np.std(fold_accs),
        "all_y_true": np.array(all_y_true),
        "all_y_pred": np.array(all_y_pred),
    }


def run_lopo(X, y, phases, le, **hparams):
    """Protocol 2: Leave-One-Phase-Out."""
    unique_phases = sorted(set(phases))
    results = {}

    for held_out in unique_phases:
        test_mask = phases == held_out
        train_mask = ~test_mask

        if np.sum(test_mask) == 0 or np.sum(train_mask) == 0:
            continue

        acc, y_pred, _, _ = train_and_evaluate(
            X[train_mask], y[train_mask], X[test_mask], y[test_mask],
            num_classes=len(le.classes_), **hparams
        )
        results[held_out] = {"acc": acc, "n_test": int(np.sum(test_mask))}

    return results


def run_cross_session(X_train, y_train, X_test, y_test, le, **hparams):
    """Protocol 3: Cross-session temporal split."""
    acc, y_pred, y_prob, train_acc = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        num_classes=len(le.classes_), **hparams
    )
    return {
        "test_acc": acc,
        "train_acc": train_acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "y_pred": y_pred,
    }


def run_hyperparam_sweep(X, y, le):
    """Protocol 5: Grid search for best generalizing model."""
    param_grid = {
        "dropout": [0.3, 0.5, 0.7],
        "weight_decay": [0, 1e-4, 1e-3, 1e-2],
        "hidden_dim": [64, 128],
        "lr": [0.0005, 0.001, 0.005],
    }

    keys = sorted(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    total = len(combos)

    results = []
    best_acc = 0
    best_params = None

    print(f"\n🔍 Hyperparameter sweep: {total} configurations × 3-fold CV")

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        try:
            cv = run_kfold_cv(X, y, None, le, n_splits=3, **params)
            mean_acc = cv["mean"]
        except Exception:
            mean_acc = 0.0

        results.append({**params, "cv_acc": mean_acc})

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_params = params

        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"   [{i+1}/{total}] best so far: {best_acc:.1%} {best_params}")

    return results, best_params, best_acc


def run_confidence_gating(y_true, y_prob, thresholds=None):
    """Protocol 6: Confidence gating analysis."""
    if thresholds is None:
        thresholds = np.arange(0.50, 0.96, 0.05)

    results = []
    for theta in thresholds:
        max_probs = np.max(y_prob, axis=1)
        accepted = max_probs >= theta
        n_accepted = np.sum(accepted)
        if n_accepted == 0:
            results.append({"theta": theta, "acc": 0.0, "coverage": 0.0, "n_accepted": 0})
            continue
        acc = accuracy_score(y_true[accepted], np.argmax(y_prob[accepted], axis=1))
        coverage = n_accepted / len(y_true)
        results.append({"theta": theta, "acc": acc, "coverage": coverage, "n_accepted": int(n_accepted)})

    return results


# ===========================================================
# OUTPUT
# ===========================================================
def generate_markdown(all_results, output_path):
    """Write results_summary.md."""
    lines = [
        "# Rigorous ML Evaluation — Study B Results",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Script:** `rigorous_eval.py`",
        "",
    ]

    # Protocol 1: K-Fold CV
    if "kfold" in all_results:
        r = all_results["kfold"]
        lines.append("## 1. Intra-Session 5-Fold Stratified CV (1,500 samples, all phases)")
        lines.append("")
        lines.append(f"**Overall: {r['mean']:.1%} ± {r['std']:.1%}**")
        lines.append("")
        lines.append("| Fold | Accuracy |")
        lines.append("|------|----------|")
        for i, a in enumerate(r["fold_accs"]):
            lines.append(f"| {i+1} | {a:.1%} |")
        lines.append("")

    # Protocol 2: LOPO
    if "lopo" in all_results:
        r = all_results["lopo"]
        lines.append("## 2. Leave-One-Phase-Out (LOPO)")
        lines.append("")
        lines.append("| Held-Out Phase | Test Acc | n |")
        lines.append("|----------------|----------|---|")
        for phase, data in sorted(r.items()):
            lines.append(f"| {phase} | {data['acc']:.1%} | {data['n_test']} |")
        mean_lopo = np.mean([v["acc"] for v in r.values()])
        lines.append(f"\n**Mean LOPO: {mean_lopo:.1%}**\n")

    # Protocol 3: Cross-session
    if "cross_session" in all_results:
        r = all_results["cross_session"]
        lines.append("## 3. Cross-Session Temporal Split (Train Feb 11, Test Feb 25)")
        lines.append("")
        lines.append(f"- Train samples: {r['n_train']}")
        lines.append(f"- Test samples: {r['n_test']}")
        lines.append(f"- Train accuracy: {r['train_acc']:.1%}")
        lines.append(f"- **Test accuracy: {r['test_acc']:.1%}**")
        lines.append("")

    # Protocol 4: Combined multi-session CV
    if "combined_cv" in all_results:
        r = all_results["combined_cv"]
        lines.append("## 4. Combined Multi-Session 5-Fold CV (Covert Only)")
        lines.append("")
        lines.append(f"**Overall: {r['mean']:.1%} ± {r['std']:.1%}**")
        lines.append("")
        lines.append("| Fold | Accuracy |")
        lines.append("|------|----------|")
        for i, a in enumerate(r["fold_accs"]):
            lines.append(f"| {i+1} | {a:.1%} |")
        lines.append("")

    # Protocol 5: Hyperparameter sweep
    if "hparam_sweep" in all_results:
        r = all_results["hparam_sweep"]
        lines.append("## 5. Hyperparameter Sweep")
        lines.append("")
        lines.append(f"**Best config: {r['best_acc']:.1%}**")
        lines.append("")
        bp = r["best_params"]
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for k, v in sorted(bp.items()):
            lines.append(f"| {k} | {v} |")
        lines.append("")
        lines.append("### Top 10 Configurations")
        lines.append("")
        lines.append("| dropout | weight_decay | hidden_dim | lr | CV Acc |")
        lines.append("|---------|-------------|------------|------|--------|")
        top10 = sorted(r["all_results"], key=lambda x: x["cv_acc"], reverse=True)[:10]
        for cfg in top10:
            lines.append(f"| {cfg['dropout']} | {cfg['weight_decay']} | {cfg['hidden_dim']} | {cfg['lr']} | {cfg['cv_acc']:.1%} |")
        lines.append("")

    # Protocol 6: Confidence gating
    if "confidence_gating" in all_results:
        r = all_results["confidence_gating"]
        lines.append("## 6. Confidence Gating (Best Model, Held-Out Predictions)")
        lines.append("")
        lines.append("| θ | Accuracy | Coverage | Accepted |")
        lines.append("|---|----------|----------|----------|")
        for entry in r:
            lines.append(f"| {entry['theta']:.2f} | {entry['acc']:.1%} | {entry['coverage']:.1%} | {entry['n_accepted']} |")
        lines.append("")

    # Protocol 7: Summary comparison table
    lines.append("## 7. Summary Comparison")
    lines.append("")
    lines.append("| Protocol | Accuracy | Notes |")
    lines.append("|----------|----------|-------|")

    if "kfold" in all_results:
        r = all_results["kfold"]
        lines.append(f"| 5-Fold CV (all phases) | {r['mean']:.1%} ± {r['std']:.1%} | Intra-session, stratified |")

    if "lopo" in all_results:
        mean_lopo = np.mean([v["acc"] for v in all_results["lopo"].values()])
        lines.append(f"| Leave-One-Phase-Out | {mean_lopo:.1%} | Cross-intensity generalization |")

    if "cross_session" in all_results:
        lines.append(f"| Cross-session (A→B+C) | {all_results['cross_session']['test_acc']:.1%} | Electrode shift test |")

    if "combined_cv" in all_results:
        r = all_results["combined_cv"]
        lines.append(f"| Combined multi-session CV | {r['mean']:.1%} ± {r['std']:.1%} | Covert only, 3 positions |")

    if "hparam_sweep" in all_results:
        lines.append(f"| Best hyperparameters | {all_results['hparam_sweep']['best_acc']:.1%} | 3-fold CV, grid search |")

    lines.append(f"| Original (train-set) | 99.7% | No held-out, as reported in paper |")
    lines.append("")

    # Past experiments reference
    lines.append("## Prior Experiments (from `022426_AccuracyExperiments/`)")
    lines.append("")
    lines.append("| Experiment | Result | Verdict |")
    lines.append("|------------|--------|---------|")
    lines.append("| SpecAugment | 59.4% | ❌ Hurts at n=932 |")
    lines.append("| Feature Importance | 100% onset | ⚠️ Only first 80ms matters |")
    lines.append("| Permutation Integrity | All pass | ✅ Real signal, no artifacts |")
    lines.append("| Raw Signal CNN | 62.0% | = MFCCs, less stable |")
    lines.append("| Phase-Weighted Curriculum | 43.3% | ❌ Worse |")
    lines.append("| Onset Masking (middle-only) | 17.6% | ❌ Chance — hardware ceiling |")
    lines.append("| Multi-session combined | 49.7% | Recovers from 21.9% cross |")
    lines.append("| 4-class (drop LEFT) | 57.0% | Meaningful improvement |")
    lines.append("| Confidence gate @60% | 77.9% | Strong selective accuracy |")
    lines.append("| CNN+RF ensemble + gate @70% | 100% (12/306) | Too selective |")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n📄 Results saved to {output_path}")


# ===========================================================
# MAIN
# ===========================================================
def main():
    print("=" * 60)
    print("  RIGOROUS ML EVALUATION — Study B")
    print("  No Stones Left Unturned")
    print("=" * 60)

    t0 = time.time()
    all_results = {}

    # ---- Load Study B full 5-phase dataset ----
    print("\n📂 Loading Study B 5-phase dataset...")
    X_full, y_labels, phases = load_dataset_from_dir(STUDYB_RAW)
    print(f"   Loaded: {len(X_full)} samples, {len(set(y_labels))} classes, {len(set(phases))} phases")

    le = LabelEncoder()
    y_full = le.fit_transform(y_labels)
    print(f"   Classes: {list(le.classes_)}")
    print(f"   Phases: {sorted(set(phases))}")
    print(f"   Feature shape: {X_full.shape}")

    # ==========================
    # PROTOCOL 1: 5-Fold CV
    # ==========================
    print("\n" + "=" * 60)
    print("  PROTOCOL 1: Intra-Session 5-Fold Stratified CV")
    print("=" * 60)

    kfold_result = run_kfold_cv(X_full, y_full, phases, le)
    print(f"\n   Fold accuracies: {[f'{a:.1%}' for a in kfold_result['fold_accs']]}")
    print(f"   ✅ Mean: {kfold_result['mean']:.1%} ± {kfold_result['std']:.1%}")
    all_results["kfold"] = kfold_result

    # ==========================
    # PROTOCOL 2: Leave-One-Phase-Out
    # ==========================
    print("\n" + "=" * 60)
    print("  PROTOCOL 2: Leave-One-Phase-Out (LOPO)")
    print("=" * 60)

    lopo_result = run_lopo(X_full, y_full, phases, le)
    for phase, data in sorted(lopo_result.items()):
        print(f"   {phase}: {data['acc']:.1%} (n={data['n_test']})")
    mean_lopo = np.mean([v["acc"] for v in lopo_result.values()])
    print(f"   ✅ Mean LOPO: {mean_lopo:.1%}")
    all_results["lopo"] = lopo_result

    # ==========================
    # PROTOCOL 3: Cross-Session
    # ==========================
    print("\n" + "=" * 60)
    print("  PROTOCOL 3: Cross-Session Temporal Split")
    print("=" * 60)

    print("   Loading covert sessions...")
    X_cov_a, y_cov_a, _ = load_dataset_from_dir(COVERT_A, phase_label="Covert_A")
    X_cov_b, y_cov_b, _ = load_dataset_from_dir(COVERT_B, phase_label="Covert_B")
    X_cov_c, y_cov_c, _ = load_dataset_from_dir(COVERT_C, phase_label="Covert_C")

    print(f"   Session A (Feb 11): {len(X_cov_a)} samples")
    print(f"   Session B (Feb 25, 1cm lower): {len(X_cov_b)} samples")
    print(f"   Session C (Feb 25, 1cm lateral): {len(X_cov_c)} samples")

    if len(X_cov_a) > 0 and len(X_cov_b) > 0:
        le_cov = LabelEncoder()
        all_cov_labels = np.concatenate([y_cov_a, y_cov_b, y_cov_c])
        le_cov.fit(all_cov_labels)

        y_a = le_cov.transform(y_cov_a)
        X_test_bc = np.concatenate([X_cov_b, X_cov_c])
        y_test_bc = le_cov.transform(np.concatenate([y_cov_b, y_cov_c]))

        cross_result = run_cross_session(X_cov_a, y_a, X_test_bc, y_test_bc, le_cov)
        print(f"   Train on A ({cross_result['n_train']}), Test on B+C ({cross_result['n_test']})")
        print(f"   Train acc: {cross_result['train_acc']:.1%}")
        print(f"   ✅ Test acc: {cross_result['test_acc']:.1%}")
        all_results["cross_session"] = cross_result
    else:
        print("   ⚠️ Insufficient data for cross-session test")

    # ==========================
    # PROTOCOL 4: Combined Multi-Session CV
    # ==========================
    print("\n" + "=" * 60)
    print("  PROTOCOL 4: Combined Multi-Session 5-Fold CV (Covert)")
    print("=" * 60)

    if len(X_cov_a) > 0 and len(X_cov_b) > 0:
        X_combined = np.concatenate([X_cov_a, X_cov_b, X_cov_c])
        y_combined = le_cov.transform(np.concatenate([y_cov_a, y_cov_b, y_cov_c]))

        combined_cv = run_kfold_cv(X_combined, y_combined, None, le_cov)
        print(f"   Total samples: {len(X_combined)}")
        print(f"   Fold accuracies: {[f'{a:.1%}' for a in combined_cv['fold_accs']]}")
        print(f"   ✅ Mean: {combined_cv['mean']:.1%} ± {combined_cv['std']:.1%}")
        all_results["combined_cv"] = combined_cv
    else:
        print("   ⚠️ Skipped (insufficient data)")

    # ==========================
    # PROTOCOL 5: Hyperparameter Sweep
    # ==========================
    print("\n" + "=" * 60)
    print("  PROTOCOL 5: Hyperparameter Sweep (3-Fold CV)")
    print("=" * 60)

    sweep_results, best_params, best_acc = run_hyperparam_sweep(X_full, y_full, le)
    print(f"\n   ✅ Best config: {best_acc:.1%}")
    for k, v in sorted(best_params.items()):
        print(f"      {k}: {v}")
    all_results["hparam_sweep"] = {
        "all_results": sweep_results,
        "best_params": best_params,
        "best_acc": best_acc,
    }

    # ==========================
    # PROTOCOL 6: Confidence Gating (best model, 5-fold)
    # ==========================
    print("\n" + "=" * 60)
    print("  PROTOCOL 6: Confidence Gating (Best Model)")
    print("=" * 60)

    # Re-run 5-fold with best params, collecting probabilities
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_y_true, all_y_prob = [], []

    for train_idx, test_idx in skf.split(X_full, y_full):
        _, _, y_prob, _ = train_and_evaluate(
            X_full[train_idx], y_full[train_idx],
            X_full[test_idx], y_full[test_idx],
            num_classes=len(le.classes_), **best_params
        )
        all_y_true.extend(y_full[test_idx])
        all_y_prob.extend(y_prob)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    gating_results = run_confidence_gating(all_y_true, all_y_prob)
    for entry in gating_results:
        marker = " ← operating point" if abs(entry["theta"] - 0.60) < 0.01 else ""
        print(f"   θ={entry['theta']:.2f}: acc={entry['acc']:.1%}, coverage={entry['coverage']:.1%}, n={entry['n_accepted']}{marker}")
    all_results["confidence_gating"] = gating_results

    # ==========================
    # GENERATE MARKDOWN
    # ==========================
    output_path = os.path.join(PROJECT_ROOT, "results_summary.md")
    generate_markdown(all_results, output_path)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  DONE — {elapsed:.0f}s elapsed")
    print(f"  Results: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
