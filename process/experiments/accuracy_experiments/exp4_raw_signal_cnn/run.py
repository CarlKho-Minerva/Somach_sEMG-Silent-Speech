#!/usr/bin/env python3
"""
Experiment 4: Raw Signal CNN (Skip MFCCs)
==========================================
BEFORE: MFCC-based CNN = 62.0%
AFTER:  Feed raw filtered 2-channel EMG directly into a deeper CNN.

If raw signal CNN beats MFCCs → the spectral envelope features are
throwing away useful information (phase, temporal microstructure).
If MFCCs beat raw → MFCCs are the right abstraction for this domain.
"""
import os, sys, glob, pickle, warnings, numpy as np, pandas as pd
import scipy.signal
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "python", "data_collection", "carl", "Phase_6_Covert")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SR=250; LOW=1.0; HIGH=50.0; NOTCH=60.0; SEED=42
TARGET_SAMPLES = 500  # 2 seconds at 250Hz
BS=32; EPOCHS=100; PAT=15; LR=0.001

# ── Raw Signal CNN (deeper, larger first kernel) ──
class RawSignalCNN(nn.Module):
    """Deeper 1D CNN designed for raw EMG input (no MFCC)."""
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1: Large kernel for capturing motor unit patterns
            nn.Conv1d(n_channels, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            # Layer 2
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            # Layer 3
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            # Layer 4
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, time, channels) → (batch, channels, time)
        x = x.permute(0, 2, 1)
        x = self.features(x).squeeze(-1)
        return self.classifier(x)

def filters(d):
    nyq=SR/2; b,a=scipy.signal.butter(4,[LOW/nyq,HIGH/nyq],btype='band')
    y=scipy.signal.filtfilt(b,a,d)
    bn,an=scipy.signal.iirnotch(NOTCH/nyq,30); y=scipy.signal.filtfilt(bn,an,y)
    mn,mx=np.min(y),np.max(y)
    return (y-mn)/(mx-mn) if mx-mn!=0 else y

def pad_or_trunc(sig, target):
    if len(sig) < target:
        sig = np.pad(sig, (0, target - len(sig)))
    else:
        sig = sig[:target]
    return sig

def load_raw_data(data_dir):
    """Load raw filtered signals (no MFCC extraction)."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    X_list, labels, errs = [], [], 0
    for f in files:
        try:
            df = pd.read_csv(f)
            chs = [c for c in df.columns if c.startswith('CH')]
            if not chs: errs += 1; continue
            label = os.path.basename(f).split('_')[0].upper()
            channels = []
            for ch in chs:
                clean = filters(df[ch].values)
                clean = pad_or_trunc(clean, TARGET_SAMPLES)
                channels.append(clean)
            # Stack channels: (time, n_channels)
            X_list.append(np.column_stack(channels))
            labels.append(label)
        except:
            errs += 1
    le = LabelEncoder()
    X = np.array(X_list); y = le.fit_transform(labels)
    return X, y, le

def train_raw_cnn(X_tr, y_tr, X_te, y_te, n_ch, n_cls):
    model = RawSignalCNN(n_ch, n_cls)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    ld = DataLoader(ds, batch_size=BS, shuffle=True)
    best_acc, best_st, pat_c = 0, None, 0
    for ep in range(EPOCHS):
        model.train(); cor = tot = 0
        for bx, by in ld:
            opt.zero_grad(); out = model(bx)
            loss = crit(out, by); loss.backward(); opt.step()
            _, pred = torch.max(out.data, 1)
            tot += by.size(0); cor += (pred == by).sum().item()
        tr_acc = 100 * cor / tot
        model.eval()
        with torch.no_grad():
            ot = model(torch.FloatTensor(X_te))
            _, pt = torch.max(ot.data, 1)
            te_acc = (pt == torch.LongTensor(y_te)).sum().item() / len(y_te) * 100
        if te_acc > best_acc:
            best_acc = te_acc; best_st = model.state_dict().copy(); pat_c = 0
        else: pat_c += 1
        if ep % 10 == 0:
            print(f"   Epoch {ep:3d} | Train: {tr_acc:.1f}% | Test: {te_acc:.1f}%")
        if pat_c >= PAT:
            print(f"   ⏹️  Early stop ep {ep}. Best: {best_acc:.1f}%"); break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        o = model(torch.FloatTensor(X_te))
        _, p = torch.max(o.data, 1)
    return model, p.numpy(), best_acc

def main():
    print("=" * 60)
    print("  EXP 4: Raw Signal CNN (No MFCCs)")
    print("=" * 60)

    X, y, le = load_raw_data(DATA_DIR)
    n_ch = X.shape[2]
    n_cls = len(le.classes_)
    print(f"✅ {X.shape[0]} samples | Shape: {X.shape} | {n_ch} channels")
    print(f"   (Raw filtered signal, {TARGET_SAMPLES} samples/recording, no MFCC)")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"📊 {len(X_tr)} train / {len(X_te)} test\n")

    print("── Raw Signal CNN (4-layer, kernel 11/7/5/3) ──")
    model, preds, raw_acc = train_raw_cnn(X_tr, y_tr, X_te, y_te, n_ch, n_cls)
    print(f"\n📋 Raw CNN Report:")
    print(classification_report(y_te, preds, target_names=le.classes_))

    mfcc_baseline = 62.0  # From baseline experiment
    delta = raw_acc - mfcc_baseline

    print(f"\n{'=' * 60}")
    print(f"  BEFORE (MFCC CNN):  {mfcc_baseline:.1f}%")
    print(f"  AFTER  (Raw CNN):   {raw_acc:.1f}% ({'+'if delta>0 else ''}{delta:.1f}pp)")
    if raw_acc > mfcc_baseline:
        print(f"  ✅ Raw signal > MFCCs! Spectral features discard useful info.")
    else:
        print(f"  📋 MFCCs still better. Spectral envelope IS the right feature.")
    print(f"{'=' * 60}")

    meta = {'mfcc_baseline': mfcc_baseline, 'raw_acc': raw_acc, 'delta': delta}
    with open(os.path.join(RESULTS_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "raw_cnn.pth"))

if __name__ == "__main__": main()
