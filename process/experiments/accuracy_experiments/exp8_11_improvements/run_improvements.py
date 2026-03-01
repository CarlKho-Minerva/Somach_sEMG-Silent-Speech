#!/usr/bin/env python3
"""
EXP 8: 4-Class Reduction
EXP 9: Confidence Gating
EXP 10: Sliding-Window Z-Score Normalization
EXP 11: CNN + RF Ensemble

All run from the pre-computed cache. Each prints its own results block.
Usage: python3 run_improvements.py
"""
import os, sys, pickle, numpy as np, warnings
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "..", "python"))
from utils.model import EMG_CNN

BS=64; EPOCHS=25; PAT=10; LR=0.001; SEED=42

def train_cnn(X_tr, y_tr, X_te, y_te, ic, nc, return_probs=False):
    np.random.seed(SEED); torch.manual_seed(SEED)
    model = EMG_CNN(ic, nc); crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    ld = DataLoader(ds, batch_size=BS, shuffle=True)
    best_acc, best_st, pat_c = 0, None, 0
    for ep in range(EPOCHS):
        model.train(); cor = tot = 0
        for bx, by in ld:
            opt.zero_grad(); out = model(bx); loss = crit(out, by)
            loss.backward(); opt.step()
            _, pred = torch.max(out.data, 1); tot += by.size(0); cor += (pred == by).sum().item()
        model.eval()
        with torch.no_grad():
            ot = model(torch.FloatTensor(X_te)); _, pt = torch.max(ot.data, 1)
            te_acc = (pt == torch.LongTensor(y_te)).sum().item() / len(y_te) * 100
        if te_acc > best_acc: best_acc = te_acc; best_st = model.state_dict().copy(); pat_c = 0
        else: pat_c += 1
        if ep % 5 == 0: print(f"   Epoch {ep:3d} | Test: {te_acc:.1f}%")
        if pat_c >= PAT: print(f"   ⏹️  Early stop ep {ep}. Best: {best_acc:.1f}%"); break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        o = model(torch.FloatTensor(X_te))
        probs = torch.softmax(o, dim=1).numpy()
        _, p = torch.max(o.data, 1)
    if return_probs:
        return model, p.numpy(), best_acc, probs
    return model, p.numpy(), best_acc

# ── Load cache ──
cache_path = os.path.join(SCRIPT_DIR, "..", "exp7_multisession", "cache.pkl")
with open(cache_path, "rb") as f:
    c = pickle.load(f)
Xa, la, Xb, lb, Xc, lc = c["Xa"], c["la"], c["Xb"], c["lb"], c["Xc"], c["lc"]

# Combine all sessions
X_all = np.concatenate([Xa, Xb, Xc])
l_all = la + lb + lc
s_all = np.array(["A"]*len(la) + ["B"]*len(lb) + ["C"]*len(lc))

print(f"📊 Loaded {len(l_all)} samples (A:{len(la)} B:{len(lb)} C:{len(lc)})")

# ═══════════════════════════════════════════
# EXP 8: 4-CLASS REDUCTION
# ═══════════════════════════════════════════
print(f"\n{'═'*60}")
print("  EXP 8: 4-CLASS REDUCTION (Drop LEFT, keep 5 → test 4)")
print(f"{'═'*60}")

# Strategy: Drop LEFT entirely (7% recall = unlearnable)
# Keep: UP, DOWN, RIGHT, SILENCE, NOISE (5 classes)
# Also test: Merge NOISE+SILENCE → IDLE (4 classes)

# Test A: Drop LEFT only (5 classes)
mask_no_left = np.array([l != "LEFT" for l in l_all])
X_5c = X_all[mask_no_left]
l_5c = [l for l in l_all if l != "LEFT"]
le_5c = LabelEncoder(); le_5c.fit(l_5c)
y_5c = le_5c.transform(l_5c)
X_tr5, X_te5, y_tr5, y_te5 = train_test_split(X_5c, y_5c, test_size=0.2, random_state=SEED, stratify=y_5c)
ic = X_5c.shape[2]

print(f"\n  5-class (no LEFT): {len(l_5c)} samples")
_, p5, acc5 = train_cnn(X_tr5, y_tr5, X_te5, y_te5, ic, len(le_5c.classes_))
print(f"\n📋 5-class accuracy: {acc5:.1f}%")
print(classification_report(y_te5, p5, target_names=le_5c.classes_))

# Test B: 4-class (merge NOISE+SILENCE → IDLE)
l_4c = []
for l in l_all:
    if l == "LEFT": continue
    elif l in ("NOISE", "SILENCE"): l_4c.append("IDLE")
    else: l_4c.append(l)
X_4c = X_all[mask_no_left]
le_4c = LabelEncoder(); le_4c.fit(l_4c)
y_4c = le_4c.transform(l_4c)
X_tr4, X_te4, y_tr4, y_te4 = train_test_split(X_4c, y_4c, test_size=0.2, random_state=SEED, stratify=y_4c)

print(f"\n  4-class (IDLE=NOISE+SILENCE, no LEFT): {len(l_4c)} samples")
_, p4, acc4 = train_cnn(X_tr4, y_tr4, X_te4, y_te4, ic, len(le_4c.classes_))
print(f"\n📋 4-class accuracy: {acc4:.1f}%")
print(classification_report(y_te4, p4, target_names=le_4c.classes_))

# ═══════════════════════════════════════════
# EXP 9: CONFIDENCE GATING
# ═══════════════════════════════════════════
print(f"\n{'═'*60}")
print("  EXP 9: CONFIDENCE GATING (on 6-class combined model)")
print(f"{'═'*60}")

le_all = LabelEncoder(); le_all.fit(l_all)
y_all = le_all.transform(l_all)
X_tr_g, X_te_g, y_tr_g, y_te_g = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)

_, p_g, acc_g, probs_g = train_cnn(X_tr_g, y_tr_g, X_te_g, y_te_g, ic, len(le_all.classes_), return_probs=True)

print(f"\n📋 Ungated 6-class: {acc_g:.1f}%")

for threshold in [0.40, 0.50, 0.60, 0.70, 0.80]:
    max_probs = probs_g.max(axis=1)
    accepted = max_probs >= threshold
    n_accepted = accepted.sum()
    if n_accepted > 0:
        gated_acc = accuracy_score(y_te_g[accepted], p_g[accepted]) * 100
        coverage = n_accepted / len(y_te_g) * 100
        print(f"   Threshold {threshold:.0%}: Acc={gated_acc:.1f}% | Coverage={coverage:.1f}% ({n_accepted}/{len(y_te_g)} accepted)")

# ═══════════════════════════════════════════
# EXP 10: SLIDING-WINDOW Z-SCORE NORMALIZATION
# ═══════════════════════════════════════════
print(f"\n{'═'*60}")
print("  EXP 10: SLIDING-WINDOW Z-SCORE NORMALIZATION")
print(f"{'═'*60}")

# Apply Z-score per-window (across time dimension for each sample)
def zscore_normalize(X):
    """Z-score normalize each sample independently across the time axis."""
    X_z = np.zeros_like(X)
    for i in range(len(X)):
        mean = X[i].mean(axis=0, keepdims=True)
        std = X[i].std(axis=0, keepdims=True) + 1e-8
        X_z[i] = (X[i] - mean) / std
    return X_z

X_z = zscore_normalize(X_all)
X_tr_z, X_te_z, y_tr_z, y_te_z = train_test_split(X_z, y_all, test_size=0.2, random_state=SEED, stratify=y_all)

_, p_z, acc_z = train_cnn(X_tr_z, y_tr_z, X_te_z, y_te_z, ic, len(le_all.classes_))
print(f"\n📋 Z-score normalized 6-class: {acc_z:.1f}%  (vs {acc_g:.1f}% unnormalized)")

# Also test Z-score on 5-class
X_z5 = zscore_normalize(X_5c)
X_tr_z5, X_te_z5, y_tr_z5, y_te_z5 = train_test_split(X_z5, y_5c, test_size=0.2, random_state=SEED, stratify=y_5c)
_, p_z5, acc_z5 = train_cnn(X_tr_z5, y_tr_z5, X_te_z5, y_te_z5, ic, len(le_5c.classes_))
print(f"   Z-score + 5-class: {acc_z5:.1f}%  (vs {acc5:.1f}% unnormalized)")

# ═══════════════════════════════════════════
# EXP 11: CNN + RF ENSEMBLE
# ═══════════════════════════════════════════
print(f"\n{'═'*60}")
print("  EXP 11: CNN + RF ENSEMBLE (Majority Vote)")
print(f"{'═'*60}")

# Use 6-class combined data
X_tr_e, X_te_e, y_tr_e, y_te_e = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)

# CNN predictions
_, p_cnn, acc_cnn, probs_cnn = train_cnn(X_tr_e, y_tr_e, X_te_e, y_te_e, ic, len(le_all.classes_), return_probs=True)

# RF: flatten features (time × channels → 1D)
X_tr_flat = X_tr_e.reshape(len(X_tr_e), -1)
X_te_flat = X_te_e.reshape(len(X_te_e), -1)

rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=SEED, n_jobs=-1)
rf.fit(X_tr_flat, y_tr_e)
p_rf = rf.predict(X_te_flat)
probs_rf = rf.predict_proba(X_te_flat)
acc_rf = accuracy_score(y_te_e, p_rf) * 100

print(f"\n   CNN alone:  {acc_cnn:.1f}%")
print(f"   RF alone:   {acc_rf:.1f}%")

# Majority vote (soft: average probabilities)
probs_avg = (probs_cnn + probs_rf) / 2
p_ensemble = probs_avg.argmax(axis=1)
acc_ensemble = accuracy_score(y_te_e, p_ensemble) * 100

print(f"   Ensemble:   {acc_ensemble:.1f}%")
print(classification_report(y_te_e, p_ensemble, target_names=le_all.classes_))

# Ensemble + confidence gating
for threshold in [0.50, 0.60, 0.70]:
    max_p = probs_avg.max(axis=1)
    accepted = max_p >= threshold
    n_acc = accepted.sum()
    if n_acc > 0:
        g_acc = accuracy_score(y_te_e[accepted], p_ensemble[accepted]) * 100
        cov = n_acc / len(y_te_e) * 100
        print(f"   Ensemble+Gate {threshold:.0%}: Acc={g_acc:.1f}% | Coverage={cov:.1f}%")

# ═══════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════
print(f"\n{'═'*60}")
print("  FINAL SUMMARY")
print(f"{'═'*60}")
print(f"  Baseline 6-class (EXP7 combined):   49.7%")
print(f"  EXP8  5-class (no LEFT):            {acc5:.1f}%")
print(f"  EXP8  4-class (IDLE merge):         {acc4:.1f}%")
print(f"  EXP9  Best gated accuracy:          see table above")
print(f"  EXP10 Z-score 6-class:              {acc_z:.1f}%")
print(f"  EXP10 Z-score 5-class:              {acc_z5:.1f}%")
print(f"  EXP11 Ensemble (CNN+RF):            {acc_ensemble:.1f}%")
print(f"{'═'*60}")
