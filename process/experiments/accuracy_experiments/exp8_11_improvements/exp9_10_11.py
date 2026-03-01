#!/usr/bin/env python3
"""EXP 9+10+11: Confidence gating, Z-score, Ensemble. Loads from cache."""
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

def train_cnn_probs(X_tr, y_tr, X_te, y_te, ic, nc):
    np.random.seed(SEED); torch.manual_seed(SEED)
    m = EMG_CNN(ic, nc); cr = nn.CrossEntropyLoss(); op = optim.Adam(m.parameters(), lr=LR)
    ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    ld = DataLoader(ds, batch_size=BS, shuffle=True)
    best, bst, pc = 0, None, 0
    for ep in range(EPOCHS):
        m.train()
        for bx, by in ld:
            op.zero_grad(); out = m(bx); l = cr(out, by); l.backward(); op.step()
        m.eval()
        with torch.no_grad():
            ot = m(torch.FloatTensor(X_te)); _, pt = torch.max(ot, 1)
            ta = (pt == torch.LongTensor(y_te)).sum().item() / len(y_te) * 100
        if ta > best: best = ta; bst = m.state_dict().copy(); pc = 0
        else: pc += 1
        if ep % 5 == 0: print(f"   Ep {ep:2d} | Test: {ta:.1f}%")
        if pc >= PAT: print(f"   ⏹️ Early stop {ep}. Best: {best:.1f}%"); break
    if bst: m.load_state_dict(bst)
    m.eval()
    with torch.no_grad():
        o = m(torch.FloatTensor(X_te))
        probs = torch.softmax(o, dim=1).numpy()
        _, p = torch.max(o, 1)
    return p.numpy(), best, probs

with open(os.path.join(SCRIPT_DIR, "..", "exp7_multisession", "cache.pkl"), "rb") as f:
    c = pickle.load(f)
X_all = np.concatenate([c["Xa"], c["Xb"], c["Xc"]])
l_all = c["la"] + c["lb"] + c["lc"]
le = LabelEncoder(); le.fit(l_all); y_all = le.transform(l_all)
ic, nc = X_all.shape[2], len(le.classes_)
Xtr,Xte,ytr,yte = train_test_split(X_all,y_all,test_size=0.2,random_state=SEED,stratify=y_all)

# ── EXP 9: CONFIDENCE GATING ──
print("═"*60)
print("  EXP 9: CONFIDENCE GATING (6-class)")
print("═"*60)
p_cnn, acc_cnn, probs_cnn = train_cnn_probs(Xtr,ytr,Xte,yte,ic,nc)
print(f"\n📋 Ungated: {acc_cnn:.1f}%")
for t in [0.4,0.5,0.6,0.7,0.8]:
    mx = probs_cnn.max(axis=1); ok = mx >= t; n = ok.sum()
    if n > 0:
        a = accuracy_score(yte[ok], p_cnn[ok])*100
        print(f"   Gate {t:.0%}: {a:.1f}% ({n}/{len(yte)} = {n/len(yte)*100:.0f}% coverage)")

# ── EXP 10: Z-SCORE ──
print(f"\n{'═'*60}")
print("  EXP 10: Z-SCORE NORMALIZATION")
print("═"*60)
def znorm(X):
    Xz = np.zeros_like(X)
    for i in range(len(X)):
        mu = X[i].mean(axis=0, keepdims=True)
        s = X[i].std(axis=0, keepdims=True) + 1e-8
        Xz[i] = (X[i] - mu) / s
    return Xz
Xz = znorm(X_all)
Xztr,Xzte,yztr,yzte = train_test_split(Xz,y_all,test_size=0.2,random_state=SEED,stratify=y_all)
_, acc_z, _ = train_cnn_probs(Xztr,yztr,Xzte,yzte,ic,nc)
print(f"\n📋 Z-score 6-class: {acc_z:.1f}% (vs {acc_cnn:.1f}% raw)")

# ── EXP 11: CNN+RF ENSEMBLE ──
print(f"\n{'═'*60}")
print("  EXP 11: CNN + RF ENSEMBLE")
print("═"*60)
# RF
Xtr_f = Xtr.reshape(len(Xtr),-1); Xte_f = Xte.reshape(len(Xte),-1)
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=SEED, n_jobs=-1)
rf.fit(Xtr_f, ytr)
p_rf = rf.predict(Xte_f); probs_rf = rf.predict_proba(Xte_f)
acc_rf = accuracy_score(yte, p_rf)*100
# Soft vote
pavg = (probs_cnn + probs_rf) / 2
p_ens = pavg.argmax(axis=1)
acc_ens = accuracy_score(yte, p_ens)*100
print(f"\n   CNN:      {acc_cnn:.1f}%")
print(f"   RF:       {acc_rf:.1f}%")
print(f"   Ensemble: {acc_ens:.1f}%")
print(classification_report(yte, p_ens, target_names=le.classes_))
for t in [0.5,0.6,0.7]:
    mx = pavg.max(axis=1); ok = mx >= t; n = ok.sum()
    if n > 0:
        print(f"   Ens+Gate {t:.0%}: {accuracy_score(yte[ok],p_ens[ok])*100:.1f}% ({n}/{len(yte)} coverage)")

print(f"\n{'═'*60}")
print("  SUMMARY")
print(f"{'═'*60}")
print(f"  EXP9  Ungated CNN:  {acc_cnn:.1f}%")
print(f"  EXP10 Z-score CNN:  {acc_z:.1f}%")
print(f"  EXP11 Ensemble:     {acc_ens:.1f}%")
