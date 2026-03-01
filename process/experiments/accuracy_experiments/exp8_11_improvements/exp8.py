#!/usr/bin/env python3
"""EXP 8: 4-class and 5-class reduction. Loads from cache."""
import os, sys, pickle, numpy as np, warnings
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
warnings.filterwarnings("ignore")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "..", "python"))
from utils.model import EMG_CNN
BS=64; EPOCHS=25; PAT=10; LR=0.001; SEED=42

def train_cnn(X_tr, y_tr, X_te, y_te, ic, nc):
    np.random.seed(SEED); torch.manual_seed(SEED)
    m = EMG_CNN(ic, nc); c = nn.CrossEntropyLoss(); o = optim.Adam(m.parameters(), lr=LR)
    ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    ld = DataLoader(ds, batch_size=BS, shuffle=True)
    best, bst, pc = 0, None, 0
    for ep in range(EPOCHS):
        m.train()
        for bx, by in ld:
            o.zero_grad(); out = m(bx); l = c(out, by); l.backward(); o.step()
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
        _, p = torch.max(m(torch.FloatTensor(X_te)), 1)
    return p.numpy(), best

with open(os.path.join(SCRIPT_DIR, "..", "exp7_multisession", "cache.pkl"), "rb") as f:
    c = pickle.load(f)
X_all = np.concatenate([c["Xa"], c["Xb"], c["Xc"]])
l_all = c["la"] + c["lb"] + c["lc"]
ic = X_all.shape[2]

# 5-class: drop LEFT
mask = np.array([l != "LEFT" for l in l_all])
X5 = X_all[mask]; l5 = [l for l in l_all if l != "LEFT"]
le5 = LabelEncoder(); le5.fit(l5); y5 = le5.transform(l5)
Xtr,Xte,ytr,yte = train_test_split(X5,y5,test_size=0.2,random_state=SEED,stratify=y5)
print(f"5-CLASS ({len(l5)} samples, classes: {list(le5.classes_)})")
p5, a5 = train_cnn(Xtr,ytr,Xte,yte,ic,len(le5.classes_))
print(f"\n📋 5-class: {a5:.1f}%")
print(classification_report(yte, p5, target_names=le5.classes_))

# 4-class: merge NOISE+SILENCE → IDLE
l4 = ["IDLE" if l in ("NOISE","SILENCE") else l for l in l5]
le4 = LabelEncoder(); le4.fit(l4); y4 = le4.transform(l4)
Xtr4,Xte4,ytr4,yte4 = train_test_split(X5,y4,test_size=0.2,random_state=SEED,stratify=y4)
print(f"\n4-CLASS ({len(l4)} samples, classes: {list(le4.classes_)})")
p4, a4 = train_cnn(Xtr4,ytr4,Xte4,yte4,ic,len(le4.classes_))
print(f"\n📋 4-class: {a4:.1f}%")
print(classification_report(yte4, p4, target_names=le4.classes_))
