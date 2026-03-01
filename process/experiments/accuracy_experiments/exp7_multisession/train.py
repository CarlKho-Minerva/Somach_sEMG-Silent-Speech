#!/usr/bin/env python3
"""
Stage 2: Fast CNN training on pre-computed cache.
Usage: python3 train.py [1|2|3]
  1 = Single-session (A only)
  2 = Cross-session (A -> B+C)
  3 = Combined (A+B+C)
"""
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
        tr_acc = 100 * cor / tot
        model.eval()
        with torch.no_grad():
            ot = model(torch.FloatTensor(X_te)); _, pt = torch.max(ot.data, 1)
            te_acc = (pt == torch.LongTensor(y_te)).sum().item() / len(y_te) * 100
        if te_acc > best_acc: best_acc = te_acc; best_st = model.state_dict().copy(); pat_c = 0
        else: pat_c += 1
        if ep % 5 == 0: print(f"   Epoch {ep:3d} | Train: {tr_acc:.1f}% | Test: {te_acc:.1f}%")
        if pat_c >= PAT: print(f"   ⏹️  Early stop ep {ep}. Best: {best_acc:.1f}%"); break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        o = model(torch.FloatTensor(X_te)); _, p = torch.max(o.data, 1)
    return model, p.numpy(), best_acc

# Load cache
cache_path = os.path.join(SCRIPT_DIR, "cache.pkl")
with open(cache_path, "rb") as f:
    c = pickle.load(f)
Xa, la, Xb, lb, Xc, lc = c["Xa"], c["la"], c["Xb"], c["lb"], c["Xc"], c["lc"]

le = LabelEncoder(); le.fit(la + lb + lc)
ya, yb, yc = le.transform(la), le.transform(lb), le.transform(lc)
ic, nc = Xa.shape[2], len(le.classes_)

test_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print(f"📊 A:{len(la)} B:{len(lb)} C:{len(lc)} = {len(la)+len(lb)+len(lc)} | {nc} classes\n")

if test_num == 1:
    print("═" * 60)
    print("  TEST 1: Single-Session (Train 80% A, Test 20% A)")
    print("═" * 60)
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.2, random_state=SEED, stratify=ya)
    _, p, acc = train_cnn(Xa_tr, ya_tr, Xa_te, ya_te, ic, nc)
    print(f"\n📋 A-only accuracy: {acc:.1f}%")
    print(classification_report(ya_te, p, target_names=le.classes_))

elif test_num == 2:
    print("═" * 60)
    print("  TEST 2: Cross-Session (Train ALL A, Test ALL B+C)")
    print("═" * 60)
    X_bc = np.concatenate([Xb, Xc]); y_bc = np.concatenate([yb, yc])
    _, p, acc = train_cnn(Xa, ya, X_bc, y_bc, ic, nc)
    print(f"\n📋 Cross-session (A→B+C): {acc:.1f}%")
    print(classification_report(y_bc, p, target_names=le.classes_))

elif test_num == 3:
    print("═" * 60)
    print("  TEST 3: Combined (Train 80% A+B+C, Test 20% A+B+C)")
    print("═" * 60)
    X_all = np.concatenate([Xa, Xb, Xc]); y_all = np.concatenate([ya, yb, yc])
    s_all = np.array(["A"]*len(la) + ["B"]*len(lb) + ["C"]*len(lc))
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X_all, y_all, s_all, test_size=0.2, random_state=SEED, stratify=y_all)
    _, p, acc = train_cnn(X_tr, y_tr, X_te, y_te, ic, nc)
    print(f"\n📋 Combined (A+B+C): {acc:.1f}%")
    print(classification_report(y_te, p, target_names=le.classes_))
    for sess in ["A", "B", "C"]:
        mask = s_te == sess
        if mask.sum() > 0:
            print(f"   Session {sess}: {accuracy_score(y_te[mask], p[mask])*100:.1f}% ({mask.sum()} test samples)")
