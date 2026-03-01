#!/usr/bin/env python3
"""
Experiment 7: Multi-Session Retraining (Streamlined)
=====================================================
Reduced to 3 critical comparisons with 25 epochs each:
  1. Single-session (A only) — baseline
  2. Cross-session (A→B+C) — quantifies the electrode shift problem
  3. Combined (A+B+C) — proves multi-session training helps
"""
import os, sys, glob, warnings, numpy as np, pandas as pd
import scipy.signal, scipy.fft
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA = os.path.join(SCRIPT_DIR, "..", "..", "..", "python", "data_collection")
SESSION_A = os.path.join(BASE_DATA, "carl", "Phase_6_Covert")
SESSION_B = os.path.join(BASE_DATA, "carl_022526_sessionb", "Phase_6_Covert")
SESSION_C = os.path.join(BASE_DATA, "carl_sessionc2", "Phase_6_Covert")
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "..", "python"))
from utils.model import EMG_CNN

SR=250; LOW=1.0; HIGH=50.0; NOTCH=60.0; N_MFCC=13; N_FFT=128; HOP=25
N_MELS=26; TSTEPS=100; BS=64; EPOCHS=25; PAT=10; LR=0.001; SEED=42

def _hz2mel(hz): return 2595*np.log10(1+hz/700)
def _mel2hz(m): return 700*(10**(m/2595)-1)
def _melfb(sr,nfft,nmels):
    pts=np.linspace(_hz2mel(0),_hz2mel(sr/2),nmels+2)
    hz=_mel2hz(pts); bins=np.floor((nfft+1)*hz/sr).astype(int)
    fb=np.zeros((nmels,nfft//2+1))
    for i in range(nmels):
        l,c,r=bins[i],bins[i+1],bins[i+2]
        for j in range(l,c):
            if c>l: fb[i,j]=(j-l)/(c-l)
        for j in range(c,r):
            if r>c: fb[i,j]=(r-j)/(r-c)
    return fb
_FB=None
def _getfb():
    global _FB
    if _FB is None: _FB=_melfb(SR,N_FFT,N_MELS)
    return _FB
def extract_mfcc(sig):
    sig=sig.astype(np.float64)
    nf=1+(len(sig)-N_FFT)//HOP
    if nf<=0: sig=np.pad(sig,(0,N_FFT-len(sig)+HOP)); nf=1+(len(sig)-N_FFT)//HOP
    w=np.hanning(N_FFT); frames=np.zeros((nf,N_FFT))
    for i in range(nf): frames[i]=sig[i*HOP:i*HOP+N_FFT]*w
    pwr=np.abs(scipy.fft.rfft(frames,n=N_FFT,axis=1))**2
    mel=np.log(np.dot(pwr,_getfb().T)+1e-10)
    feat=scipy.fft.dct(mel,type=2,axis=1,norm='ortho')[:,:N_MFCC]
    if len(feat)<TSTEPS: feat=np.pad(feat,((0,TSTEPS-len(feat)),(0,0)))
    else: feat=feat[:TSTEPS]
    return feat
def filters(d):
    nyq=SR/2; b,a=scipy.signal.butter(4,[LOW/nyq,HIGH/nyq],btype='band')
    y=scipy.signal.filtfilt(b,a,d)
    bn,an=scipy.signal.iirnotch(NOTCH/nyq,30); y=scipy.signal.filtfilt(bn,an,y)
    mn,mx=np.min(y),np.max(y)
    return (y-mn)/(mx-mn) if mx-mn!=0 else y

def load_session(data_dir, tag):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    X_list, labels, sessions, errs = [], [], [], 0
    for f in files:
        try:
            df = pd.read_csv(f)
            chs = [c for c in df.columns if c.startswith('CH')]
            if not chs: errs += 1; continue
            label = os.path.basename(f).split('_')[0].upper()
            feats = [extract_mfcc(filters(df[ch].values)) for ch in chs]
            X_list.append(np.hstack(feats)); labels.append(label); sessions.append(tag)
        except: errs += 1
    return np.array(X_list), labels, sessions, errs

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

def main():
    import sys as _sys
    test_num = int(_sys.argv[1]) if len(_sys.argv) > 1 else 0

    # Load all sessions
    print("📁 Loading sessions...")
    Xa, la, sa, ea = load_session(SESSION_A, "A")
    Xb, lb, sb, eb = load_session(SESSION_B, "B")
    Xc, lc, sc, ec = load_session(SESSION_C, "C")
    print(f"   A: {len(la)} | B: {len(lb)} | C: {len(lc)} | Total: {len(la)+len(lb)+len(lc)}")

    le = LabelEncoder(); le.fit(la + lb + lc)
    ya, yb, yc = le.transform(la), le.transform(lb), le.transform(lc)
    ic, nc = Xa.shape[2], len(le.classes_)

    if test_num == 1 or test_num == 0:
        print(f"\n{'═'*60}")
        print("  TEST 1: Single-Session (Train A, Test A)")
        print(f"{'═'*60}")
        Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.2, random_state=SEED, stratify=ya)
        _, p1, acc1 = train_cnn(Xa_tr, ya_tr, Xa_te, ya_te, ic, nc)
        print(f"\n📋 A-only: {acc1:.1f}%")
        print(classification_report(ya_te, p1, target_names=le.classes_))

    if test_num == 2 or test_num == 0:
        print(f"\n{'═'*60}")
        print("  TEST 2: Cross-Session (Train A, Test B+C)")
        print(f"{'═'*60}")
        X_bc = np.concatenate([Xb, Xc]); y_bc = np.concatenate([yb, yc])
        _, p2, acc2 = train_cnn(Xa, ya, X_bc, y_bc, ic, nc)
        print(f"\n📋 Cross-session (A→B+C): {acc2:.1f}%")
        print(classification_report(y_bc, p2, target_names=le.classes_))

    if test_num == 3 or test_num == 0:
        print(f"\n{'═'*60}")
        print("  TEST 3: Combined (Train A+B+C, Test held-out)")
        print(f"{'═'*60}")
        X_all = np.concatenate([Xa, Xb, Xc]); y_all = np.concatenate([ya, yb, yc])
        s_all = np.array(sa + sb + sc)
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X_all, y_all, s_all, test_size=0.2, random_state=SEED, stratify=y_all)
        _, p3, acc3 = train_cnn(X_tr, y_tr, X_te, y_te, ic, nc)
        print(f"\n📋 Combined (A+B+C): {acc3:.1f}%")
        print(classification_report(y_te, p3, target_names=le.classes_))
        for sess in ["A", "B", "C"]:
            mask = s_te == sess
            if mask.sum() > 0:
                print(f"   Session {sess}: {accuracy_score(y_te[mask], p3[mask])*100:.1f}% ({mask.sum()} samples)")

if __name__ == "__main__":
    main()
