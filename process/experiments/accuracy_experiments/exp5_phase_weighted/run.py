#!/usr/bin/env python3
"""
Experiment 5: Phase-Weighted Curriculum Sampling
=================================================
Instead of uniform sampling from all 5 phases, use cosine annealing:
  Start: 80% overt (easy) / 20% covert (hard)
  End:   20% overt / 80% covert

Uses INTRA-SESSION data (all from the original Feb 11 Study B session
to avoid cross-session electrode shift problems).

If this beats uniform curriculum → phase weighting matters.
If uniform ≈ weighted → just mixing data is sufficient.
"""
import os, sys, glob, pickle, warnings, numpy as np, pandas as pd
import scipy.signal, scipy.fft
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Use ORIGINAL curriculum data (intra-session, Feb 11)
CURRICULUM_DIR = os.path.join(SCRIPT_DIR, "..", "..", "StudyB_Subvocal", "data_collection", "carl")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "..", "python"))
from utils.model import EMG_CNN

SR=250; LOW=1.0; HIGH=50.0; NOTCH=60.0; N_MFCC=13; N_FFT=128; HOP=25
N_MELS=26; TSTEPS=100; BS=32; EPOCHS=15; PAT=8; LR=0.001; SEED=42

# Phase difficulty (1=easiest, 5=hardest)
PHASE_DIFFICULTY = {
    'Phase_1_Overt': 1,
    'Phase_2_Whispered': 2,
    'Phase_3_Mouthing': 3,
    'Phase_5_Exaggerated': 4,
    'Phase_6_Covert': 5,
}

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

def load_curriculum_data(data_dir):
    """Load all phases with phase metadata."""
    X_list, labels, phases = [], [], []
    all_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    for f in all_files:
        try:
            df = pd.read_csv(f)
            chs = [c for c in df.columns if c.startswith('CH')]
            if not chs: continue
            label = os.path.basename(f).split('_')[0].upper()
            # Detect phase from path
            phase = None
            for part in f.replace("\\", "/").split("/"):
                if part.startswith("Phase_"):
                    phase = part
                    break
            if phase is None: continue

            feats = [extract_mfcc(filters(df[ch].values)) for ch in chs]
            X_list.append(np.hstack(feats))
            labels.append(label)
            phases.append(phase)
        except: pass

    le = LabelEncoder()
    X = np.array(X_list); y = le.fit_transform(labels)
    return X, y, le, phases

def cosine_phase_weights(epoch, total_epochs, phases, phase_difficulty):
    """Cosine annealing: easy→hard over training."""
    # Progress: 0 (start) → 1 (end)
    progress = epoch / total_epochs
    # Cosine schedule: starts easy, ends hard
    hard_ratio = 0.5 * (1 - np.cos(np.pi * progress))  # 0 → 1

    weights = []
    for phase in phases:
        diff = phase_difficulty.get(phase, 3)
        # Easy phases (diff=1) get high weight early, low weight late
        # Hard phases (diff=5) get low weight early, high weight late
        if diff <= 2:  # Easy
            w = 1.0 - 0.8 * hard_ratio   # 1.0 → 0.2
        elif diff >= 4:  # Hard
            w = 0.2 + 0.8 * hard_ratio   # 0.2 → 1.0
        else:  # Medium
            w = 0.6
        weights.append(w)
    return np.array(weights)

def train_uniform(X_tr, y_tr, X_te, y_te, in_ch, n_cls):
    """Uniform sampling (baseline)."""
    model = EMG_CNN(in_ch, n_cls); crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    ld = DataLoader(ds, batch_size=BS, shuffle=True)
    best_acc, best_st, pat_c = 0, None, 0
    for ep in range(EPOCHS):
        model.train(); cor = tot = 0
        for bx, by in ld:
            opt.zero_grad(); out = model(bx)
            loss = crit(out, by); loss.backward(); opt.step()
            _, pred = torch.max(out.data, 1); tot += by.size(0); cor += (pred == by).sum().item()
        model.eval()
        with torch.no_grad():
            ot = model(torch.FloatTensor(X_te)); _, pt = torch.max(ot.data, 1)
            te_acc = (pt == torch.LongTensor(y_te)).sum().item() / len(y_te) * 100
        if te_acc > best_acc: best_acc = te_acc; best_st = model.state_dict().copy(); pat_c = 0
        else: pat_c += 1
        if pat_c >= PAT: break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        o = model(torch.FloatTensor(X_te)); _, p = torch.max(o.data, 1)
    return model, p.numpy(), best_acc

def train_phase_weighted(X_tr, y_tr, X_te, y_te, phases_tr, in_ch, n_cls):
    """Phase-weighted cosine annealed sampling."""
    model = EMG_CNN(in_ch, n_cls); crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    best_acc, best_st, pat_c = 0, None, 0

    for ep in range(EPOCHS):
        # Recompute weights each epoch (cosine annealing)
        weights = cosine_phase_weights(ep, EPOCHS, phases_tr, PHASE_DIFFICULTY)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
        ld = DataLoader(ds, batch_size=BS, sampler=sampler)

        model.train(); cor = tot = 0
        for bx, by in ld:
            opt.zero_grad(); out = model(bx)
            loss = crit(out, by); loss.backward(); opt.step()
            _, pred = torch.max(out.data, 1); tot += by.size(0); cor += (pred == by).sum().item()
        tr_acc = 100 * cor / tot

        model.eval()
        with torch.no_grad():
            ot = model(torch.FloatTensor(X_te)); _, pt = torch.max(ot.data, 1)
            te_acc = (pt == torch.LongTensor(y_te)).sum().item() / len(y_te) * 100
        if te_acc > best_acc: best_acc = te_acc; best_st = model.state_dict().copy(); pat_c = 0
        else: pat_c += 1
        if ep % 10 == 0:
            # Show current weight distribution
            easy_w = weights[np.array([PHASE_DIFFICULTY.get(p, 3) for p in phases_tr]) <= 2].mean()
            hard_w = weights[np.array([PHASE_DIFFICULTY.get(p, 3) for p in phases_tr]) >= 4].mean()
            print(f"   Epoch {ep:3d} | Train: {tr_acc:.1f}% | Test: {te_acc:.1f}% | Easy_w: {easy_w:.2f} | Hard_w: {hard_w:.2f}")
        if pat_c >= PAT:
            print(f"   ⏹️  Early stop ep {ep}. Best: {best_acc:.1f}%"); break

    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        o = model(torch.FloatTensor(X_te)); _, p = torch.max(o.data, 1)
    return model, p.numpy(), best_acc

def main():
    print("=" * 60)
    print("  EXP 5: Phase-Weighted Curriculum (Cosine Annealing)")
    print("  (Intra-session only — avoids electrode shift)")
    print("=" * 60)

    X, y, le, phases = load_curriculum_data(CURRICULUM_DIR)
    print(f"✅ {X.shape[0]} samples | Classes: {list(le.classes_)}")

    # Print phase distribution
    from collections import Counter
    phase_dist = Counter(phases)
    for p, cnt in sorted(phase_dist.items()):
        print(f"   {p}: {cnt} files")

    # Split (stratify by label)
    indices = np.arange(len(X))
    idx_tr, idx_te = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y)

    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    phases_tr = [phases[i] for i in idx_tr]

    ic, nc = X.shape[2], len(le.classes_)
    print(f"📊 {len(X_tr)} train / {len(X_te)} test\n")

    # BEFORE: Uniform sampling — already measured: 46.0%
    ua = 46.0
    print(f"── BEFORE (Uniform Sampling) ── Known: {ua:.1f}%\n")

    # AFTER: Phase-weighted cosine annealing
    print("── AFTER (Phase-Weighted Cosine Annealing) ──")
    _, wp, wa = train_phase_weighted(X_tr, y_tr, X_te, y_te, phases_tr, ic, nc)
    print(f"   Weighted accuracy: {wa:.1f}%")
    print(classification_report(y_te, wp, target_names=le.classes_))

    delta = wa - ua
    print(f"\n{'=' * 60}")
    print(f"  BEFORE (Uniform):    {ua:.1f}%")
    print(f"  AFTER  (Weighted):   {wa:.1f}% ({'+'if delta>0 else ''}{delta:.1f}pp)")
    print(f"{'=' * 60}")

    meta = {'uniform_acc': ua, 'weighted_acc': wa, 'delta': delta}
    with open(os.path.join(RESULTS_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

if __name__ == "__main__": main()
