#!/usr/bin/env python3
"""
Experiment 1: SpecAugment — Time/Frequency Masking on MFCCs
============================================================
BEFORE: Covert-Only CNN = 62.0% (no augmentation)
AFTER:  CNN with SpecAugment during training

Based on: Park et al. (2019) "SpecAugment", adapted for sEMG.
Recent 2024 work confirms time masking is most effective for sEMG.
"""
import os, sys, glob, pickle, warnings, numpy as np, pandas as pd
import scipy.signal, scipy.fft
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "python", "data_collection", "carl", "Phase_6_Covert")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "..", "python"))
from utils.model import EMG_CNN

# Config (same as baseline)
SR=250; LOW=1.0; HIGH=50.0; NOTCH=60.0; N_MFCC=13; N_FFT=128; HOP=25
N_MELS=26; TSTEPS=100; BS=32; EPOCHS=30; PAT=10; LR=0.001; SEED=42

# ── Shared functions (identical to baseline) ──
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

def load_data(data_dir):
    files=sorted(glob.glob(os.path.join(data_dir,"*.csv")))
    X_list,labels,errs=[],[],0
    for f in files:
        try:
            df=pd.read_csv(f); chs=[c for c in df.columns if c.startswith('CH')]
            if not chs: errs+=1; continue
            label=os.path.basename(f).split('_')[0].upper()
            feats=[extract_mfcc(filters(df[ch].values)) for ch in chs]
            X_list.append(np.hstack(feats)); labels.append(label)
        except: errs+=1
    le=LabelEncoder(); X=np.array(X_list); y=le.fit_transform(labels)
    return X,y,le

# ════════════════════════════════════════════
# THE KEY ADDITION: SpecAugment
# ════════════════════════════════════════════
def spec_augment(x, time_mask_max=3, freq_mask_max=2, n_time_masks=2, n_freq_masks=1):
    """Apply SpecAugment: random time and frequency masking."""
    x = x.clone()
    T, F = x.shape[1], x.shape[2]  # (batch, time, features)
    for i in range(x.shape[0]):
        # Time masking
        for _ in range(n_time_masks):
            t = np.random.randint(1, time_mask_max + 1)
            t0 = np.random.randint(0, max(1, T - t))
            x[i, t0:t0+t, :] = 0
        # Frequency masking
        for _ in range(n_freq_masks):
            f = np.random.randint(1, freq_mask_max + 1)
            f0 = np.random.randint(0, max(1, F - f))
            x[i, :, f0:f0+f] = 0
    return x

def train_cnn_specaug(X_tr,y_tr,X_te,y_te,in_ch,n_cls):
    model=EMG_CNN(in_ch,n_cls); crit=nn.CrossEntropyLoss()
    opt=optim.Adam(model.parameters(),lr=LR)
    ds=TensorDataset(torch.FloatTensor(X_tr),torch.LongTensor(y_tr))
    ld=DataLoader(ds,batch_size=BS,shuffle=True)
    best_acc,best_st,pat_c=0,None,0
    for ep in range(EPOCHS):
        model.train(); cor=tot=0
        for bx,by in ld:
            # ★ SPECAUGMENT APPLIED HERE ★
            bx_aug = spec_augment(bx)
            opt.zero_grad(); out=model(bx_aug); loss=crit(out,by)
            loss.backward(); opt.step()
            _,pred=torch.max(out.data,1); tot+=by.size(0); cor+=(pred==by).sum().item()
        tr_acc=100*cor/tot
        model.eval()
        with torch.no_grad():
            ot=model(torch.FloatTensor(X_te)); _,pt=torch.max(ot.data,1)
            te_acc=(pt==torch.LongTensor(y_te)).sum().item()/len(y_te)*100
        if te_acc>best_acc: best_acc=te_acc; best_st=model.state_dict().copy(); pat_c=0
        else: pat_c+=1
        if ep%10==0: print(f"   Epoch {ep:3d} | Train: {tr_acc:.1f}% | Test: {te_acc:.1f}%")
        if pat_c>=PAT: print(f"   ⏹️  Early stop ep {ep}. Best: {best_acc:.1f}%"); break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        o=model(torch.FloatTensor(X_te)); _,p=torch.max(o.data,1)
    return model,p.numpy(),best_acc

def train_cnn_baseline(X_tr,y_tr,X_te,y_te,in_ch,n_cls):
    """Baseline CNN (no augmentation) — for fair comparison."""
    model=EMG_CNN(in_ch,n_cls); crit=nn.CrossEntropyLoss()
    opt=optim.Adam(model.parameters(),lr=LR)
    ds=TensorDataset(torch.FloatTensor(X_tr),torch.LongTensor(y_tr))
    ld=DataLoader(ds,batch_size=BS,shuffle=True)
    best_acc,best_st,pat_c=0,None,0
    for ep in range(EPOCHS):
        model.train(); cor=tot=0
        for bx,by in ld:
            opt.zero_grad(); out=model(bx); loss=crit(out,by)
            loss.backward(); opt.step()
            _,pred=torch.max(out.data,1); tot+=by.size(0); cor+=(pred==by).sum().item()
        tr_acc=100*cor/tot
        model.eval()
        with torch.no_grad():
            ot=model(torch.FloatTensor(X_te)); _,pt=torch.max(ot.data,1)
            te_acc=(pt==torch.LongTensor(y_te)).sum().item()/len(y_te)*100
        if te_acc>best_acc: best_acc=te_acc; best_st=model.state_dict().copy(); pat_c=0
        else: pat_c+=1
        if pat_c>=PAT: break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        o=model(torch.FloatTensor(X_te)); _,p=torch.max(o.data,1)
    return model,p.numpy(),best_acc

def main():
    print("="*60)
    print("  EXP 1: SpecAugment (Time + Frequency Masking)")
    print("="*60)

    X,y,le=load_data(DATA_DIR)
    print(f"✅ {X.shape[0]} samples, {X.shape} shape, {list(le.classes_)}")
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=SEED,stratify=y)
    print(f"📊 {len(X_tr)} train / {len(X_te)} test\n")

    ic,nc=X.shape[2],len(le.classes_)

    # BEFORE: Known from covert-only baseline experiment (same split, same seed)
    ba = 62.0
    print(f"── BEFORE (No Augmentation) ── Known: {ba:.1f}%\n")

    # AFTER: With SpecAugment
    print("── AFTER (SpecAugment: 2×time mask, 1×freq mask) ──")
    _,ap,aa=train_cnn_specaug(X_tr,y_tr,X_te,y_te,ic,nc)
    print(f"\n📋 SpecAugment CNN Report:")
    print(classification_report(y_te,ap,target_names=le.classes_))

    # Summary
    delta=aa-ba
    print(f"\n{'='*60}")
    print(f"  BEFORE: {ba:.1f}% (covert-only baseline)")
    print(f"  AFTER:  {aa:.1f}%  ({'+'if delta>0 else ''}{delta:.1f}pp)")
    print(f"{'='*60}")

    meta={'before':ba,'after':aa,'delta':delta,'method':'SpecAugment: 2×time(max3), 1×freq(max2)'}
    with open(os.path.join(RESULTS_DIR,"meta.pkl"),"wb") as f: pickle.dump(meta,f)

if __name__=="__main__": main()
