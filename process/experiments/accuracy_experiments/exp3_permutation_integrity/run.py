#!/usr/bin/env python3
"""
Experiment 3: Permutation Importance — Integrity Check
======================================================
For EACH class, shuffle its test data and measure accuracy drop.
If shuffling NOISE barely affects NOISE accuracy → model uses artifacts.
This is your scientific integrity check.
"""
import os, sys, glob, pickle, warnings, numpy as np, pandas as pd
import scipy.signal, scipy.fft
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "python", "data_collection", "carl", "Phase_6_Covert")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SR=250; LOW=1.0; HIGH=50.0; NOTCH=60.0; N_MFCC=13; N_FFT=128; HOP=25
N_MELS=26; TSTEPS=100; SEED=42

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
    X_list,labels=[],[]
    for f in files:
        try:
            df=pd.read_csv(f); chs=[c for c in df.columns if c.startswith('CH')]
            if not chs: continue
            label=os.path.basename(f).split('_')[0].upper()
            feats=[extract_mfcc(filters(df[ch].values)) for ch in chs]
            X_list.append(np.hstack(feats)); labels.append(label)
        except: pass
    le=LabelEncoder(); X=np.array(X_list); y=le.fit_transform(labels)
    return X,y,le

def main():
    print("="*60)
    print("  EXP 3: Permutation Importance — Integrity Check")
    print("="*60)

    X,y,le=load_data(DATA_DIR)
    print(f"✅ {X.shape[0]} samples")
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=SEED,stratify=y)

    X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
    X_te_flat = X_te.reshape(X_te.shape[0], -1)

    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr_flat, y_tr)

    baseline_acc = accuracy_score(y_te, rf.predict(X_te_flat)) * 100
    print(f"\n📊 Baseline RF accuracy: {baseline_acc:.1f}%")

    # ── Per-class permutation test ──
    print(f"\n{'='*60}")
    print(f"  PER-CLASS PERMUTATION TEST")
    print(f"  (Shuffle samples of one class, measure accuracy drop)")
    print(f"{'='*60}\n")

    results = {}
    n_repeats = 10  # Average over 10 random shuffles

    for cls_idx, cls_name in enumerate(le.classes_):
        cls_mask = y_te == cls_idx
        acc_drops = []

        for r in range(n_repeats):
            X_te_perm = X_te_flat.copy()
            # Shuffle the features of this class's samples
            perm_idx = np.where(cls_mask)[0]
            rng = np.random.RandomState(SEED + r)
            for idx in perm_idx:
                X_te_perm[idx] = X_te_flat[rng.choice(len(X_te_flat))]

            perm_acc = accuracy_score(y_te, rf.predict(X_te_perm)) * 100
            acc_drops.append(baseline_acc - perm_acc)

        mean_drop = np.mean(acc_drops)
        std_drop = np.std(acc_drops)
        n_samples = cls_mask.sum()

        results[cls_name] = {'drop': mean_drop, 'std': std_drop, 'n': n_samples}

        flag = ""
        if mean_drop < 1.0:
            flag = "⚠️  SUSPECT: Model barely uses this class's signal content!"
        elif mean_drop > 5.0:
            flag = "✅ GOOD: Model relies on actual signal."

        print(f"   {cls_name:>10s}: Acc drop = {mean_drop:+.1f}% ± {std_drop:.1f}%  (n={n_samples})  {flag}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  INTEGRITY ASSESSMENT")
    print(f"{'='*60}")

    suspects = [k for k, v in results.items() if v['drop'] < 1.0]
    clean = [k for k, v in results.items() if v['drop'] >= 3.0]

    if suspects:
        print(f"\n⚠️  SUSPECT classes (drop < 1pp): {suspects}")
        print(f"   These may be classified by artifacts (recording length,")
        print(f"   amplitude statistics) rather than signal content.")
    else:
        print(f"\n✅  No suspect classes detected.")

    if clean:
        print(f"\n✅  CLEAN classes (drop ≥ 3pp): {clean}")
        print(f"   These are genuinely using MFCC signal content.")

    with open(os.path.join(RESULTS_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({'baseline': baseline_acc, 'results': results}, f)

if __name__ == "__main__": main()
