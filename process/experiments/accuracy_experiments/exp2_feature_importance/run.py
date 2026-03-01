#!/usr/bin/env python3
"""
Experiment 2: Feature Importance Analysis (Random Forest)
==========================================================
Visualizes WHAT the model actually looks at.
Plots which MFCC coefficients and time steps drive predictions.
If early time steps dominate → model detects jaw clench ONSET, not speech.
If mid-signal dominates → model detects articulatory PATTERNS.
"""
import os, sys, glob, pickle, warnings, numpy as np, pandas as pd
import scipy.signal, scipy.fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
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
    print("  EXP 2: Feature Importance Analysis")
    print("="*60)

    X,y,le=load_data(DATA_DIR)
    print(f"✅ {X.shape[0]} samples")
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=SEED,stratify=y)

    # Flatten for RF
    X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
    X_te_flat = X_te.reshape(X_te.shape[0], -1)

    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr_flat, y_tr)
    preds = rf.predict(X_te_flat)
    acc = 100 * (preds == y_te).mean()
    print(f"\nRF Accuracy: {acc:.1f}%")
    print(classification_report(y_te, preds, target_names=le.classes_))

    # ── Feature Importance Heatmap ──
    importances = rf.feature_importances_
    # Reshape: (100 timesteps × 26 features) = 2600 features
    imp_map = importances.reshape(TSTEPS, X.shape[2])  # (100, 26)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Full heatmap
    ax = axes[0]
    im = ax.imshow(imp_map.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Time Step (0=signal start, 100=end)')
    ax.set_ylabel('Feature (MFCC coeff × channel)')
    ax.set_title('Feature Importance Heatmap')
    plt.colorbar(im, ax=ax, label='Importance')

    # 2. Importance by time step (summed across features)
    ax = axes[1]
    time_imp = imp_map.sum(axis=1)
    ax.bar(range(TSTEPS), time_imp, color='coral', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Summed Importance')
    ax.set_title('Importance by Time Step')
    # Mark early (onset) vs mid (pattern) vs late (offset)
    ax.axvspan(0, 20, alpha=0.1, color='red', label='Onset (0-20)')
    ax.axvspan(20, 80, alpha=0.1, color='green', label='Pattern (20-80)')
    ax.axvspan(80, 100, alpha=0.1, color='blue', label='Offset (80-100)')
    ax.legend(fontsize=8)

    # 3. Importance by MFCC coefficient (summed across time)
    ax = axes[2]
    feat_imp = imp_map.sum(axis=0)
    n_ch = X.shape[2] // N_MFCC
    colors = ['steelblue'] * N_MFCC + ['coral'] * N_MFCC if n_ch == 2 else ['steelblue'] * X.shape[2]
    ax.barh(range(X.shape[2]), feat_imp, color=colors[:X.shape[2]])
    ax.set_ylabel('Feature Index')
    ax.set_xlabel('Summed Importance')
    ax.set_title('Importance by MFCC Coefficient')
    if n_ch == 2:
        ax.axhline(N_MFCC - 0.5, color='gray', linestyle='--', alpha=0.5)
        ax.text(feat_imp.max() * 0.7, N_MFCC // 2, 'CH1', fontsize=10)
        ax.text(feat_imp.max() * 0.7, N_MFCC + N_MFCC // 2, 'CH2', fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Plot saved to {plot_path}")

    # ── Interpretation ──
    onset_imp = time_imp[:20].sum()
    pattern_imp = time_imp[20:80].sum()
    offset_imp = time_imp[80:].sum()
    total = onset_imp + pattern_imp + offset_imp

    print(f"\n📊 TEMPORAL DISTRIBUTION OF IMPORTANCE:")
    print(f"   Onset  (t=0-20):   {onset_imp/total*100:.1f}%")
    print(f"   Pattern (t=20-80): {pattern_imp/total*100:.1f}%")
    print(f"   Offset (t=80-100): {offset_imp/total*100:.1f}%")

    if onset_imp/total > 0.4:
        print("\n⚠️  CONCERN: Model relies heavily on signal ONSET.")
        print("   It may be detecting jaw clench timing, not speech patterns.")
    elif pattern_imp/total > 0.5:
        print("\n✅  GOOD: Model primarily uses mid-signal articulatory patterns.")
        print("   This suggests it's learning genuine speech features.")
    else:
        print("\n📋  Importance is distributed across the signal.")

    meta = {
        'rf_acc': acc,
        'onset_pct': onset_imp/total*100,
        'pattern_pct': pattern_imp/total*100,
        'offset_pct': offset_imp/total*100,
    }
    with open(os.path.join(RESULTS_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

if __name__ == "__main__": main()
