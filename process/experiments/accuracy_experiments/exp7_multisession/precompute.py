#!/usr/bin/env python3
"""Stage 1: Pre-compute MFCC features from all 3 sessions and cache as .npy"""
import os, sys, glob, warnings, numpy as np, pandas as pd, pickle
import scipy.signal, scipy.fft
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA = os.path.join(SCRIPT_DIR, "..", "..", "..", "python", "data_collection")
SESSION_A = os.path.join(BASE_DATA, "carl", "Phase_6_Covert")
SESSION_B = os.path.join(BASE_DATA, "carl_022526_sessionb", "Phase_6_Covert")
SESSION_C = os.path.join(BASE_DATA, "carl_sessionc2", "Phase_6_Covert")

SR=250; LOW=1.0; HIGH=50.0; NOTCH=60.0; N_MFCC=13; N_FFT=128; HOP=25
N_MELS=26; TSTEPS=100

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
_FB=_melfb(SR,N_FFT,N_MELS)
def extract_mfcc(sig):
    sig=sig.astype(np.float64)
    nf=1+(len(sig)-N_FFT)//HOP
    if nf<=0: sig=np.pad(sig,(0,N_FFT-len(sig)+HOP)); nf=1+(len(sig)-N_FFT)//HOP
    w=np.hanning(N_FFT); frames=np.zeros((nf,N_FFT))
    for i in range(nf): frames[i]=sig[i*HOP:i*HOP+N_FFT]*w
    pwr=np.abs(scipy.fft.rfft(frames,n=N_FFT,axis=1))**2
    mel=np.log(np.dot(pwr,_FB.T)+1e-10)
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
    X_list, labels, errs = [], [], 0
    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f)
            chs = [c for c in df.columns if c.startswith('CH')]
            if not chs: errs += 1; continue
            label = os.path.basename(f).split('_')[0].upper()
            feats = [extract_mfcc(filters(df[ch].values)) for ch in chs]
            X_list.append(np.hstack(feats)); labels.append(label)
        except: errs += 1
        if (i+1) % 100 == 0: print(f"   {tag}: {i+1}/{len(files)}")
    print(f"   {tag}: {len(labels)} loaded ({errs} errors)")
    return np.array(X_list), labels

print("Pre-computing MFCCs for all 3 sessions...")
Xa, la = load_session(SESSION_A, "A")
Xb, lb = load_session(SESSION_B, "B")
Xc, lc = load_session(SESSION_C, "C")

cache = {"Xa": Xa, "la": la, "Xb": Xb, "lb": lb, "Xc": Xc, "lc": lc}
cache_path = os.path.join(SCRIPT_DIR, "cache.pkl")
with open(cache_path, "wb") as f:
    pickle.dump(cache, f)
print(f"\n✅ Cached to {cache_path}")
print(f"   A: {Xa.shape} | B: {Xb.shape} | C: {Xc.shape}")
print(f"   Total: {len(la)+len(lb)+len(lc)} samples")
