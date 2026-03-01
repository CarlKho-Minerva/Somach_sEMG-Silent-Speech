# 6-extract-features.py
# Feature Extraction: Convert cleaned EMG signals to MFCC spectrograms
# Part of the AlterEgo ML Pipeline
# UPDATED: --per-phase flag for curriculum learning evaluation
# UPDATED: Recursive search, Dynamic Labels, N-Channel Support

import numpy as np
import pandas as pd
import librosa
import os
import argparse
import glob
import pickle
from sklearn.preprocessing import LabelEncoder

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLE_RATE = 250   # Hz (Standardized to 250Hz)
HOP_LENGTH = 25      # samples
N_MFCC = 13          # MFCCs per channel
N_FFT = 128
TARGET_TIMESTEPS = 100

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def extract_mfcc(signal, sr):
    signal = signal.astype(float)
    # Extract MFCCs -> Shape: (n_mfcc, time)
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return mfcc.T  # -> (time, n_mfcc)

def pad_or_truncate(features, target_len):
    """Ensure fixed length for CNN input."""
    if len(features) < target_len:
        pad_width = target_len - len(features)
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        features = features[:target_len, :]
    return features

# ==========================================
# MAIN
# ==========================================
def main():
    print("╔════════════════════════════════════════╗")
    print("║      AlterEgo Feature Extractor        ║")
    print("║      (Dynamic Labels & Channels)       ║")
    print("╚════════════════════════════════════════╝")

    parser = argparse.ArgumentParser(description="Extract MFCC features.")
    parser.add_argument("--input", default="data_clean", help="Input directory (clean CSVs)")
    parser.add_argument("--output", default="features", help="Output directory")
    parser.add_argument("--per-phase", action="store_true",
                        help="Output separate X.npy/y.npy per Phase_* subdirectory (for curriculum eval)")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Recursive search
    search_path = os.path.join(args.input, "**", "*.csv")
    files = glob.glob(search_path, recursive=True)

    print(f"🔍 Found {len(files)} CSV files in {args.input} (recursive)")
    if len(files) == 0:
        print("❌ No CSV files found. Run 5-preprocess.py first.")
        return

    if args.per_phase:
        _extract_per_phase(files, args)
    else:
        _extract_flat(files, args)


def _process_file(f, channel_count_ref):
    """Process a single CSV, return (features, label, channel_count) or None."""
    df = pd.read_csv(f)
    ch_cols = [c for c in df.columns if c.startswith('CH')]
    if not ch_cols:
        return None

    if channel_count_ref[0] == 0:
        channel_count_ref[0] = len(ch_cols)
        print(f"ℹ️  Detected {len(ch_cols)} channels: {ch_cols}")
    elif len(ch_cols) != channel_count_ref[0]:
        print(f"⚠️  {os.path.basename(f)}: {len(ch_cols)} ch (expected {channel_count_ref[0]}). Skip.")
        return None

    all_channel_features = []
    for ch in ch_cols:
        sig = df[ch].values
        feats = extract_mfcc(sig, SAMPLE_RATE)
        feats = pad_or_truncate(feats, TARGET_TIMESTEPS)
        all_channel_features.append(feats)

    stacked = np.hstack(all_channel_features)

    if 'Label' in df.columns:
        label = str(df['Label'].iloc[0]).upper().strip()
    else:
        label = os.path.basename(f).split('_')[0].upper()

    return stacked, label


def _save_dataset(dataset, labels_raw, output_dir):
    """Encode labels and save X.npy, y.npy, label_encoder.pkl."""
    if not dataset:
        print(f"   ❌ No valid samples for {output_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    X = np.array(dataset)
    le = LabelEncoder()
    y = le.fit_transform(labels_raw)

    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)

    print(f"   💾 {output_dir}: {X.shape[0]} samples, {list(le.classes_)}")


def _extract_flat(files, args):
    """Original behavior: output one X.npy/y.npy for all files."""
    dataset = []
    labels_raw = []
    ch_ref = [0]

    for f in files:
        try:
            result = _process_file(f, ch_ref)
            if result:
                dataset.append(result[0])
                labels_raw.append(result[1])
        except Exception as e:
            print(f"  ⚠️  Skipping {os.path.basename(f)}: {e}")

    _save_dataset(dataset, labels_raw, args.output)

    if dataset:
        X = np.array(dataset)
        print(f"\n✅ X shape: {X.shape} (Samples, Timesteps, Features)")


def _extract_per_phase(files, args):
    """Per-phase mode: group files by Phase_* directory, output separate features."""
    import re

    # Group files by phase
    phase_files = {}  # phase_name -> [file_paths]
    for f in files:
        # Look for Phase_N_Name in the path
        parts = f.replace("\\", "/").split("/")
        phase = None
        for part in parts:
            if re.match(r"Phase_\d+_", part):
                phase = part
                break

        if phase is None:
            phase = "_ungrouped"
        phase_files.setdefault(phase, []).append(f)

    print(f"\n📂 Found {len(phase_files)} phase groups:")
    for phase, flist in sorted(phase_files.items()):
        print(f"   {phase}: {len(flist)} files")

    ch_ref = [0]

    for phase, flist in sorted(phase_files.items()):
        dataset = []
        labels_raw = []

        for f in flist:
            try:
                result = _process_file(f, ch_ref)
                if result:
                    dataset.append(result[0])
                    labels_raw.append(result[1])
            except Exception as e:
                print(f"  ⚠️  Skipping {os.path.basename(f)}: {e}")

        out_dir = os.path.join(args.output, phase)
        _save_dataset(dataset, labels_raw, out_dir)

    print(f"\n✅ Per-phase features saved to {args.output}/Phase_*/")


if __name__ == "__main__":
    main()
