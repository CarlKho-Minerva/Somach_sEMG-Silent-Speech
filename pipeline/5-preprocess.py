# 5-preprocess.py
# Signal Cleaning: Bandpass Filter + Notch Filter + Normalization
# Part of the AlterEgo ML Pipeline
# UPDATED: Recursive search & Dynamic channel support

import numpy as np
import scipy.signal
import os
import argparse
import glob
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLE_RATE = 250  # Hz (Standardized to 250Hz for AlterEgo replication)
                    # NOTE: ESP32 standard sampling for this project is 250Hz.
                    # This matches 4-curriculum-recorder.py target and 8-realtime-predict.py.
LOW_CUT = 1.0      # Hz (high-pass cutoff to remove DC drift)
HIGH_CUT = 50.0    # Hz (low-pass cutoff to keep EMG band)
NOTCH_FREQ = 60.0  # Hz (power line frequency to kill)

# ==========================================
# FILTER FUNCTIONS
# ==========================================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a

def notch_filter(freq, fs, q=30):
    nyq = 0.5 * fs
    freq = freq / nyq
    b, a = scipy.signal.iirnotch(freq, q)
    return b, a

def apply_filters(data, fs):
    # 1. Bandpass filter
    b_band, a_band = butter_bandpass(LOW_CUT, HIGH_CUT, fs)
    y = scipy.signal.filtfilt(b_band, a_band, data)

    # 2. Notch filter (60Hz power line)
    b_notch, a_notch = notch_filter(NOTCH_FREQ, fs)
    y = scipy.signal.filtfilt(b_notch, a_notch, y)

    # 3. Min-Max Normalize (avoid div by zero)
    val_min = np.min(y)
    val_max = np.max(y)
    if val_max - val_min == 0:
        return y  # Flat signal
    y = (y - val_min) / (val_max - val_min)

    return y

# ==========================================
# MAIN
# ==========================================
def main():
    print("╔════════════════════════════════════════╗")
    print("║      AlterEgo Signal Preprocessor      ║")
    print("║      (Recursive & Dynamic Channels)    ║")
    print("╚════════════════════════════════════════╝")

    parser = argparse.ArgumentParser(description="Clean raw EMG data.")
    parser.add_argument("--input", default="data_collection", help="Input root directory")
    parser.add_argument("--output", default="data_clean", help="Output root directory")
    args = parser.parse_args()

    # Recursive search for all .csv files
    search_path = os.path.join(args.input, "**", "*.csv")
    files = glob.glob(search_path, recursive=True)

    print(f"🔍 Found {len(files)} CSV files in {args.input} (recursive)")

    if len(files) == 0:
        print("❌ No CSV files found. Check your input path.")
        return

    success_count = 0

    for f in files:
        try:
            # Replicate directory structure in output
            rel_path = os.path.relpath(f, args.input)
            out_path = os.path.join(args.output, rel_path)
            out_dir = os.path.dirname(out_path)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            df = pd.read_csv(f)
            clean_df = df.copy()

            # Dynamic Channel Detection: Find all columns starting with "CH"
            ch_cols = [c for c in df.columns if c.startswith('CH')]

            if not ch_cols:
                print(f"  ⚠️  Skipping {rel_path}: No 'CH' columns found.")
                continue

            for ch in ch_cols:
                clean_df[ch] = apply_filters(df[ch].values, SAMPLE_RATE)

            clean_df.to_csv(out_path, index=False)
            print(f"  ✅ Cleaned: {rel_path} ({len(ch_cols)} ch)")
            success_count += 1

        except Exception as e:
            print(f"  ❌ Error processing {f}: {e}")

    print(f"\n✅ Done! Cleaned {success_count}/{len(files)} files.")
    print(f"   Output mirrored to: {args.output}/")

if __name__ == "__main__":
    main()
