# 8-realtime-predict.py
# Real-Time Inference: Live silent speech recognition from ESP32
# Part of the AlterEgo ML Pipeline
# UPDATED: Clean Dashboard UI, Signal Quality Check, Dynamic Architecture

import torch
import torch.nn as nn
import numpy as np
import serial
import serial.tools.list_ports
import time
import argparse
import scipy.signal
import scipy.fft
import pickle
import os
import sys

# ==========================================
# CONFIGURATION & THEME
# ==========================================
SAMPLE_RATE = 250       # Hz (AlterEgo standard)
BUFFER_SIZE = 250 * 2   # 2 seconds of data
N_MFCC = 13             # MFCCs per channel
N_FFT = 128
HOP_LENGTH = 25
TARGET_TIMESTEPS = 100

# Filter parameters
LOW_CUT = 1.0
HIGH_CUT = 50.0
NOTCH_FREQ = 60.0

# Colors for Dashboard
C_RESET  = "\033[0m"
C_ORANGE = "\033[38;5;208m"
C_GREY   = "\033[38;5;240m"
C_WHITE  = "\033[37m"
C_RED    = "\033[31m"
C_GREEN  = "\033[32m"
C_BOLD   = "\033[1m"
C_CYAN   = "\033[36m"

# ==========================================
# UI HELPERS
# ==========================================
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_box(title, subtitle=""):
    print(f"{C_GREY}╔{'═'*78}╗{C_RESET}")
    print(f"{C_GREY}║ {C_ORANGE}{C_BOLD}{title.center(76)}{C_RESET} {C_GREY}║{C_RESET}")
    if subtitle:
        print(f"{C_GREY}║ {C_GREY}{subtitle.center(76)}{C_RESET} {C_GREY}║{C_RESET}")
    print(f"{C_GREY}╚{'═'*78}╝{C_RESET}")

def print_dashboard(pred_label, conf, all_probs_dict, channels, quality_status, history, n_channels):
    """
    Full-screen dashboard redraw with Multi-Class Confidence Bars.
    """
    clear_screen()
    draw_box("LIVE INFERENCE (Study B)", f"{n_channels} Channels • Ctrl+C to Stop")

    # 1. Prediction (Big & Bold)
    # Silence Rejection: If confidence < 60%, show UNCERTAIN
    if conf < 60.0 and pred_label not in ['SILENCE', 'NOISE']:
        display_label = "UNCERTAIN"
        lbl_color = C_ORANGE
    elif pred_label in ['SILENCE', 'NOISE']:
        display_label = pred_label
        lbl_color = C_GREY
    else:
        display_label = pred_label
        lbl_color = C_GREEN + C_BOLD

    bar_len = int((conf / 100) * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    print(f"  {C_WHITE}PREDICTION:{C_RESET}  {lbl_color}{display_label:<10}{C_RESET}  {C_GREEN}{bar}{C_RESET}  {conf:>5.1f}%")

    # 2. Detailed Class Confidence (Prof. Watson's Table)
    print(f"\n  {C_WHITE}CLASS CONFIDENCE:{C_RESET}")
    # Sort by confidence, descending
    sorted_probs = sorted(all_probs_dict.items(), key=lambda x: x[1], reverse=True)

    for lbl, prob in sorted_probs:
        # Skip low probability noise/silence to keep UI clean, unless they are the top prediction
        if prob < 5.0 and lbl != pred_label: continue

        c_bar_len = int((prob / 100) * 40)
        c_bar = "█" * c_bar_len + " " * (40 - c_bar_len)

        # Color coding
        if lbl == pred_label:
            row_color = C_GREEN if lbl not in ['SILENCE', 'NOISE'] else C_GREY
            pointer = "◄"
        else:
            row_color = C_GREY
            pointer = " "

        print(f"    {row_color}{lbl:<10} │{c_bar}│ {prob:>5.1f}% {pointer}{C_RESET}")

    # 3. Channel values + quality
    print(f"\n  {C_WHITE}SIGNAL RAW:{C_RESET}")
    ch_parts = []
    for idx, val in enumerate(channels):
        status = quality_status[idx] if idx < len(quality_status) else "WAIT"
        if status == "DEAD": c_val = C_RED
        elif status == "NOISY": c_val = C_ORANGE
        else: c_val = C_CYAN
        ch_parts.append(f"CH{idx+1}: {c_val}{val:>5}{C_RESET}")
    print(f"    {'   '.join(ch_parts)}")

    # 4. Status
    print(f"    {get_quality_msg(quality_status)}")

    # 5. History
    print(f"\n  {C_WHITE}HISTORY:{C_RESET}")
    for ts, lbl, c in history[-3:]: # Show last 3
        if lbl in ['SILENCE', 'NOISE']: hc = C_GREY
        else: hc = C_GREEN
        print(f"    {C_GREY}{ts}{C_RESET}  {hc}{lbl:<10}{C_RESET} {c:.0f}%")

    sys.stdout.flush()

def get_quality_msg(statuses):
    if "DEAD" in statuses:
        return f"{C_RED}⚠️  CHECK ELECTRODE CONNECTION (Flatline detected){C_RESET}"
    elif "NOISY" in statuses:
        return f"{C_ORANGE}⚠️  HIGH NOISE detected{C_RESET}"
    else:
        return f"{C_GREEN}✅  SIGNAL GOOD{C_RESET}"

def check_signal_quality(buffer):
    """Simple heuristic to check if signal is dead or noisy."""
    if len(buffer) < 50: return "WAIT"
    rng = np.max(buffer) - np.min(buffer)
    if rng < 50: return "DEAD"      # Flatline (disconnected)
    if rng > 3500: return "NOISY"   # Railing (loose connection)
    return "GOOD"

# ==========================================
# PROCESSING & MODEL
# ==========================================
from utils.model import EMG_CNN  # Shared model (adaptive pooling, N-channel)

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

def preprocess_channel(buffer):
    b_band, a_band = butter_bandpass(LOW_CUT, HIGH_CUT, SAMPLE_RATE)
    y = scipy.signal.filtfilt(b_band, a_band, buffer)
    b_notch, a_notch = notch_filter(NOTCH_FREQ, SAMPLE_RATE)
    y = scipy.signal.filtfilt(b_notch, a_notch, y)
    if np.max(y) - np.min(y) == 0: return y
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    return y

# ── Pure scipy/numpy MFCC (replaces librosa) ──
def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def _mel_filterbank(sr, n_fft, n_mels=26):
    fmax = sr / 2.0
    mel_points = np.linspace(_hz_to_mel(0), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i+1], bin_points[i+2]
        for j in range(left, center):
            if center > left: fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center: fb[i, j] = (right - j) / (right - center)
    return fb

_MEL_FB_CACHE = None
def _get_mel_fb():
    global _MEL_FB_CACHE
    if _MEL_FB_CACHE is None:
        _MEL_FB_CACHE = _mel_filterbank(SAMPLE_RATE, N_FFT)
    return _MEL_FB_CACHE

def extract_mfcc(signal):
    signal = signal.astype(np.float64)
    n_frames = 1 + (len(signal) - N_FFT) // HOP_LENGTH
    if n_frames <= 0:
        signal = np.pad(signal, (0, N_FFT - len(signal) + HOP_LENGTH))
        n_frames = 1 + (len(signal) - N_FFT) // HOP_LENGTH
    window = np.hanning(N_FFT)
    frames = np.zeros((n_frames, N_FFT))
    for i in range(n_frames):
        s = i * HOP_LENGTH
        frames[i] = signal[s:s + N_FFT] * window
    fft_r = scipy.fft.rfft(frames, n=N_FFT, axis=1)
    power = np.abs(fft_r) ** 2
    mel_fb = _get_mel_fb()
    mel_spec = np.log(np.dot(power, mel_fb.T) + 1e-10)
    features = scipy.fft.dct(mel_spec, type=2, axis=1, norm='ortho')[:, :N_MFCC]
    if len(features) < TARGET_TIMESTEPS:
        features = np.pad(features, ((0, TARGET_TIMESTEPS - len(features)), (0, 0)), mode='constant')
    else:
        features = features[:TARGET_TIMESTEPS, :]
    return features

def process_multichannel(buffers, n_channels):
    all_features = []
    for ch in range(n_channels):
        clean = preprocess_channel(buffers[ch])
        mfcc = extract_mfcc(clean)
        all_features.append(mfcc)
    return np.hstack(all_features)

def find_esp32():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(x in p.description.upper() for x in ["SLAB", "USB", "UART", "CP210", "CH340"]):
            return p.device
    return None

# ==========================================
# MAIN
# ==========================================
def main():
    clear_screen()
    draw_box("AlterEgo Real-Time Predictor", "Dynamic Multi-Channel • Live Inference")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.pth")
    parser.add_argument("--meta", default="models/model_meta.pkl")
    args = parser.parse_args()

    # 1. Load Metadata
    print(f"\n{C_GREY}   [INIT] LOADING METADATA... {args.meta}{C_RESET}")
    try:
        with open(args.meta, "rb") as f:
            meta = pickle.load(f)
        input_channels = meta['input_channels']
        num_classes = meta['num_classes']
        LABELS = meta['classes']
        n_physical_channels = input_channels // N_MFCC
        print(f"   [OK]   {n_physical_channels} CHANNELS • {num_classes} CLASSES: {LABELS}")
    except Exception as e:
        print(f"   {C_RED}[ERR]  FAILED: {e}{C_RESET}")
        return

    # 2. Load Model
    print(f"{C_GREY}   [INIT] LOADING MODEL...    {args.model}{C_RESET}")
    try:
        model = EMG_CNN(input_channels=input_channels, num_classes=num_classes)
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
        model.eval()
        print(f"   [OK]   MODEL LOADED")
    except Exception as e:
        print(f"   {C_RED}[ERR]  MODEL FAILED: {e}{C_RESET}")
        return

    # 3. Connect ESP32
    print(f"{C_GREY}   [INIT] SEARCHING DEVICE... {C_RESET}")
    port = find_esp32()
    if not port:
        print(f"   {C_RED}[ERR]  DEVICE NOT FOUND{C_RESET}")
        port = input(f"   {C_ORANGE}Enter port manually > {C_RESET}").strip()
        if not port: return

    print(f"   [SYS]  CONNECTING TO {port}...")
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)
    except Exception as e:
        print(f"   {C_RED}[ERR]  CONNECTION FAILED: {e}{C_RESET}")
        return

    # 4. Dashboard Loop
    buffers = [[] for _ in range(n_physical_channels)]
    last_pred = "WAITING"
    last_conf = 0.0
    quality_status = ["WAIT"] * n_physical_channels
    pred_history = []  # [(timestamp, label, confidence)]

    try:
        while True:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                parts = line.split(',')

                # Parse
                current_vals = []
                if len(parts) == 1 + n_physical_channels and parts[0].isdigit():
                    for ch in range(n_physical_channels):
                        try:
                            val = int(parts[ch + 1])
                            buffers[ch].append(val)
                            current_vals.append(val)
                        except:
                            current_vals.append(0)

                    # Update Quality Status (every 50 samples)
                    if len(buffers[0]) % 50 == 0:
                        quality_status = [check_signal_quality(b[-250:]) for b in buffers]

                    # Predict when full
                    if len(buffers[0]) >= BUFFER_SIZE:
                        # Process
                        buffer_arrays = [np.array(b[:BUFFER_SIZE]) for b in buffers]
                        features = process_multichannel(buffer_arrays, n_physical_channels)

                        # Inference
                        with torch.no_grad():
                            x = torch.FloatTensor(features).unsqueeze(0)
                            output = model(x)
                            probs = torch.softmax(output, dim=1)
                            confidence, predicted = torch.max(probs, 1)

                            label = LABELS[predicted.item()]
                            conf = confidence.item() * 100

                            # Create probs dict
                            probs_dict = {LABELS[i]: probs[0][i].item() * 100 for i in range(len(LABELS))}

                            last_pred = label
                            last_conf = conf

                            # Add to history
                            import datetime
                            ts = datetime.datetime.now().strftime("%H:%M:%S")
                            pred_history.append((ts, label, conf))
                            pred_history = pred_history[-5:]  # Keep last 5

                        # Slide Buffers
                        # NEW: Slide by 1/4 buffer (0.5s) for faster response
                        slide_amount = BUFFER_SIZE // 4
                        for ch in range(n_physical_channels):
                            buffers[ch] = buffers[ch][slide_amount:]

                    # Update Dashboard (throttled)
                    if len(buffers[0]) % 20 == 0 and current_vals: # Faster refresh
                        # Use last known probs if we haven't predicted yet this cycle
                        current_probs = probs_dict if 'probs_dict' in locals() else {}
                        if not current_probs and 'LABELS' in locals():
                             current_probs = {l: 0.0 for l in LABELS}

                        print_dashboard(
                            last_pred, last_conf, current_probs, current_vals,
                            quality_status, pred_history, n_physical_channels
                        )

    except KeyboardInterrupt:
        print(f"\n\n{C_GREY}   [SYS] STOPPED.{C_RESET}\n")
        ser.close()

if __name__ == "__main__":
    main()
