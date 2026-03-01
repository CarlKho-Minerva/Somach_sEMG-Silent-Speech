#!/usr/bin/env python3
"""
AlterEgo Live Demo — Real-Time Silent Speech Inference
=======================================================
Designed for: Apple M2 Air (8GB) live presentation
Hardware:     ESP32 + 2× AD8232 → USB serial @ 250 Hz
Model:        ~47K param 1D CNN (EMG_CNN)

Usage:
  python live_demo.py
  python live_demo.py --model path/to/model.pth --meta path/to/model_meta.pkl
  python live_demo.py --port /dev/cu.usbserial-0001
  python live_demo.py --no-serial          # demo mode with synthetic data

Keys (during run):
  q / Ctrl+C  — quit
  r           — reset prediction history
  c           — toggle confidence gating (default: ON @ θ=0.60)
"""

import argparse
import os
import sys
import time
import pickle
import signal
import struct
import datetime

import numpy as np
import scipy.signal
import scipy.fft

import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════
# CONSTANTS — must match training pipeline (rigorous_eval.py)
# ══════════════════════════════════════════════════════════════
SAMPLE_RATE       = 250      # Hz
LOW_CUT           = 1.0      # Hz (bandpass low)
HIGH_CUT          = 50.0     # Hz (bandpass high)
NOTCH_FREQ        = 60.0     # Hz (mains hum)
N_MFCC            = 13
N_FFT             = 128      # 512 ms window at 250 Hz
HOP_LENGTH        = 25       # 100 ms hop
N_MELS            = 26
TARGET_TIMESTEPS  = 100      # pad/truncate to this
BUFFER_SIZE       = 500      # 2 seconds of data
SLIDE_AMOUNT      = 125      # slide by 0.5s for ~2 Hz prediction rate
CONFIDENCE_GATE   = 0.60     # θ threshold

# ══════════════════════════════════════════════════════════════
# MODEL DEFINITION (self-contained — no external imports needed)
# ══════════════════════════════════════════════════════════════
class EMG_CNN(nn.Module):
    """1D CNN for EMG classification. Matches utils/model.py exactly."""
    def __init__(self, input_channels, num_classes):
        super(EMG_CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, F) → (B, F, T)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.adaptive_pool(x).squeeze(-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ══════════════════════════════════════════════════════════════
# SIGNAL PROCESSING (matches rigorous_eval.py exactly)
# ══════════════════════════════════════════════════════════════
def butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(order, [low / nyq, high / nyq], btype='band')
    return b, a

def notch_filter(freq, fs, q=30):
    b, a = scipy.signal.iirnotch(freq / (0.5 * fs), q)
    return b, a

# Pre-compute filter coefficients (avoid recomputing every frame)
_BP_B, _BP_A = butter_bandpass(LOW_CUT, HIGH_CUT, SAMPLE_RATE)
_NOTCH_B, _NOTCH_A = notch_filter(NOTCH_FREQ, SAMPLE_RATE)

def preprocess_channel(buffer):
    """Bandpass + notch + min-max normalize a single channel buffer."""
    y = scipy.signal.filtfilt(_BP_B, _BP_A, buffer)
    y = scipy.signal.filtfilt(_NOTCH_B, _NOTCH_A, y)
    drange = np.max(y) - np.min(y)
    if drange == 0:
        return y
    return (y - np.min(y)) / drange


# ── Pure scipy/numpy MFCC (no librosa dependency) ──
def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def _mel_filterbank(sr, n_fft, n_mels=N_MELS):
    fmax = sr / 2.0
    mel_points = np.linspace(_hz_to_mel(0), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        for j in range(left, center):
            if center > left:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                fb[i, j] = (right - j) / (right - center)
    return fb

# Cache mel filterbank
_MEL_FB = _mel_filterbank(SAMPLE_RATE, N_FFT)
_HANN_WINDOW = np.hanning(N_FFT)

def extract_mfcc(signal_data):
    """Extract MFCC features from a single channel. Returns (TARGET_TIMESTEPS, N_MFCC)."""
    signal_data = signal_data.astype(np.float64)
    n_frames = 1 + (len(signal_data) - N_FFT) // HOP_LENGTH
    if n_frames <= 0:
        signal_data = np.pad(signal_data, (0, N_FFT - len(signal_data) + HOP_LENGTH))
        n_frames = 1 + (len(signal_data) - N_FFT) // HOP_LENGTH

    frames = np.zeros((n_frames, N_FFT))
    for i in range(n_frames):
        s = i * HOP_LENGTH
        frames[i] = signal_data[s:s + N_FFT] * _HANN_WINDOW

    fft_r = scipy.fft.rfft(frames, n=N_FFT, axis=1)
    power = np.abs(fft_r) ** 2
    mel_spec = np.log(np.dot(power, _MEL_FB.T) + 1e-10)
    features = scipy.fft.dct(mel_spec, type=2, axis=1, norm='ortho')[:, :N_MFCC]

    # Pad or truncate to TARGET_TIMESTEPS
    if len(features) < TARGET_TIMESTEPS:
        features = np.pad(features,
                          ((0, TARGET_TIMESTEPS - len(features)), (0, 0)),
                          mode='constant')
    else:
        features = features[:TARGET_TIMESTEPS, :]
    return features

def process_multichannel(buffers, n_channels):
    """Process all channels → single feature matrix (TARGET_TIMESTEPS, n_channels * N_MFCC)."""
    all_features = []
    for ch in range(n_channels):
        clean = preprocess_channel(buffers[ch])
        mfcc = extract_mfcc(clean)
        all_features.append(mfcc)
    return np.hstack(all_features)  # (100, 26) for 2 channels


# ══════════════════════════════════════════════════════════════
# SIGNAL QUALITY CHECK
# ══════════════════════════════════════════════════════════════
def check_signal_quality(buffer):
    """Return 'GOOD', 'NOISY', or 'DEAD' based on signal characteristics."""
    if len(buffer) < 50:
        return 'WAIT'
    arr = np.array(buffer[-250:])
    std = np.std(arr)
    ptp = np.ptp(arr)
    if ptp < 10:
        return 'DEAD'
    if std > 800:
        return 'NOISY'
    return 'GOOD'


# ══════════════════════════════════════════════════════════════
# SERIAL PORT DETECTION
# ══════════════════════════════════════════════════════════════
def find_esp32():
    """Auto-detect ESP32 serial port."""
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        keywords = ["SLAB", "USB", "UART", "CP210", "CH340", "ESP"]
        for p in ports:
            desc = (p.description + " " + (p.manufacturer or "")).upper()
            if any(k in desc for k in keywords):
                return p.device
        # Fallback: check common macOS paths
        for path in ["/dev/cu.usbserial-0001", "/dev/cu.SLAB_USBtoUART",
                     "/dev/cu.usbserial-110", "/dev/cu.wchusbserial-110"]:
            if os.path.exists(path):
                return path
    except ImportError:
        pass
    return None


# ══════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR (for --no-serial demo mode)
# ══════════════════════════════════════════════════════════════
class SyntheticSerial:
    """Generates fake 2-channel EMG data for demo mode (no hardware needed)."""
    def __init__(self, n_channels=2):
        self.n_channels = n_channels
        self.t = 0
        self._baseline = 2048
        self._noise_std = 30
        self._burst_prob = 0.003  # probability of onset burst per sample
        self._burst_remaining = [0] * n_channels
        self._burst_class = 0
        self.in_waiting = True  # always has data

    def readline(self):
        vals = []
        for ch in range(self.n_channels):
            base = self._baseline + np.random.normal(0, self._noise_std)
            # Simulate onset burst
            if self._burst_remaining[ch] > 0:
                burst_amp = 200 * (self._burst_remaining[ch] / 20)
                base += burst_amp * (1 + 0.3 * ch)  # different per channel
                self._burst_remaining[ch] -= 1
            elif np.random.random() < self._burst_prob:
                self._burst_remaining[ch] = 20  # ~80ms burst at 250Hz
                self._burst_class = np.random.randint(0, 6)
            base += 50 * np.sin(2 * np.pi * 60 * self.t / SAMPLE_RATE)  # 60 Hz hum
            vals.append(int(np.clip(base, 0, 4095)))
        self.t += 1
        time.sleep(1.0 / SAMPLE_RATE)  # simulate real-time rate
        line = f"{self.t}," + ",".join(str(v) for v in vals) + "\n"
        return line.encode('utf-8')

    def close(self):
        pass


# ══════════════════════════════════════════════════════════════
# TERMINAL UI
# ══════════════════════════════════════════════════════════════
# Colors
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_DIM    = "\033[2m"
C_RED    = "\033[91m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE   = "\033[94m"
C_CYAN   = "\033[96m"
C_WHITE  = "\033[97m"
C_GREY   = "\033[90m"
C_BG_GREEN  = "\033[42m"
C_BG_RED    = "\033[41m"
C_BG_YELLOW = "\033[43m"

# Direction arrows for visual feedback
ARROWS = {
    "UP":      "  ⬆️   UP  ",
    "DOWN":    "  ⬇️  DOWN ",
    "LEFT":    "  ⬅️  LEFT ",
    "RIGHT":   "  ➡️  RIGHT",
    "SILENCE": "  ⏸️  QUIET",
    "NOISE":   "  🔊  NOISE",
}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def confidence_bar(conf, width=30):
    """Render a colored confidence bar."""
    filled = int(conf / 100 * width)
    if conf >= 80:
        color = C_GREEN
    elif conf >= 60:
        color = C_YELLOW
    else:
        color = C_RED
    bar = color + "█" * filled + C_GREY + "░" * (width - filled) + C_RESET
    return bar

def quality_indicator(status):
    if status == 'GOOD':
        return f"{C_GREEN}● GOOD{C_RESET}"
    elif status == 'NOISY':
        return f"{C_YELLOW}● NOISY{C_RESET}"
    elif status == 'DEAD':
        return f"{C_RED}● DEAD{C_RESET}"
    return f"{C_GREY}○ WAIT{C_RESET}"

def render_dashboard(prediction, confidence, probs, quality_status,
                     history, n_channels, gating_on, latency_ms,
                     samples_collected, fps):
    """Render the full terminal dashboard."""
    clear_screen()

    # Header
    print(f"\n  {C_BOLD}{C_CYAN}╔══════════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_BOLD}{C_CYAN}║     ⚡ AlterEgo Silent Speech — Live Demo ⚡        ║{C_RESET}")
    print(f"  {C_BOLD}{C_CYAN}╚══════════════════════════════════════════════════════╝{C_RESET}")

    # Status bar
    gate_str = f"{C_GREEN}ON (θ={CONFIDENCE_GATE:.2f}){C_RESET}" if gating_on else f"{C_RED}OFF{C_RESET}"
    print(f"\n  {C_DIM}Gating: {gate_str}  │  Latency: {C_WHITE}{latency_ms:.1f}ms{C_RESET}  │  "
          f"{C_DIM}Buffer: {samples_collected}/{BUFFER_SIZE}  │  ~{fps:.1f} pred/s{C_RESET}")

    # Signal quality
    qual_str = "  Signal: "
    for ch in range(n_channels):
        qual_str += f" CH{ch+1} {quality_indicator(quality_status[ch])}"
    print(qual_str)

    # ── Main prediction ──
    print(f"\n  {C_DIM}{'─' * 54}{C_RESET}")

    if prediction == "REJECTED":
        print(f"\n    {C_BG_YELLOW}{C_BOLD}  🚫 LOW CONFIDENCE — REJECTED  {C_RESET}")
        print(f"    {C_DIM}Confidence {confidence:.1f}% < threshold {CONFIDENCE_GATE*100:.0f}%{C_RESET}")
    elif prediction == "WAITING":
        print(f"\n    {C_DIM}  ⏳ Collecting data...{C_RESET}")
    else:
        arrow = ARROWS.get(prediction, f"  ❓ {prediction}")
        if confidence >= 70:
            bg = C_BG_GREEN
        elif confidence >= 50:
            bg = C_BG_YELLOW
        else:
            bg = C_BG_RED
        print(f"\n    {bg}{C_BOLD}{C_WHITE} {arrow} {C_RESET}  {C_BOLD}{confidence:.1f}%{C_RESET}")
        print(f"    {confidence_bar(confidence)}")

    # ── Class probabilities ──
    print(f"\n  {C_DIM}{'─' * 54}{C_RESET}")
    print(f"  {C_BOLD}Class Probabilities:{C_RESET}")
    if probs:
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs:
            marker = " ◀" if label == prediction and prediction != "REJECTED" else ""
            bar_width = int(prob / 100 * 25)
            if label == prediction and prediction != "REJECTED":
                color = C_GREEN
            else:
                color = C_GREY
            bar = color + "█" * bar_width + C_RESET
            print(f"    {label:>8s} {prob:5.1f}% {bar}{marker}")

    # ── Prediction history ──
    print(f"\n  {C_DIM}{'─' * 54}{C_RESET}")
    print(f"  {C_BOLD}History (last 8):{C_RESET}")
    if history:
        for ts, lbl, conf in history[-8:]:
            icon = ARROWS.get(lbl, f"  {lbl}").strip()
            if conf >= 60:
                c = C_GREEN
            elif conf >= 40:
                c = C_YELLOW
            else:
                c = C_RED
            print(f"    {C_DIM}{ts}{C_RESET}  {c}{icon:>10s}  {conf:5.1f}%{C_RESET}")
    else:
        print(f"    {C_DIM}No predictions yet...{C_RESET}")

    # Footer
    print(f"\n  {C_DIM}{'─' * 54}{C_RESET}")
    print(f"  {C_DIM}[q] quit  [r] reset history  [c] toggle gating{C_RESET}\n")


# ══════════════════════════════════════════════════════════════
# KEYBOARD INPUT (non-blocking)
# ══════════════════════════════════════════════════════════════
def setup_keyboard():
    """Set up non-blocking keyboard input on macOS/Linux."""
    import termios, tty
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    return old_settings

def restore_keyboard(old_settings):
    import termios
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def check_keypress():
    """Return pressed key or None (non-blocking)."""
    import select
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="AlterEgo Live Demo")
    parser.add_argument("--model", default="models/model.pth",
                        help="Path to trained model.pth")
    parser.add_argument("--meta", default="models/model_meta.pkl",
                        help="Path to model_meta.pkl")
    parser.add_argument("--port", default=None,
                        help="Serial port (auto-detected if omitted)")
    parser.add_argument("--no-serial", action="store_true",
                        help="Run with synthetic data (no hardware)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "mps", "cpu"],
                        help="PyTorch device (default: auto → MPS if available)")
    args = parser.parse_args()

    clear_screen()

    # ─── Device selection ───
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"\n  {C_BOLD}{C_CYAN}AlterEgo Live Demo{C_RESET}")
    print(f"  {C_DIM}{'─' * 40}{C_RESET}")
    print(f"  Device: {C_WHITE}{device}{C_RESET}")

    # ─── Load metadata ───
    print(f"  {C_DIM}Loading metadata: {args.meta}{C_RESET}")
    try:
        with open(args.meta, "rb") as f:
            meta = pickle.load(f)
        input_channels = meta['input_channels']
        num_classes = meta['num_classes']
        LABELS = meta['classes']
        n_physical_channels = input_channels // N_MFCC
        print(f"  {C_GREEN}✓{C_RESET} {n_physical_channels} channels, {num_classes} classes: {LABELS}")
    except FileNotFoundError:
        print(f"  {C_RED}✗ Metadata file not found: {args.meta}{C_RESET}")
        print(f"  {C_DIM}  Train a model first with 7-train-model.py{C_RESET}")
        return
    except Exception as e:
        print(f"  {C_RED}✗ Failed to load metadata: {e}{C_RESET}")
        return

    # ─── Load model ───
    print(f"  {C_DIM}Loading model: {args.model}{C_RESET}")
    try:
        model = EMG_CNN(input_channels=input_channels, num_classes=num_classes)
        state_dict = torch.load(args.model, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Warm up: run one dummy inference to compile MPS kernels
        dummy = torch.zeros(1, TARGET_TIMESTEPS, input_channels, device=device)
        with torch.no_grad():
            _ = model(dummy)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  {C_GREEN}✓{C_RESET} Model loaded ({param_count:,} params)")
    except FileNotFoundError:
        print(f"  {C_RED}✗ Model file not found: {args.model}{C_RESET}")
        return
    except Exception as e:
        print(f"  {C_RED}✗ Failed to load model: {e}{C_RESET}")
        return

    # ─── Connect serial / synthetic ───
    if args.no_serial:
        print(f"  {C_YELLOW}⚠ Demo mode (synthetic data){C_RESET}")
        ser = SyntheticSerial(n_channels=n_physical_channels)
    else:
        port = args.port or find_esp32()
        if not port:
            print(f"  {C_RED}✗ No ESP32 detected.{C_RESET}")
            print(f"  {C_DIM}  Use --port /dev/cu.xxx or --no-serial for demo mode{C_RESET}")
            return
        print(f"  {C_DIM}Connecting to {port}...{C_RESET}")
        try:
            import serial
            ser = serial.Serial(port, 115200, timeout=1)
            time.sleep(2)  # ESP32 reset delay
            ser.flushInput()
            print(f"  {C_GREEN}✓{C_RESET} Connected to {port}")
        except Exception as e:
            print(f"  {C_RED}✗ Serial connection failed: {e}{C_RESET}")
            return

    print(f"\n  {C_GREEN}Starting in 2 seconds...{C_RESET}")
    time.sleep(2)

    # ─── Dashboard loop ───
    buffers = [[] for _ in range(n_physical_channels)]
    quality_status = ["WAIT"] * n_physical_channels
    prediction = "WAITING"
    confidence = 0.0
    probs_dict = {}
    history = []
    gating_on = True
    last_render = 0
    pred_count = 0
    pred_start_time = time.time()

    old_kb = setup_keyboard()

    try:
        while True:
            # ── Check keyboard ──
            key = check_keypress()
            if key == 'q':
                break
            elif key == 'r':
                history.clear()
                pred_count = 0
                pred_start_time = time.time()
            elif key == 'c':
                gating_on = not gating_on

            # ── Read serial ──
            if hasattr(ser, 'in_waiting') and not ser.in_waiting:
                time.sleep(0.001)
                continue

            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
            except Exception:
                continue

            parts = line.split(',')
            if len(parts) != 1 + n_physical_channels:
                continue
            if not parts[0].isdigit():
                continue

            # Parse channel values
            for ch in range(n_physical_channels):
                try:
                    val = int(parts[ch + 1])
                    buffers[ch].append(val)
                except (ValueError, IndexError):
                    pass

            # ── Signal quality (every 50 samples) ──
            if len(buffers[0]) % 50 == 0 and len(buffers[0]) > 0:
                quality_status = [check_signal_quality(b) for b in buffers]

            # ── Predict when buffer is full ──
            if len(buffers[0]) >= BUFFER_SIZE:
                buffer_arrays = [np.array(b[:BUFFER_SIZE], dtype=np.float32) for b in buffers]
                features = process_multichannel(buffer_arrays, n_physical_channels)

                # Inference
                t0 = time.perf_counter()
                with torch.no_grad():
                    x = torch.FloatTensor(features).unsqueeze(0).to(device)
                    output = model(x)
                    probs = torch.softmax(output, dim=1)
                    conf_val, predicted_idx = torch.max(probs, 1)
                latency_ms = (time.perf_counter() - t0) * 1000

                label = LABELS[predicted_idx.item()]
                conf = conf_val.item() * 100
                probs_dict = {LABELS[i]: probs[0][i].item() * 100 for i in range(len(LABELS))}

                # Apply confidence gating
                if gating_on and conf < CONFIDENCE_GATE * 100:
                    prediction = "REJECTED"
                    confidence = conf
                else:
                    prediction = label
                    confidence = conf

                ts = datetime.datetime.now().strftime("%H:%M:%S")
                history.append((ts, prediction, conf))
                pred_count += 1

                # Slide buffer
                for ch in range(n_physical_channels):
                    buffers[ch] = buffers[ch][SLIDE_AMOUNT:]

            # ── Update display (throttled ~4 Hz) ──
            now = time.time()
            if now - last_render > 0.25:
                elapsed = now - pred_start_time
                fps = pred_count / elapsed if elapsed > 0 else 0
                latency = latency_ms if 'latency_ms' in dir() else 0
                render_dashboard(
                    prediction, confidence, probs_dict, quality_status,
                    history, n_physical_channels, gating_on,
                    latency_ms if 'latency_ms' in locals() else 0.0,
                    len(buffers[0]), fps
                )
                last_render = now

    except KeyboardInterrupt:
        pass
    finally:
        restore_keyboard(old_kb)
        ser.close()
        print(f"\n  {C_DIM}Session ended. {pred_count} predictions in "
              f"{time.time() - pred_start_time:.1f}s{C_RESET}\n")


if __name__ == "__main__":
    main()
