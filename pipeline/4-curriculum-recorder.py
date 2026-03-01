#!/usr/bin/env python3
"""
OpenEMG Terminal Recorder v4
============================
High-performance, zero-latency EMG data collection tool.
Features continuous threaded capture, dynamic channel detection,
live signal dashboard, and the complete 6-phase curriculum.

Usage:
  python3 4-curriculum-recorder.py

Requirements:
  pip install pyserial pynput

macOS Note:
  Grant Terminal accessibility permissions if press-and-hold fails.
"""

import serial
import serial.tools.list_ports
import time
import os
import csv
import random
import sys
import threading
from collections import deque
from datetime import datetime

# Try to import pynput
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

# ==========================================
# CONFIGURATION & THEME
# ==========================================
BAUD_RATE = 115200
DATA_DIR = "data_collection"
MIN_SAMPLES = 25   # Minimum valid sample size
TARGET_SAMPLES = 250 # samples for "Green Light" visual feedback
BUFFER_SIZE = 10000

# ANSI Colors for "Departure" Aesthetic (Orange/Grey/Black)
C_RESET  = "\033[0m"
C_ORANGE = "\033[38;5;208m"
C_GREY   = "\033[38;5;240m"
C_WHITE  = "\033[37m"
C_RED    = "\033[31m"
C_GREEN  = "\033[32m"
C_BOLD   = "\033[1m"
C_CYAN   = "\033[36m"

# ==========================================
# NOISE CLASS: Randomized Non-Command Actions
# ==========================================
# 100 diverse movements to prevent the model from overfitting
# to a single "noise" pattern (e.g., always head-shaking).
# Each NOISE sample shows a random prompt from this list.
NOISE_PROMPTS = [
    # -- FACIAL (20) --
    "Raise your eyebrows",
    "Scrunch your nose",
    "Puff out your cheeks",
    "Bite your lower lip",
    "Lick your lips slowly",
    "Flare your nostrils",
    "Wink your left eye",
    "Wink your right eye",
    "Squint hard",
    "Open your mouth wide (yawn)",
    "Clench your teeth",
    "Smile wide",
    "Frown deeply",
    "Purse your lips",
    "Blow air out (no sound)",
    "Suck in your cheeks",
    "Wiggle your nose",
    "Chew imaginary gum",
    "Fake sneeze motion",
    "Move your jaw side-to-side",
    # -- JAW & CHIN (15) --
    "Scratch your chin",
    "Rub your jawline",
    "Open and close mouth 3x",
    "Jut your jaw forward",
    "Touch chin to chest",
    "Clench and release jaw",
    "Pop your jaw left",
    "Pop your jaw right",
    "Grind teeth gently",
    "Press lips together tight",
    "Pout like a duck",
    "Drop jaw fully open",
    "Bite the inside of your cheek",
    "Run tongue over front teeth",
    "Press tongue behind lower teeth",
    # -- HEAD MOVEMENTS (15) --
    "Nod yes slowly",
    "Shake head no",
    "Tilt head to left shoulder",
    "Tilt head to right shoulder",
    "Look up at ceiling",
    "Look down at floor",
    "Turn head left",
    "Turn head right",
    "Roll head in a circle",
    "Quick head shake (like saying 'nah')",
    "Double chin pose",
    "Extend neck forward (turtle)",
    "Pull chin back (reverse turtle)",
    "Tilt head back slightly",
    "Bob head up and down",
    # -- THROAT & NECK (15) --
    "Swallow",
    "Clear your throat (silent)",
    "Gulp hard",
    "Hum quietly",
    "Tense your neck muscles",
    "Touch your Adam's apple",
    "Stretch neck to the left",
    "Stretch neck to the right",
    "Massage under your jaw",
    "Yawn (real or fake)",
    "Cough silently",
    "Pretend to whistle (no sound)",
    "Sigh deeply",
    "Flex throat muscles",
    "Tighten and release neck",
    # -- BREATHING (10) --
    "Deep breath in through nose",
    "Exhale slowly through mouth",
    "Three quick sniffs",
    "Hold breath 2 seconds",
    "Belly breathe",
    "Shallow quick breaths 3x",
    "Exhale through nose forcefully",
    "Breathe in, hold, release",
    "Pant like a dog (no sound)",
    "Slow breath cycle",
    # -- RANDOM BODY (15) --
    "Shrug your shoulders",
    "Roll your shoulders",
    "Scratch behind your ear",
    "Adjust your glasses/headset",
    "Rub your forehead",
    "Touch your earlobe",
    "Drum fingers on desk",
    "Cross and uncross arms",
    "Stretch your arms up",
    "Crack your knuckles",
    "Fidget with something",
    "Lean back in chair",
    "Shift sitting position",
    "Look around the room",
    "Blink rapidly 5 times",
    # -- SPEECH-LIKE (non-command) (10) --
    "Mouth the word 'HELLO'",
    "Mouth the word 'BANANA'",
    "Mouth the number 'SEVEN'",
    "Mouth the word 'WATER'",
    "Mouth 'ABCDE' silently",
    "Mouth 'SHHH'",
    "Mouth 'PIZZA'",
    "Mouth 'COMPUTER'",
    "Mouth 'GOODBYE'",
    "Mouth 'ELEPHANT'",
]

def print_styled(msg, color=C_WHITE, bold=False):
    prefix = C_BOLD if bold else ""
    print(f"{prefix}{color}{msg}{C_RESET}")

# ==========================================
# THREADED SERIAL READER (ZERO LATENCY)
# ==========================================
class SerialReader:
    def __init__(self, port, baud_rate):
        self.serial = serial.Serial(port, baud_rate, timeout=1)
        self.running = False
        self.thread = None
        self.data_queue = deque(maxlen=BUFFER_SIZE)
        self.lock = threading.Lock()
        self.channels = []
        self.sample_rate_estimate = 0
        self.latest_sample = [] # For dashboard

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        time.sleep(1.0)
        self._detect_channels()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.serial.is_open:
            self.serial.close()

    def _read_loop(self):
        packet_count = 0
        start_t = time.time()

        while self.running:
            try:
                if self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 2 and parts[0].isdigit():
                            with self.lock:
                                self.data_queue.append(parts)
                                self.latest_sample = parts # Update dashboard cache
                            packet_count += 1

                            # Estimate sample rate every second
                            now = time.time()
                            if now - start_t >= 1.0:
                                self.sample_rate_estimate = packet_count
                                packet_count = 0
                                start_t = now
                else:
                    time.sleep(0.0001) # Ultra-fast poll
            except Exception:
                time.sleep(0.001)  # Brief pause on error, don't kill thread
                continue

    def _detect_channels(self):
        """Auto-detect number of channels from incoming data stream."""
        print_styled("   [SYS] DETECTING DATA STREAM...", C_GREY)
        attempts = 0
        while attempts < 20:
            with self.lock:
                if len(self.data_queue) > 0:
                    sample = self.data_queue[-1]
                    num_cols = len(sample)
                    if num_cols >= 2:
                        self.channels = ["Timestamp"] + [f"CH{i}" for i in range(1, num_cols)]
                        print_styled(f"   [OK]  DETECTED {num_cols-1} CHANNELS", C_GREEN)
                        return
            time.sleep(0.1)
            attempts += 1

        # Fallback: prompt user for channel count instead of assuming
        print_styled("   [WARN] NO DATA STREAM DETECTED.", C_ORANGE)
        n_input = input(f"{C_ORANGE}   CHANNELS? {C_GREY}(default: 2) > {C_RESET}").strip()
        n_ch = int(n_input) if n_input.isdigit() and int(n_input) > 0 else 2
        self.channels = ["Timestamp"] + [f"CH{i}" for i in range(1, n_ch + 1)]
        print_styled(f"   [SYS] SET TO {n_ch} CHANNELS (manual)", C_GREY)

    def get_latest_data(self, clear=True):
        with self.lock:
            data = list(self.data_queue)
            if clear:
                self.data_queue.clear()
            return data

    def flush_all(self):
        """Clear the Python-level data queue. Does NOT touch the serial port."""
        with self.lock:
            self.data_queue.clear()

    def get_current_values(self):
        with self.lock:
            return self.latest_sample

# ==========================================
# KEYBOARD HANDLER
# ==========================================
class KeyboardRecorder:
    def __init__(self):
        self.is_recording = False
        self.should_quit = False
        self.space_pressed = False
        self.undo_pressed = False

    def on_press(self, key):
        if key == keyboard.Key.space and not self.space_pressed:
            self.space_pressed = True
            self.is_recording = True
        elif key == keyboard.Key.backspace:
            self.undo_pressed = True
        elif key == keyboard.Key.esc:
            self.should_quit = True
            return False

    def on_release(self, key):
        if key == keyboard.Key.space:
            self.space_pressed = False
            self.is_recording = False
        elif key == keyboard.Key.backspace:
            self.undo_pressed = False

# ==========================================
# UI HELPERS
# ==========================================
def draw_box(title):
    print(f"{C_GREY}╔{'═'*58}╗{C_RESET}")
    print(f"{C_GREY}║ {C_ORANGE}{C_BOLD}{title.center(56)}{C_RESET} {C_GREY}║{C_RESET}")
    print(f"{C_GREY}╚{'═'*58}╝{C_RESET}")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# ==========================================
# MAIN LOOP
# ==========================================
def run_session(reader, labels, samples_per_label, save_path, phase_name):
    kb = KeyboardRecorder()
    listener = keyboard.Listener(on_press=kb.on_press, on_release=kb.on_release)
    listener.start()

    # Build Queue
    queue = []
    for label in labels:
        queue.extend([label] * samples_per_label)
    random.shuffle(queue)

    total = len(queue)
    completed = 0
    counts = {l: 0 for l in labels}

    def draw_session_header(current_label, current_idx):
        """Redraw the full session header with BIG target label."""
        clear_screen()
        draw_box("OpenEMG // DATA ACQUISITION")
        print("\n")
        print_styled(f"   SESSION: {phase_name}", C_WHITE)
        print_styled(f"   TARGET:  {total} SAMPLES ({samples_per_label}/LABEL)", C_GREY)
        print_styled(f"   LABELS:  {', '.join(labels)}", C_GREY)

        # Per-label progress
        prog_parts = []
        for lbl in labels:
            done = counts.get(lbl, 0)
            if lbl == current_label:
                prog_parts.append(f"{C_ORANGE}{C_BOLD}{lbl}:{done}/{samples_per_label}{C_RESET}")
            else:
                prog_parts.append(f"{C_GREY}{lbl}:{done}/{samples_per_label}{C_RESET}")
        print(f"   COUNTS:  {' | '.join(prog_parts)}")

        print(f"\n{C_GREY}   {'─'*58}{C_RESET}")
        # BIG target banner
        if current_label == "NOISE":
            noise_action = random.choice(NOISE_PROMPTS)
            print(f"\n{C_RED}{C_BOLD}   ╔══════════════════════════════════════════════╗{C_RESET}")
            print(f"{C_RED}{C_BOLD}   ║  NOISE: {noise_action:<36} ║{C_RESET}")
            print(f"{C_RED}{C_BOLD}   ║         [{current_idx+1:03d}/{total:03d}]                            ║{C_RESET}")
            print(f"{C_RED}{C_BOLD}   ╚══════════════════════════════════════════════╝{C_RESET}\n")
        elif current_label == "SILENCE":
            print(f"\n{C_GREY}{C_BOLD}   ╔══════════════════════════════════════════════╗{C_RESET}")
            print(f"{C_GREY}{C_BOLD}   ║  SILENCE: Relax completely   [{current_idx+1:03d}/{total:03d}]    ║{C_RESET}")
            print(f"{C_GREY}{C_BOLD}   ╚══════════════════════════════════════════════╝{C_RESET}\n")
        else:
            print(f"\n{C_ORANGE}{C_BOLD}   ╔══════════════════════════════════════════════╗{C_RESET}")
            print(f"{C_ORANGE}{C_BOLD}   ║  >>> SAY: {current_label:<12}  [{current_idx+1:03d}/{total:03d}]       ║{C_RESET}")
            print(f"{C_ORANGE}{C_BOLD}   ╚══════════════════════════════════════════════╝{C_RESET}\n")
        print_styled("   [HOLD SPACE] RECORD   [BACKSPACE] UNDO LAST   [ESC] QUIT", C_GREY)
        print(f"{C_GREY}   {'─'*58}{C_RESET}\n")

    reader.get_latest_data(clear=True) # Flush

    # Keep track of last recorded file for Undo
    last_file_path = None
    last_label = None

    for i, label in enumerate(queue):
        if kb.should_quit: break

        # Redraw header with current target
        draw_session_header(label, completed)

        # IDLE LOOP: Show dashboard, handle undo, drain stale serial data
        while not kb.is_recording and not kb.should_quit:
            # CHECK UNDO
            if kb.undo_pressed:
                if last_file_path and os.path.exists(last_file_path):
                    try:
                        os.remove(last_file_path)
                        counts[last_label] -= 1
                        completed -= 1
                        print(f"\r{C_RED}   UNDO  {C_GREY}Deleted {os.path.basename(last_file_path)}{C_RESET}                                   ")
                        queue.append(last_label)
                        last_file_path = None
                        time.sleep(1.0)
                    except Exception as e:
                        print(f"\r{C_RED}   ERR   {C_GREY}Could not delete: {e}{C_RESET}")
                else:
                    print(f"\r{C_ORANGE}   WARN  {C_GREY}Nothing to undo!{C_RESET}                                   ")
                    time.sleep(1.0)

                kb.undo_pressed = False
                draw_session_header(label, completed)  # Redraw after undo

            # Drain stale data from queue (keeps latest_sample for dashboard)
            reader.get_latest_data(clear=True)

            # Show live dashboard
            current = reader.get_current_values()
            dash_str = ""
            if len(current) > 1:
                for ch_idx, val in enumerate(current[1:]):
                    dash_str += f"[{reader.channels[ch_idx+1]}: {val:>4}] "
            else:
                dash_str = "[WAITING FOR DATA...]"

            hz = f"{reader.sample_rate_estimate}Hz"
            progress = f"[{completed+1:03d}/{total:03d}]"
            sys.stdout.write(f"\r   {C_GREY}{progress}{C_RESET} {C_WHITE}{label:<8}{C_RESET} {C_GREY}{dash_str}{C_RESET}\033[K")
            sys.stdout.flush()

            time.sleep(0.05)  # 20 Hz refresh, prevents CPU spin

        if kb.should_quit: break

        # RECORDING STARTED — clean start
        reader.flush_all()          # Clear any queued data from idle period
        start_t = time.time()
        samples = 0
        data = []

        while kb.is_recording:
            chunk = reader.get_latest_data(clear=True)
            if chunk:
                data.extend(chunk)
                samples += len(chunk)

            elapsed = time.time() - start_t
            # Visual Bar
            # Threshold is now 250 samples (TARGET_SAMPLES)
            progress_ratio = min(1.0, samples / TARGET_SAMPLES)
            bar_len = int(progress_ratio * 20)

            # Color change: Grey -> Orange -> Green
            bar_color = C_GREY
            if samples > 50: bar_color = C_ORANGE
            if samples >= TARGET_SAMPLES: bar_color = C_GREEN

            bar = "█" * bar_len
            # Overwrite the same line with recording status
            sys.stdout.write(f"\r{C_ORANGE}   REC   {C_WHITE}{elapsed:.1f}s {C_GREY}|{bar_color}{bar:<20}{C_GREY}| {samples}smp{C_RESET}\033[K")
            sys.stdout.flush()
            time.sleep(0.005)

        # SAVE
        duration = time.time() - start_t
        if samples >= MIN_SAMPLES:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            counts[label] += 1
            fname = f"{label}_{counts[label]:03d}_{ts}.csv"
            fpath = os.path.join(save_path, fname)

            headers = reader.channels + ["Label", "Phase"]

            with open(fpath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for row in data:
                    while len(row) < len(reader.channels): row.append("0")
                    writer.writerow(row[:len(reader.channels)] + [label, phase_name])

            print(f"\r{C_GREEN}   PASS  {C_GREY}{fname} ({samples}smp){C_RESET}                                   ")
            completed += 1
            last_file_path = fpath
            last_label = label
        else:
            print(f"\r{C_RED}   FAIL  {C_GREY}Too Short (<{MIN_SAMPLES}){C_RESET}                                   ")
            queue.append(label) # Retry

        time.sleep(0.2)

    listener.stop()
    print("\n")
    draw_box("SESSION COMPLETE")

# ==========================================
# FALLBACK MODE (NO PYNPUT)
# ==========================================
def run_session_fallback(reader, labels, samples_per_label, save_path, phase_name):
    # Build Queue
    queue = []
    for label in labels:
        queue.extend([label] * samples_per_label)
    random.shuffle(queue)

    total = len(queue)
    completed = 0
    counts = {l: 0 for l in labels}

    clear_screen()
    draw_box("OpenEMG // FALLBACK MODE")
    print("\n")
    print_styled(f"   SESSION: {phase_name}", C_WHITE)
    print_styled(f"   NOTE:    pynput not found. Using ENTER key mode.", C_ORANGE)
    print(f"\n{C_GREY}   {'─'*58}{C_RESET}\n")

    reader.get_latest_data(clear=True)

    for i, label in enumerate(queue):
        # 1. Prompt
        progress = f"[{completed+1:03d}/{total:03d}]"
        print_styled(f"   {progress} TARGET: {label}", C_WHITE, bold=True)

        # 2. Wait for Enter
        input(f"   {C_ORANGE}Press ENTER to record (2s)...{C_RESET}")

        # 3. Record fixed duration
        print(f"   {C_ORANGE}REC...{C_RESET}", end="", flush=True)
        start_t = time.time()
        data = []
        reader.get_latest_data(clear=True)

        while time.time() - start_t < 2.0:
            chunk = reader.get_latest_data(clear=True)
            if chunk: data.extend(chunk)
            time.sleep(0.01)

        # 4. Save
        samples_len = len(data)
        if samples_len >= MIN_SAMPLES:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            counts[label] += 1
            fname = f"{label}_{counts[label]:03d}_{ts}.csv"
            fpath = os.path.join(save_path, fname)

            headers = reader.channels + ["Label", "Phase"]
            with open(fpath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for row in data:
                    while len(row) < len(reader.channels): row.append("0")
                    writer.writerow(row[:len(reader.channels)] + [label, phase_name])

            print(f" {C_GREEN}PASS{C_RESET} ({samples_len}smp)")
            completed += 1
        else:
            print(f" {C_RED}FAIL{C_RESET} (Too Short)")
            queue.append(label)

        print(f"   {C_GREY}{'─'*30}{C_RESET}")

    draw_box("SESSION COMPLETE")

# ==========================================
# MAIN STARTUP
# ==========================================
def main():
    clear_screen()
    draw_box("OpenEMG // SYSTEM INIT")

    # Instructions
    print(f"\n{C_GREY}   HOW TO USE:{C_RESET}")
    print(f"   1. Select '{C_BOLD}Phase{C_RESET}' (e.g., Overt, Mouthing).")
    print(f"   2. Enter '{C_BOLD}Labels{C_RESET}' (words to record).")
    if PYNPUT_AVAILABLE:
        print(f"   3. {C_ORANGE}HOLD SPACE{C_RESET} to record while performing the action.")
        print(f"   4. {C_ORANGE}RELEASE SPACE{C_RESET} to stop and save.")
    else:
        print(f"   3. {C_ORANGE}PRESS ENTER{C_RESET} to record associated word (2s fix).")
    print(f"   5. Watch the dashboard to ensure meaningful signal jumps.")
    print(f"{C_GREY}   {'─'*58}{C_RESET}")

    # Permission Check
    if not os.access(os.getcwd(), os.W_OK):
        print_styled(f"\n   [ERR] NO WRITE PERMISSION IN: {os.getcwd()}", C_RED)
        return

    # Port Detection
    port = None
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(x in p.description.upper() for x in ["SLAB", "USB", "UART", "CP210", "CH340"]):
            port = p.device
            break

    if not port:
        print_styled("\n   [ERR] SENSOR NOT FOUND", C_RED)
        port = input(f"{C_GREY}   Enter Port Manually > {C_RESET}").strip()
        if not port: return

    print_styled(f"\n   [SYS] LINKING {port} @ {BAUD_RATE}...", C_GREY)

    try:
        reader = SerialReader(port, BAUD_RATE)
        reader.start()
    except Exception as e:
        print_styled(f"   [ERR] CONNECTION FAILED: {e}", C_RED)
        return

    # Metadata & Phases
    print("\n")
    subj = input(f"{C_ORANGE}   SUBJECT ID {C_GREY}(default: test) > {C_RESET}").strip().lower() or "test"

    print(f"\n{C_GREY}   PHASES:{C_RESET}")
    print("   1. Overt (Speaking aloud)")
    print("   2. Whispered")
    print("   3. Mouthing (Lip sync only)")
    print("   4. Rest (Silence/Noise)")
    print("   5. Exaggerated (Over-articulation)")
    print("   6. Covert (Subvocalization)")

    phase = input(f"{C_ORANGE}   SELECT PHASE [1-6] > {C_RESET}").strip() or "1"

    phase_map = {
        "1": "Phase_1_Overt",
        "2": "Phase_2_Whispered",
        "3": "Phase_3_Mouthing",
        "4": "Phase_4_Rest",
        "5": "Phase_5_Exaggerated",
        "6": "Phase_6_Covert"
    }
    phase_name = phase_map.get(phase, "Phase_1_Overt")

    save_path = os.path.join(DATA_DIR, subj, phase_name)
    if not os.path.exists(save_path): os.makedirs(save_path)

    print("\n")
    print_styled(f"   [SYS] DATASTORE: {save_path}", C_GREY)

    # Labels
    l_input = input(f"{C_ORANGE}   LABELS {C_GREY}(csv) > {C_RESET}").strip().upper()
    labels = [x.strip() for x in l_input.split(',')] if l_input else ["ONE", "SILENCE"]

    n_input = input(f"{C_ORANGE}   SAMPLES/LBL {C_GREY}(10) > {C_RESET}").strip()
    count = int(n_input) if n_input.isdigit() else 10

    if PYNPUT_AVAILABLE:
        run_session(reader, labels, count, save_path, phase_name)
    else:
        run_session_fallback(reader, labels, count, save_path, phase_name)

    reader.stop()

if __name__ == "__main__":
    main()
