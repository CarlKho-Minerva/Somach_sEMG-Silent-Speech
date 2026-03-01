# validation_test.py
# Run this WITHOUT electrodes to test hardware

import serial
import time
import numpy as np

# ⚠️ CHANGE THIS to your port!
# Mac: /dev/cu.usbserial-XXXX or /dev/cu.SLAB_USBtoUART
# Windows: COM3, COM4, etc.
# Linux: /dev/ttyUSB0
PORT = '/dev/cu.usbserial-10'

ser = serial.Serial(PORT, 115200, timeout=1)
time.sleep(2)  # Wait for ESP32 to reset

print("=" * 50)
print("HARDWARE VALIDATION - NO ELECTRODES")
print("=" * 50)

# Collect 30 samples
# Collect 30 valid samples
samples = []
max_attempts = 200  # Prevent infinite loop
attempts = 0

print("Attemping to read data...")
while len(samples) < 30 and attempts < max_attempts:
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        attempts += 1

        if not line:
            continue

        print(f"Raw: {line}") # Debug print

        if 'CH1:' in line:
            parts = line.split()
            # Expected format: CH1:2048 CH2:2048 CH3:2048
            if len(parts) >= 3:
                ch1 = int(parts[0].split(':')[1])
                ch2 = int(parts[1].split(':')[1])
                ch3 = int(parts[2].split(':')[1])
                samples.append([ch1, ch2, ch3])
        elif ',' in line:
            # Expected CSV format: timestamp, ch1, ch2, ch3
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    ch1 = int(parts[1])
                    ch2 = int(parts[2])
                    ch3 = int(parts[3])
                    samples.append([ch1, ch2, ch3])
                except ValueError:
                    pass # Header or garbage
    except Exception as e:
        print(f"Error parsing line: {line} -> {e}")

if len(samples) == 0:
    print("\n❌ NO DATA RECEIVED.")
    print("Check: 1) ESP32 is flashing? 2) Correct Port? 3) Baud rate 115200?")
    ser.close()
    exit()

# Analyze
samples = np.array(samples)
for i, name in enumerate(['CH1 (Chin)', 'CH2 (L-Throat)', 'CH3 (R-Throat)']):
    mean = np.mean(samples[:, i])
    std = np.std(samples[:, i])

    print(f"\n{name}:")
    print(f"  Mean: {mean:.1f} (expect ~2048)")
    print(f"  Std:  {std:.1f} (expect < 50)")

    if mean < 100:
        print("  ❌ DEAD - SDN connected to GND? Move to 3.3V (+)")
    elif mean > 4000:
        print("  ⚠️  SATURATED (MAX) - Inputs likely floating (normal if no electrodes)")
    elif std > 100:
        print("  ⚠️  NOISY - LO+/LO- not connected to 3.3V")
    elif 1900 < mean < 2200 and std < 50:
        print("  ✅ GOOD")

ser.close()