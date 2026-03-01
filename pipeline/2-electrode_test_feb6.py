# electrode_test.py
# Run WITH electrodes attached

import serial
import numpy as np
import time

# ⚠️ CHANGE THIS to your port!
PORT = '/dev/cu.usbserial-110'

ser = serial.Serial(PORT, 115200)
time.sleep(2)

def collect_samples(duration=3):
    samples = []
    start = time.time()
    while time.time() - start < duration:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if 'CH1:' in line:
            parts = line.split()
            ch1 = int(parts[0].split(':')[1])
            ch2 = int(parts[1].split(':')[1])
            ch3 = int(parts[2].split(':')[1])
            samples.append([ch1, ch2, ch3])
    return np.array(samples)

print("ELECTRODE CONTACT TEST")
print("=" * 40)

# Test 1: Baseline
print("\n[TEST 1] BASELINE - Sit still")
input("Press Enter when ready...")
baseline = collect_samples(3)
print(f"CH1 range: {np.ptp(baseline[:,0]):.0f}")
print(f"CH2 range: {np.ptp(baseline[:,1]):.0f}")
print(f"CH3 range: {np.ptp(baseline[:,2]):.0f}")
print("(Good if all < 80)")

# Test 2: Swallow
print("\n[TEST 2] SWALLOW - Do a big swallow")
input("Press Enter, then SWALLOW...")
swallow = collect_samples(3)
print(f"CH2 spike: {np.ptp(swallow[:,1]):.0f} (expect > 200)")
print(f"CH3 spike: {np.ptp(swallow[:,2]):.0f} (expect > 200)")

# Test 3: Jaw clench
print("\n[TEST 3] JAW CLENCH - Clench your jaw hard")
input("Press Enter, then CLENCH...")
clench = collect_samples(3)
print(f"CH1 spike: {np.ptp(clench[:,0]):.0f} (expect > 300)")

# Results
print("\n" + "=" * 40)
if (np.ptp(baseline[:,0]) < 80 and
    np.ptp(swallow[:,1]) > 200 and
    np.ptp(clench[:,0]) > 300):
    print("✅ ALL TESTS PASSED - Ready for data collection!")
else:
    print("⚠️ Some tests may need adjustment")
    print("Check electrode placement and try again")

ser.close()