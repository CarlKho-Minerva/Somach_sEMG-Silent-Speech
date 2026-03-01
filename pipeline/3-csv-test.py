# collect_data.py
import serial
import time
import csv
import os

# ⚠️ CHANGE THIS to your port!
PORT = '/dev/cu.usbserial-110'
BAUD_RATE = 115200
DURATION = 2  # Seconds per recording
OUTPUT_FILE = 'emg_data.csv'

# Connect to ESP32
ser = serial.Serial(PORT, BAUD_RATE)
time.sleep(2)  # Wait for connection

# Initialize CSV if it doesn't exist
if not os.path.isfile(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "label", "ch1", "ch2", "ch3"])
    print(f"Created {OUTPUT_FILE}")

print("=" * 50)
print("DATA COLLECTION TOOL")
print("=" * 50)

while True:
    label = input("\nEnter LABEL for this word (or 'q' to quit): ").strip().upper()
    if label == 'Q':
        break

    input(f"Get ready to say '{label}'... Press ENTER to start recording.")

    print("🔴 RECORDING...")
    start_time = time.time()
    samples = []

    # Clear buffer
    ser.reset_input_buffer()

    while time.time() - start_time < DURATION:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if 'CH1:' in line:
                parts = line.split()
                ch1 = int(parts[0].split(':')[1])
                ch2 = int(parts[1].split(':')[1])
                ch3 = int(parts[2].split(':')[1])

                timestamp = time.time()
                samples.append([timestamp, label, ch1, ch2, ch3])
        except:
            pass

    print("✅ DONE.")

    # Save to CSV
    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(samples)

    print(f"Saved {len(samples)} samples to {OUTPUT_FILE}")

ser.close()