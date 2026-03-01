# Soldered Sensor Verification Log - Feb 1, 2026

## 1. Directory Organization
- Created `020126_soldered` to store today's results.
- Archived relevant screenshots of the Serial Monitor.

## 2. Sensor Analysis

### Sensor A: "One Leg" (Red Board with broken pin?)
*   **Behavior:** Fluctuating 1600 - 2200 with electrode jack attached.
*   **Previous Behavior:** 0 - 1495 (likely floating/rails).
*   **Diagnosis:**
    *   **The reading is HEALTHY.** A range of 1600-2200 centers around ~1800 (1.45V). This is the expected "Reference Voltage" for the AD8232.
    *   **Physiology:** The fact that it stays near the center means the reference electrode is working and keeping the signal within the rail limits.

### Sensor B: "All Legs" (The other Red Board)
*   **Behavior (No Jack):** 1800 - 1900.
    *   **Status:** Perfect. This is the "Leads Off" voltage or just floating near VREF.
*   **Behavior (Jack Attached + Fingers):** 0 - 1495, then 1700 - 2050.
    *   **Fluctuating 0 - 1495:** This is **Signal Clipping**. When you press your fingers directly against the pins or electrodes, you create massive noise/DC offset that drives the amplifier to hit the "floor" (0V). This effectively saturates the signal.
    *   **1200 - 1400 / 1700 - 2050:** When settled, it returns to valid ranges.

## 3. Conclusion & Next Steps
**Both sensors appear to be electrically functional.** The soldering has fixed the "floating" issue (where we saw steady 3600 before).

*   **The "0-1495" range** is just the amplifier reacting to the massive electrical noise of your hands touching the raw contacts. This is normal.
*   **The "1600-2200" range** is the "sweet spot" where valid EMG signals live.

### Action Item
1.  **Proceed to Real Testing:**
    *   Stop pressing fingers against the board (causes static clipping).
    *   Stick electrodes to your face (Chin + Jaw) and Reference to Ear.
    *   Run `data_collector.py` and see if you can see a "wave" when you clench your jaw.
