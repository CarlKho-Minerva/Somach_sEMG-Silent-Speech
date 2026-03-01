# OpenEMG Curriculum Recorder Guide

> **Two studies, one pipeline.** This guide covers both Study A and Study B. Pick your study, wire up, and follow it top to bottom.

---

## Which Study Am I Running?

| | Study A: Lingual-Mandibular (Transfer Learning) | Study B: Subvocal (Curriculum Learning) |
|---|---|---|
| **Goal** | Open-mouth → closed-mouth transfer | Overt → covert curriculum degradation |
| **Sensor 1** | Chin (Mentalis) — GPIO 34 | Chin (Mentalis) — GPIO 34 |
| **Sensor 2** | **Under-Chin** (Mylohyoid) — GPIO 36 | **Left Throat** (Thyrohyoid) — GPIO 36 |
| **Reference** | Left earlobe | Left earlobe |
| **Vocabulary** | UP, DOWN, LEFT, RIGHT, SILENCE, NOISE | UP, DOWN, LEFT, RIGHT, SILENCE, NOISE |
| **Phases** | 1 (Overt) → 3 (Mouthing) → 5 (Exaggerated, closed) → 6 (Covert, closed) | 1 (Overt) → 2 (Whispered) → 3 (Mouthing) → 6 (Covert) |
| **Samples/class/phase** | 50 | 50 |

---

## Step 1: Electrode Placement

### Sensor 1 — Chin (Same for both studies)
- **Yellow (LO+):** Center of chin, on the bony prominence
- **Green (LO-):** 1.5 cm below Yellow, toward the jawline
- **Note:** On many chins, there is very little space. It's okay if these two electrodes are right next to each other (touching or nearly touching).
- **Red (Reference):** Left earlobe (clip-on)

### Sensor 2 — Study A: Under-Chin
- **Yellow (LO+):** Center of the soft area under your jaw, directly below the chin bone
- **Green (LO-):** 1.5 cm further back toward the throat

> **Test:** Press your tongue to the roof of your mouth. The under-chin area should feel hard. If your CH2 value doesn't spike on the dashboard, reposition.

### Sensor 2 — Study B: Left Throat
- Find your Adam's apple (thyroid cartilage)
- **Yellow (LO+):** Slide 1.5 cm to the LEFT of the Adam's apple peak, on the flat "wing"
- **Green (LO-):** 1.5 cm directly below Yellow

> **Test:** Hum quietly. CH2 should spike. If it doesn't, the electrode is too far from the larynx.

---

## Step 2: Firmware

Flash `3-high-speed-capture.ino` to your ESP32. Confirm these lines in the sketch:

```cpp
const int CHANNEL_PINS[] = {34, 36};  // 2 channels
const int NUM_CHANNELS = sizeof(CHANNEL_PINS) / sizeof(CHANNEL_PINS[0]);
```

**Verify:** Open Arduino Serial Monitor (115200 baud). You should see:
```
12345,2048,1950
12346,2051,1948
```
Two values after the timestamp = 2 channels detected. **Close Serial Monitor before running Python.**

---

## Step 3: Record Data

```bash
# Activate the virtual environment first
cd /Users/cvk/Downloads/carl/md-capstonefall25_25TPE/master-progress-list/020526_INSTRUCTABLES_GuideForMeEveryone/code
source .venv/bin/activate

# Then navigate to your study directory and run from there
cd sessions/StudyA_Lingual    # or StudyB_Subvocal
python3 ../../python/4-curriculum-recorder.py
```

### Prompts

| Prompt | Study A Answer | Study B Answer |
|--------|---------------|---------------|
| SUBJECT ID | `carl` | `carl` |
| PHASE | See phase table below | See phase table below |
| LABELS | `UP,DOWN,LEFT,RIGHT,SILENCE,NOISE` | `UP,DOWN,LEFT,RIGHT,SILENCE,NOISE` |
| SAMPLES/LBL | `50` | `50` |

### Phase Recording Order

Run the recorder **once per phase.** Re-run the script each time to select a new phase.

**Study A — Lingual-Mandibular (Open → Closed Mouth Transfer):**

| Run | Phase | Mouth | What You Do |
|-----|-------|-------|-------------|
| 1 | `1` (Overt) | **OPEN** | Say UP/DOWN/LEFT/RIGHT aloud + full tongue gesture |
| 2 | `3` (Mouthing) | **OPEN** | Lip sync, tongue active, no voice |
| 3 | `5` (Exaggerated) | **CLOSED** | Mouth shut. Exaggerate the tongue press hard. No visible movement. |
| 4 | `6` (Covert) | **CLOSED** | Mouth shut. Think the tongue gesture with minimal effort. |

> **Transfer learning story:** Train on Phases 1+3 (open mouth), evaluate on Phases 5+6 (closed mouth). Shows the same signal pattern transfers when jaw is shut.

**Study B — Subvocal (Curriculum Learning):**

| Run | Phase | What You Do |
|-----|-------|-------------|
| 1 | `1` (Overt) | Speak UP/DOWN/LEFT/RIGHT aloud — strongest signal |
| 2 | `2` (Whispered) | Whisper the commands — reduced amplitude |
| 3 | `3` (Mouthing) | Lip sync only, no voice — practical target |
| 4 | `6` (Covert) | Internal speech, mouth closed — stretch goal |

> **Curriculum story:** Same model, same vocab. Accuracy degrades as phonation gets weaker. Throat sensor captures what chin sensor can't.

### The Tongue Gesture Table (Study A)

| Label | Think | Physical Action | What to Feel |
|-------|-------|----------------|-------------|
| **UP** | "Nnnn" | Tongue flat → hard palate (roof of mouth) | Strong upward jaw-floor pressure |
| **DOWN** | "Gah" | Tongue back/down, like starting a yawn | Deep tension at tongue root |
| **LEFT** | "Llll" | Tongue → left cheek/molars | One-sided jaw pressure (left) |
| **RIGHT** | "Errr" | Curl tongue tip back toward soft palate | Tight tip tension, hollow underneath |
| **SILENCE** | Nothing | Relax completely | No tension |
| **NOISE** | Anything | Swallow, yawn, scratch face | Random non-command movement |

### Recording Tips
- **HOLD SPACE** while doing the gesture, **RELEASE** when done
- If dashboard shows `FAIL Too Short`, you tapped too fast — hold longer
- Watch CH2 values: they should spike during UP/DOWN/LEFT/RIGHT and stay flat during SILENCE
- If CH2 is flatlined, your electrode detached. Stop and re-stick it.
- Record all 50 samples of one label before moving to the next

---

## Step 4: Run the Full Pipeline

After recording all phases, stay in the session directory and run these **in order**:

```bash
# Make sure venv is active (if not already):
# source /Users/cvk/Downloads/carl/md-capstonefall25_25TPE/master-progress-list/020526_INSTRUCTABLES_GuideForMeEveryone/code/.venv/bin/activate

# You should already be in:
# code/sessions/StudyA_Lingual/  (or StudyB_Subvocal/)

# 1. Clean raw signals (bandpass + notch filter)
python3 ../../python/5-preprocess.py --input data_collection --output data_clean

# 2. Extract features PER PHASE (for curriculum evaluation)
python3 ../../python/6-extract-features.py --input data_clean --output features --per-phase

# 3. Also extract COMBINED features (for training the final model)
python3 ../../python/6-extract-features.py --input data_clean --output features/all

# 4. CURRICULUM EVAL — the professor's deliverable
python3 ../../python/10-curriculum-eval.py --features-dir features

# 5. Train final model (for real-time demo)
python3 ../../python/7-train-model.py --features features/all --output models

# 6. Real-time prediction (live demo)
python3 ../../python/8-realtime-predict.py --model models/model.pth --meta models/model_meta.pkl
```

### What `10-curriculum-eval.py` Outputs

This is the table your professor wants to see:

```
╔═══════════════════════════════════════════╗
║   CURRICULUM LEARNING EVALUATION          ║
╚═══════════════════════════════════════════╝

Phase 1 (Overt):      96.2%  ██████████████████░░
Phase 3 (Mouthing):   74.1%  ██████████████░░░░░░
Phase 5 (Exaggerated):61.0%  ████████████░░░░░░░░
Phase 6 (Covert):     52.3%  ██████████░░░░░░░░░░

Δ (Overt → weakest): -43.9%

[Per-phase confusion matrices follow]
```

It trains ONE model on ALL phases combined, then evaluates each phase separately to show the degradation.

---

## 📂 Directory Structure After a Full Run

```
StudyA_Lingual/
├── data_collection/carl/
│   ├── Phase_1_Overt/          ← 300 raw CSVs (50 × 6 labels)
│   ├── Phase_3_Mouthing/       ← 300 raw CSVs
│   ├── Phase_5_Exaggerated/    ← 300 raw CSVs
│   └── Phase_6_Covert/         ← 300 raw CSVs
├── data_clean/carl/            ← Filtered copies of above
├── features/
│   ├── Phase_1_Overt/          ← X.npy, y.npy (per-phase)
│   ├── Phase_3_Mouthing/
│   ├── Phase_5_Exaggerated/
│   ├── Phase_6_Covert/
│   └── all/                    ← X.npy, y.npy (combined)
└── models/
    ├── model.pth               ← Trained model weights
    └── model_meta.pkl          ← Classes + architecture info
```

---

## 🔧 Troubleshooting

| Issue | Cause | Fix |
|:---|:---|:---|
| "Sensor Not Found" | Serial port not detected | `ls /dev/cu.*`, enter port manually |
| "Permission Denied" | macOS blocking input | System Settings → Privacy → Accessibility → Enable Terminal |
| Flatline CH1 or CH2 | Electrode detached | Re-stick electrode, use more gel |
| Values stuck at 4095 | Signal clipping | Reposition electrode, loosen pressure |
| SILENCE and NOISE only in real-time | Model can't distinguish commands | Record more data, ensure signal spikes during gestures |
| `--per-phase` finds 0 phases | CSVs not in Phase_* directories | Check your `data_clean/` structure matches the expected layout |
