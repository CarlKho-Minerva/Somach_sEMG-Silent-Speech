# Data Description

All sEMG data was collected from a single subject (male, 22, BMI ~21) using two AD8232 ECG-repurposed modules connected to an ESP32 microcontroller at 250 Hz sampling rate, 12-bit ADC resolution.

## CSV Format

Each CSV file is one recording (~1.5 seconds, ~375 samples):

| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | int | Milliseconds since ESP32 boot |
| `CH1` | int | ADC reading (0–4095), chin electrode (mentalis) |
| `CH2` | int | ADC reading (0–4095), throat or under-chin electrode |
| `Label` | str | Command class: UP, DOWN, LEFT, RIGHT, SILENCE, NOISE |
| `Phase` | str | Speech intensity phase during recording |

## Study A — Chin + Under-Chin (Lingual-Mandibular)

**Date:** February 11, 2026, 15:48–17:07 (~79 min)
**CH2 placement:** Submental triangle (mylohyoid/anterior digastric)
**Phases:** 3 (Overt, Mouthing, Exaggerated)
**Samples:** 900 (300 per phase × 6 classes × 50 per class)

```
data/studyA_3phase/
├── Phase_1_Overt/        (300 CSVs)
├── Phase_3_Mouthing/     (300 CSVs)
└── Phase_5_Exaggerated/  (300 CSVs)
```

## Study B — Chin + Throat (Curriculum Learning)

**Date:** February 11, 2026, 22:12–23:23 (~71 min) + February 24–25 (covert)
**CH2 placement:** Left thyrohyoid membrane
**Phases:** 5 (Overt → Whispered → Mouthing → Exaggerated → Covert)
**Samples:** 1,500 (300 per phase × 6 classes × 50 per class)

```
data/studyB_5phase/
├── Phase_1_Overt/        (300 CSVs)
├── Phase_2_Whispered/    (300 CSVs)
├── Phase_3_Mouthing/     (300 CSVs)
├── Phase_5_Exaggerated/  (300 CSVs)
└── Phase_6_Covert/       (300 CSVs)
```

## Supplemental Covert Sessions

Additional covert-only recordings collected across three separate sessions for multi-session experiments.

```
data/covert_sessions/
├── session_A/   (934 CSVs — Feb 24, 4 recording repeats)
├── session_B/   (300 CSVs — Feb 25)
└── session_C/   (300 CSVs — Feb 25, session C)
```

## Pilot Mouthing Data

Early pilot recordings (Feb 9) with 3-channel setup before the third sensor failed.

```
data/pilot_mouthing/   (99 CSVs)
```

## Electrode Configuration

- **CH1:** Two Ag/AgCl electrodes on chin (mentalis muscle), GPIO 34
- **CH2 (Study A):** Two electrodes under chin (mylohyoid), GPIO 36
- **CH2 (Study B):** Two electrodes on left throat (thyrohyoid), GPIO 36
- **REF:** Single electrode on mastoid process, shared by both AD8232 modules

## Ethics

All data was collected exclusively from the author in an auto-experimental paradigm. No external human subjects participated. Formal IRB review was not required under Minerva University's exemption policy for self-experimentation.

## License

CC BY 4.0 — see LICENSE-DATA in repository root.
