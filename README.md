# Electrode Placement Boundaries for Ultra-Low-Cost Silent Speech Interfaces

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
> **arXiv Preprint IDs Pending Submission**

<img width="2048" height="1046" alt="image" src="https://github.com/user-attachments/assets/bd73bf73-f488-4588-be3f-a5155c68ddf0" />

This repository contains the **complete dataset, source code, evaluation pipeline, and LaTeX sources** for two companion papers on silent speech interface (SSI) classification using a \$40 consumer-grade sEMG system (2× AD8232 + ESP32).

**Replication Guide:** [https://somach.vercel.app/editorial/complete-guide.html](https://somach.vercel.app/editorial/complete-guide.html)

## Abstract

Consumer-grade biosensors (AD8232 + ESP32, 12-bit ADC, 250 Hz) cost \$40 but introduce severe quantization noise compared to research-grade 24-bit systems (\$1,000+). We evaluate what is and is not achievable with these hardware constraints for silent speech classification across two electrode configurations and a 5-phase speech-intensity curriculum (Overt → Whispered → Mouthing → Exaggerated → Covert).

## Key Results

| Study | Electrode Config | 5-Fold CV | Chance | Gated Accuracy |
|-------|-----------------|-----------|--------|----------------|
| **B** (Paper 1) | Chin + Throat | 48.9% ± 3.1% | 16.7% | 64.1% @ θ=0.60 |
| **A** (Paper 2) | Chin + Under-chin | 51.8% ± 2.8% | 16.7% | — |

Cross-study transfer: 25–31% (near chance), establishing that electrode placement creates **fundamentally incompatible feature spaces** on consumer-grade ADCs.

## 📄 Read the Papers (Pre-arXiv PDFs)

> arXiv submissions planned after **March 20, 2026**. PDFs below are the final versions.

| Paper | PDF | Key Result |
|-------|-----|------------|
| **Study B** — Curriculum Learning, Chin + Throat | [StudyB_CurriculumLearning_SilentSpeech_ChinThroat.pdf](StudyB_CurriculumLearning_SilentSpeech_ChinThroat.pdf) | 48.9% ± 3.1% CV · 64.1% gated |
| **Study A** — Electrode Placement, Chin + Under-chin | [StudyA_ElectrodePlacement_LingualMandibular.pdf](StudyA_ElectrodePlacement_LingualMandibular.pdf) | 51.8% ± 2.8% CV · 18 configs benchmarked |

## Papers

### Paper 1 — Study B: Curriculum Learning with Chin + Throat Electrodes
*17 pages, 5 figures, 16 references*

A 5-phase speech-intensity curriculum mapped to a 1D CNN, evaluated under 16 protocols including 5-fold CV, leave-one-phase-out, cross-session testing, hyperparameter sweep, architecture comparison, and confidence gating.

- **📄 PDF:** [StudyB_CurriculumLearning_SilentSpeech_ChinThroat.pdf](StudyB_CurriculumLearning_SilentSpeech_ChinThroat.pdf)
- **LaTeX source (updated):** [`paper1_UPDATED/main.tex`](paper1_UPDATED/main.tex)
- **LaTeX source (original):** [`papers/paper1_StudyB_ChinThroat/main.tex`](papers/paper1_StudyB_ChinThroat/main.tex)

### Paper 2 — Study A: Electrode Placement Comparison (Negative Result)
*13 pages, 4 figures, 19 references*

A controlled comparison establishing that chin + under-chin (lingual-mandibular) electrodes achieve comparable single-session accuracy to chin + throat but fail on cross-study transfer due to shared CN V3 innervation and volume-conduction crosstalk.

- **📄 PDF:** [StudyA_ElectrodePlacement_LingualMandibular.pdf](StudyA_ElectrodePlacement_LingualMandibular.pdf)
- **LaTeX source (updated):** [`paper2_UPDATED/main.tex`](paper2_UPDATED/main.tex)
- **LaTeX source (original):** [`papers/paper2_StudyA_ChinUnderChin/main.tex`](papers/paper2_StudyA_ChinUnderChin/main.tex)

## Repository Structure

```
├── README.md                         ← you are here
├── LICENSE                           ← MIT (code)
├── LICENSE-DATA                      ← CC BY 4.0 (data + papers)
├── .gitignore
│
├── data/                             ← 4,033 CSV files (87 MB)
│   ├── README.md                     ← CSV format documentation
│   ├── studyA_3phase/                ← Study A: 900 CSVs (overt/whisper/mouthing)
│   ├── studyB_5phase/                ← Study B: 1,500 CSVs (5-phase curriculum)
│   ├── covert_sessions/              ← 1,534 cross-session covert CSVs
│   └── pilot_mouthing/              ← 99 pilot-phase CSVs
│
├── firmware/                         ← Arduino sketches for ESP32 + AD8232
│   ├── test_semg_serial.ino
│   ├── three_channel_semg.ino
│   └── high_speed_capture.ino
│
├── pipeline/                         ← Python data-processing pipeline
│   ├── 1-validation.py               ← CSV integrity checks
│   ├── 2-explore-data.py             ← Signal visualization
│   ├── 3-normalize-data.py           ← Z-score normalization
│   ├── 4-train-model.py              ← 1D CNN training
│   ├── 5-evaluate-model.py           ← Hold-out evaluation
│   ├── 6-predict.py                  ← Real-time inference
│   ├── 7-phase-eval.py               ← Leave-one-phase-out
│   ├── 8-cross-session-eval.py       ← Cross-session transfer
│   ├── 9-hyperparameter-sweep.py     ← Grid search
│   ├── 10-curriculum-eval.py         ← Curriculum-order ablation
│   ├── model.py                      ← 1D CNN architecture definition
│   ├── check_serial.py               ← Serial port utility
│   └── RECORDER_GUIDE.md             ← Data recording instructions
│
├── evaluation/                       ← Reproducible evaluation
│   ├── rigorous_eval.py              ← 5-fold CV (Table I in both papers)
│   ├── generate_all_figures.py       ← Reproduces all 9 paper figures
│   ├── live_demo.py                  ← Real-time ESP32 prediction demo
│   ├── colab_results_summary.md      ← Key metrics from Colab runs
│   └── colab_walkthrough.md          ← Step-by-step Colab guide
│
├── figures_generated/                ← All paper figures (PNG)
│   ├── fig1_electrode_placement.png
│   ├── fig2_signal_visualization.png
│   ├── fig3_training_curve.png
│   ├── fig4_confusion_matrix.png
│   ├── fig5_confidence_gating.png
│   ├── p2_fig1_electrode_placement.png
│   ├── p2_fig2_training_curve.png
│   ├── p2_fig3_confusion_matrix.png
│   └── p2_fig4_cross_study_transfer.png
│
├── paper1_UPDATED/                   ← Updated Paper 1 (arxiv.sty format)
│   ├── main.tex
│   ├── arxiv.sty
│   └── figures/
│
├── paper2_UPDATED/                   ← Updated Paper 2 (arxiv.sty format)
│   ├── main.tex
│   ├── arxiv.sty
│   └── figures/
│
├── papers/                           ← Original LaTeX sources + compiled PDFs
│   ├── paper1_StudyB_ChinThroat/
│   │   ├── main.tex
│   │   ├── main_REFERENCE.pdf
│   │   └── figures/
│   └── paper2_StudyA_ChinUnderChin/
│       ├── main.tex
│       ├── main_REFERENCE.pdf
│       └── figures/
│
├── process/                          ← Research process documentation
│   ├── hardware_logs/                ← Build notes, troubleshooting
│   ├── strategy_docs/                ← Research plans, status reviews
│   ├── experiments/                  ← Session protocols, ablation logs
│   ├── oral_defense/                 ← Defense prep materials
│   └── writing/                      ← Draft evolution, audit trails
│
└── support/
    ├── generate_all_figures.py
    └── SESSION_CHANGELOG.md          ← Full change log
```

## Hardware

| Component | Part | INR | USD | Role |
|-----------|------|-----|-----|------|
| Biosensor × 2 | AD8232 ECG breakout | ₹790 × 2 = ₹1,580 | $17.34 | Analog front-end |
| Microcontroller | ESP32 NodeMCU DevKit | ₹549 | $6.03 | ADC + serial streaming |
| Electrodes | Pediatric Ag/AgCl (50-pk) | ₹523 | $5.74 | Low-impedance skin contact |
| Breadboard + wires | — | ₹279 | $3.06 | Prototyping connections |
| USB-C cable | — | ₹198 | $2.17 | Power + serial data |
| Extra breadboard | ESP32 mount | ₹67 | $0.74 | Dedicated mount |
| **Total** | | **₹3,196** | **$35.09** | **Complete 2-ch sEMG system** |

> All prices verified on Amazon India (February 2026). The "$40" figure in the papers is a conservative rounding.

- **ADC:** 12-bit (0–4095), 250 Hz sampling rate
- **Channels:** 2 differential (CH1: GPIO 34, CH2: GPIO 36)
- **Reference:** Single mastoid electrode shared by both AD8232 modules

Full build guide with photos: [somach.vercel.app/editorial/complete-guide.html](https://somach.vercel.app/editorial/complete-guide.html)

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
# or manually:
pip install numpy pandas scikit-learn torch librosa scipy matplotlib seaborn pyserial
```

### Reproducing Paper Results (5-Fold CV)

```bash
cd evaluation/
python rigorous_eval.py --data ../data/studyB_5phase/   # → Paper 1, Table I
python rigorous_eval.py --data ../data/studyA_3phase/   # → Paper 2, Table I
```

### Regenerating All Figures

```bash
cd evaluation/
python generate_all_figures.py
```

### Compiling Papers from LaTeX

```bash
# Updated papers (arxiv.sty format)
cd paper1_UPDATED/
pdflatex main.tex && pdflatex main.tex

cd ../paper2_UPDATED/
pdflatex main.tex && pdflatex main.tex

# Original papers
cd ../papers/paper1_StudyB_ChinThroat/
pdflatex main.tex && pdflatex main.tex

cd ../paper2_StudyA_ChinUnderChin/
pdflatex main.tex && pdflatex main.tex
```

### Running the Live Demo

```bash
# Connect ESP32 via USB, flash firmware/high_speed_capture.ino
cd evaluation/
python live_demo.py --port /dev/tty.usbserial-*
```

## Data Format

Each CSV contains 250 Hz sEMG samples:

| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | int | Milliseconds since recording start |
| `CH1` | int | ADC reading 0–4095 (channel 1) |
| `CH2` | int | ADC reading 0–4095 (channel 2) |
| `Label` | str | One of: `hello`, `stop`, `yes`, `no`, `thanks`, `help` |
| `Phase` | str | Speech intensity: `overt`, `whisper`, `mouthing`, `exaggerated`, `covert` |

See [`data/README.md`](data/README.md) for full documentation.

## Ethics & Data Privacy

All data in this repository was collected through **self-experimentation** by the sole author. No external human subjects were involved. The sEMG signals are 12-bit ADC voltage readings from jaw/throat muscles; they contain no physiological identifiers (no ECG, EEG, or biometric data). The data cannot be used to re-identify individuals.

Per Minerva University's IRB-equivalent policy, self-experimentation is exempt from Human Subjects Research committee review.

## Citation

If you use this data, code, or build on this work, please cite:

```bibtex
@article{kho2026curriculum,
  title   = {Curriculum Learning for Silent Speech Classification:
             A Proof-of-Concept \$40 Two-Channel sEMG System},
  author  = {Kho, Carl Vincent Ladres},
  year    = {2026},
  note    = {arXiv preprint (submitted)}
}

@article{kho2026electrode,
  title   = {Lingual-Mandibular Electrode Configuration for Silent Speech:
             Throat Sensor Necessity on Consumer-Grade ADCs},
  author  = {Kho, Carl Vincent Ladres},
  year    = {2026},
  note    = {arXiv preprint (submitted)}
}
```

## License

- **Code** (pipeline, firmware, evaluation scripts): [MIT License](LICENSE)
- **Data & Papers** (CSVs, LaTeX, figures): [CC BY 4.0](LICENSE-DATA)

## Author

**Carl Vincent Ladres Kho**
Minerva University — Class of 2026
[kho@uni.minerva.edu](mailto:kho@uni.minerva.edu)
