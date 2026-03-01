# Rigorous ML Evaluation — Interpretation & Paper Recommendations

**Colab notebook:** [022826FinalCapstone.ipynb](https://colab.research.google.com/drive/1vByQDtNsl-RW0PLcGGFF6_8GqmgMUbaI?authuser=1)
**Hardware:** NVIDIA A100-SXM4-80GB, 167 GB RAM, 29.6 min runtime
**Data:** 3,934 CSVs across 5 datasets, evaluated via 16 protocols

---

## The Big Picture: What Just Happened

Your model was reporting **99.7% accuracy** because it was training and testing on the same data. With proper held-out evaluation, the honest numbers are:

| What | Old Claim | Honest Number | Interpretation |
|------|-----------|---------------|----------------|
| Paper 1 (Study B) | 99.7% | **48.9% ± 3.1%** | 2.9× above chance (16.7%) |
| Paper 2 (Study A) | ~45% | **51.8% ± 2.8%** | 3.1× above chance |
| Cross-session | never tested | **22.8%** | Near chance — electrode shift kills it |
| Combined multi-session | — | **58.2% ± 3.1%** | Best honest number |

> [!IMPORTANT]
> **This is NOT bad.** 49% on 6-class single-session with a $40 AD8232 is scientifically meaningful. AlterEgo used $5,000+ hardware. Your contribution is proving it partially works on consumer hardware AND documenting exactly where it fails.

---

## Protocol-by-Protocol Interpretation

### P1: 5-Fold CV → 48.9% ± 3.1% (Study B)
- **Chance = 16.7%** (6 classes). You're **2.9× above chance**.
- Train acc = 71-85%, test = 44-52%. The gap (~25pp) = moderate overfitting but not catastrophic.
- **Paper update:** Replace "99.7%" with "48.9% ± 3.1% (5-fold stratified CV, k=5)".

### P2: LOPO → 44.7% mean
| Phase | Acc | Why |
|-------|-----|-----|
| Whispered (52.3%) | 🟢 Best | Closest to training distribution |
| Mouthing (52.7%) | 🟢 Best | Similar jaw mechanics |
| Overt (50.3%) | 🟡 Good | Different muscle activation pattern |
| Exaggerated (35.0%) | 🔴 Poor | Over-activation saturates the AD8232 |
| Covert (33.3%) | 🔴 Poor | Minimal signal → near chance |

**Interpretation:** Curriculum learning's value is real — phases with more jaw activation transfer better. Exaggerated and Covert fail because they occupy opposite extremes of muscle activation intensity.

### P3: Cross-Session → 22.8%
- Train on Session A (Feb 11), test on B+C (Feb 25, shifted electrodes)
- **22.8% is barely above 16.7% chance.** This confirms electrode shift is THE dominant failure mode.
- Train acc = 73.9% but test = 22.8% → the model memorizes session-specific electrode artifacts.

### P4: Combined Multi-Session → 58.2% ± 3.1%
- **This is your best honest number** and your strongest scientific claim.
- By pooling data from 3 electrode positions, the model starts learning position-invariant features.
- 58.2% > 48.9% (single-session) — proves multi-session recording recovers ~10pp.

### P5: Hyperparameter Sweep → 50.7%
Best config: `dropout=0.3, hidden=128, lr=0.001, weight_decay=0`

**Key insight:** The default config was already near-optimal. No amount of regularization meaningfully improves beyond ~51%. The ceiling is the **signal**, not the model.

Also notable: weight_decay actually *hurts* — the top configs all have `wd=0` or `wd=1e-4`. Heavy regularization (`wd=1e-2`) collapses to 16.6% (chance). The model needs enough capacity to learn the onset spike.

### P6: Architecture Comparison
| Model | Acc | Verdict |
|-------|-----|---------|
| **CNN** | **49.3%** | ✅ Only viable architecture |
| LSTM | 16.6% | ❌ Complete failure (chance) |
| Transformer | 36.4% | ❌ Too data-hungry |

**LSTM at exactly 16.6% (chance) is informative.** LSTMs need temporal structure beyond the onset spike. Since the AD8232 only captures the initiation burst (~80ms), there's no sequential pattern for the LSTM to model. It predicts the majority class.

**Transformer at 36.4%** — better than LSTM but still loses to CNN. With 1,500 samples, the self-attention mechanism can't learn meaningful representations. Would need 10-100× more data.

### P7: Learning Curve → Saturates at ~52%
| Data | Test Acc | Train Acc | Gap |
|------|----------|-----------|-----|
| 10% (120) | 24.3% | 78.3% | 54pp — severe overfit |
| 20% (240) | 42.0% | 79.6% | 38pp |
| 60% (720) | 51.7% | 78.3% | 27pp |
| 100% (1200) | 51.7% | 71.8% | 20pp |

**Saturation at ~600 samples** — adding more data beyond this doesn't improve test accuracy but DOES reduce overfitting (train-test gap narrows from 54pp → 20pp). This means: the ceiling is hardware SNR, not data quantity.

### P8: Per-Class Analysis
| Class | F1 | Quality |
|-------|-----|---------|
| SILENCE | **0.80** | ✅ Jaw-at-rest is distinctive |
| NOISE | 0.51 | 🟡 Random jaw movement detectable |
| UP | 0.50 | 🟡 Upward tongue requires jaw elevation |
| DOWN | 0.46 | 🟡 Downward tongue uses jaw depression |
| RIGHT | 0.46 | 🟡 Lateral tongue barely detectable |
| LEFT | **0.36** | ❌ Worst — throat electrodes can't resolve |

**SILENCE dominates** because jaw-at-rest produces a uniquely flat EMG signature. LEFT is worst because lateral tongue movements don't propagate strongly to the chin/throat placement.

### P9: Confidence Gating
| θ | Accuracy | Coverage | Practical Use |
|---|----------|----------|---------------|
| 0.50 | 57.4% | 79.6% | Aggressive — accepts most |
| **0.60** | **64.1%** | **62.1%** | **Best operating point** |
| 0.70 | 69.3% | 48.3% | Conservative |
| 0.80 | 74.0% | 36.4% | Very selective |
| 0.95 | 82.5% | 18.3% | Only SILENCE |

**θ=0.60 is the sweet spot:** 64% accuracy on 62% of predictions. The system says "I don't know" for 38% of inputs — an honest, deployable behavior.

### P10: Multi-Seed → 51.5% ± 1.0%
Five different random seeds all produce 50-52%. **Results are reproducible**, not fluky. The variance (±1.0%) is negligible. This is a genuine signal.

### P11-12: Study A (Paper 2) → 51.8% ± 2.8%
- Study A (chin + under-chin) actually performs **slightly better** than Study B!
- LOPO shows Overt phase is weakest at 31% — surprising. The under-chin placement may capture different onset signatures for overt speech.

### P14: Study A Architecture Comparison
- CNN: 49.6%, Transformer: **46.0% ± 7.1%** — closer than Study B. More variance suggests the Transformer is overfitting differently with only 900 samples.

### P15: Cross-Study → 31.3% / 25.2%
| Direction | Acc | Meaning |
|-----------|-----|---------|
| A→B | 31.3% | Under-chin features partially transfer |
| B→A | 25.2% | Throat features barely transfer |

**Different electrode placements create incompatible feature spaces.** This is the strongest evidence that the classification relies on placement-specific onset patterns, not universal articulatory features.

---

## What To Update In Each Paper

### Paper 1 (Study B)
1. **Replace** "99.7% accuracy" → "48.9% ± 3.1% (5-fold stratified CV)"
2. **Add** "58.2% ± 3.1% with multi-session electrode-shifted data"
3. **Add** per-class F1 table (SILENCE=0.80, LEFT=0.36)
4. **Add** confidence gating: "At θ=0.60, accuracy reaches 64.1% on 62% of predictions"
5. **Add** "CNN outperforms LSTM (16.6%) and Transformer (36.4%)"
6. **Reframe** as: "Consumer-grade sEMG achieves 2.9× chance on 6-class onset classification"
7. **Add cross-session paragraph:** electrode shift drops to 22.8%, multi-session recovers to 58.2%

### Paper 2 (Study A)
1. **Update** accuracy to "51.8% ± 2.8% (5-fold CV)"
2. **Add** LOPO showing Overt=31%, Mouthing=50.3%, Exaggerated=44%
3. **Add** cross-study: incompatible feature spaces (25-31% transfer)
4. **Strengthen** the negative finding: under-chin doesn't improve AND transfers poorly

---

## Colab Reference

- **Notebook:** [022826FinalCapstone.ipynb](https://colab.research.google.com/drive/1vByQDtNsl-RW0PLcGGFF6_8GqmgMUbaI?authuser=1)
- **Hardware:** NVIDIA A100-SXM4-80GB · 167 GB RAM · 235 GB disk
- **Runtime:** 29.6 minutes (1,773 seconds)
- **Full results:** [results_summary.md](file:///Users/cvk/Downloads/carl/md-capstonefall25_25TPE/master-progress-list/022826_code/results_summary.md)
- **Script:** [rigorous_eval_colab.py](file:///Users/cvk/Downloads/carl/md-capstonefall25_25TPE/master-progress-list/022826_code/colab_package/rigorous_eval_colab.py)
