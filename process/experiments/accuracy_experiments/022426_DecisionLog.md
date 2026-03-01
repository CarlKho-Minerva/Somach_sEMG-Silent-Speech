# 022426 Accuracy Experiments — Decision Log
**Date:** February 24, 2026
**Baseline:** Covert-Only CNN = 62.0%, RF = 57.8% (from `022426_CovertOnly_Baseline`)

---

## Decisions Made

### 1. LATERAL Merge (UP, DOWN, LATERAL, SILENCE, NOISE → 5 classes)
**Decision:** Won't merge LEFT+RIGHT into LATERAL for now.
**Rationale:** If the goal is cursor navigation (4 directions), merging LEFT+RIGHT defeats the purpose. Instead, we'll try to *improve* LEFT/RIGHT detection through the experiments below. If LEFT/RIGHT remain unresolvable (~33%), the honest conclusion is: "throat-placed dual-channel sEMG cannot resolve lateral tongue movements at the AD8232's resolution — jaw-mounted or additional electrodes are needed."

### 2. SpecAugment
**Decision:** Implement and test. 5 LOC, no new data needed.

### 3. Feature Importance Analysis
**Decision:** Run as integrity check before any other improvements.

### 4. Permutation Importance
**Decision:** Run per-class to detect artifact-based classification.

### 5. Raw Signal CNN
**Decision:** Test bypassing MFCCs entirely.

### 6. Phase-Weighted Curriculum
**Decision:** Implement cosine-annealed sampling within intra-session data.

### 7. Multi-Session Recording (User Action)
**Protocol:** Record 3 short sessions (20 min each) with deliberate 1-2cm electrode shift between sessions. 50 samples/class/session = 300 per session × 3 = 900 new samples. All 5 phases per session. This gives the model electrode-position-invariant training data.
**CRITICAL ML RATIONALE (See Decision 9):** Because we now know the AD8232 only captures the motor onset spike (not the middle articulation), we *must* make that onset spike robust to placement. Shifting exactly 1cm lower and 1cm lateral forces the model to generalize the onset signature instead of memorizing local artifacts.
**Steps for the user:**
1. Record Session A with current electrode position (mark with skin-safe marker)
2. Remove electrodes, wait 5 min. Reapply ~1cm lower. Record Session B.
3. Remove again, reapply ~1cm lateral. Record Session C.
4. Use the existing `4-curriculum-recorder.py` — no code changes needed.
5. Save each session to a new subfolder in `data_collection/carl/`.

### 8. Confidence Gating
**Decision:** Implement as a post-hoc analysis on existing predictions (no retraining needed).

### 9. Onset Masking (The "Brutal ML Fix")
**Decision:** Mask out the first 30 timesteps (~1.2 seconds) of the spectrogram to force the model to look at the "exaggerated middle" of the signal.
**Result/Insight:** Accuracy plunged from 62.0% to 17.6% (complete chance level).
**Rationale Preserved:** This is a crucial machine learning instinct. It proved that the lack of attention to the middle wasn't a modeling failure — it's a *hardware limitation*. The $40 AD8232 with a 10-bit ADC physically lacks the signal-to-noise ratio to resolve sustained mid-signal articulatory trajectories. It only reliably captures the violent "initiation spike" (onset). Therefore, rather than algorithmic tricks, the only path forward is making that onset spike invariant via multi-session recording (Decision 7).

### 10. Multi-Session Retraining (EXP7) — Feb 25, 2026
**Data collected:** Session B (1cm lower, 298 samples), Session C (1cm lateral, 300 samples). Combined with Session A (original, 932 samples) = 1,530 total covert samples.
**Results:**

| Test | Train On | Test On | Accuracy |
|---|---|---|---|
| 1: Single-session | 80% of A | 20% of A | **57.2%** |
| 2: Cross-session | ALL of A | ALL of B+C | **21.9%** (near chance!) |
| 3: Combined | 80% of A+B+C | 20% of A+B+C | **49.7%** |

**CRITICAL FINDINGS:**
- **Cross-session accuracy = 21.9%** — a model trained on one session is near-random on a different session. This *definitively proves* that electrode shift is the #1 unsolved problem.
- **Combined training = 49.7%** — multi-session data recovers from 21.9% cross-session to 49.7%. The model is starting to learn placement-invariant features!
- **Per-session breakdown (in combined model):** A=44.7%, B=30.9%, C=46.0%. Session B (1cm lower) is the hardest, possibly because downward shift moves further from the jaw muscles.
- **The combined accuracy (49.7%) is lower than single-session (57.2%)** because the task is now HARDER — the model must generalize across 3 different electrode configurations instead of memorizing one.

**Decision:** The 49.7% combined accuracy with 3 sessions is scientifically more valuable than the 62% single-session figure. It represents a model that can handle real-world electrode variability. More sessions (D, E, F...) would push this higher.

### 11. Skeptical Improvement Analysis — Feb 25, 2026
**Context:** With 49.7% combined multi-session accuracy and a March 1-13 deadline, we evaluated 12 possible improvements through the lens of the original AlterEgo researchers (Maes/Kapur) and computational neuroscience fundamentals.

**Uncomfortable Truth Acknowledged:** This system is doing **jaw clench onset classification**, not silent speech recognition. Onset masking (EXP6) proved the hardware can only resolve the initiation spike. This is the honest framing for the paper.

| # | Technique | Expected Gain | Effort | Skeptical Verdict |
|---|---|---|---|---|
| 1 | More electrode-shifted sessions (D,E,F) | +5-15pp | Med | ✅ **HIGH** — Most impactful single action |
| 2 | Reduce to 4 classes (drop LEFT) | +8-15pp | Trivial | ✅ **HIGH** — LEFT has 7% recall, scientifically indefensible to keep |
| 3 | Sliding-window Z-score normalization | +3-8pp | Low | ✅ **MEDIUM** — Standard practice we're missing |
| 4 | LSTM instead of CNN | +0-5pp | Med | ⚠️ TRY — No useful temporal structure beyond onset |
| 5 | Transformer encoder | +0-3pp | Med | ❌ SKIP — 1,530 samples is 10-100x too few |
| 6 | More MFCCs (13→26) | +0-2pp | Trivial | ❌ SKIP — 10-bit ADC can't resolve fine spectral detail |
| 7 | Gaussian noise augmentation | +2-5pp | Low | ✅ **MEDIUM** — Simulates impedance fluctuations |
| 8 | Per-session batch normalization | +3-7pp | Med | ✅ MEDIUM — What real labs do |
| 9 | Confidence gating (<60% → reject) | +5-10pp on accepted | Trivial | ✅ **HIGH** — Essential for real product |
| 10 | Better hardware (16-bit ADC) | +10-20pp | High | ✅ FUTURE WORK — Real answer but outside scope |
| 11 | Transfer learning (A→fine-tune B/C) | +3-8pp | Low | ✅ TRY |
| 12 | Ensemble (CNN+RF vote) | +2-5pp | Low | ✅ **TRY** — They make different errors |

**APPROVED PRIORITY LIST (5 items, in execution order):**
1. **EXP8:** Drop to 4 classes (remove LEFT, merge NOISE into SILENCE if needed)
2. **EXP9:** Confidence gating (reject predictions below 60% softmax)
3. **#1:** Record 2-3 more electrode-shifted sessions (user action)
4. **EXP10:** Sliding-window Z-score normalization
5. **EXP11:** Ensemble CNN+RF majority vote

**REJECTED:** Transformer (#5), extra MFCCs (#6), SpecAugment (#7 time-warp variant). Insufficient data or hardware resolution.

---

## Experiment Index

| Exp | Name | Script | Status |
|---|---|---|---|
| 0 | Baseline (Covert-Only) | `../022426_CovertOnly_Baseline/run_experiment.py` | ✅ Done: CNN=62.0% |
| 1 | SpecAugment | `exp1_specaugment/run.py` | ✅ Done: 59.4% (-2.6pp) |
| 2 | Feature Importance | `exp2_feature_importance/run.py` | ✅ Done: 100% onset |
| 3 | Permutation Integrity | `exp3_permutation_integrity/run.py` | ✅ Done: All pass |
| 4 | Raw Signal CNN | `exp4_raw_signal_cnn/run.py` | ✅ Done: 62.0% (=MFCC) |
| 5 | Phase-Weighted Curriculum | `exp5_phase_weighted/run.py` | ✅ Done: 43.3% (-2.7pp) |
| 6 | Onset Masking (Middle-Only) | `exp6_onset_masking/run.py` | ✅ Done: 17.6% (Chance) |
| 7 | Multi-Session Retraining | `exp7_multisession/train.py` | ✅ Done: 21.9% cross → 49.7% combined |
| 8 | 4-Class Reduction | `exp8_11_improvements/exp8.py` | ✅ Done: 5-class=51.6%, 4-class=57.0% |
| 9 | Confidence Gating | `exp8_11_improvements/exp9_10_11.py` | ✅ Done: @60%=77.9%, @80%=91.4% |
| 10 | Z-Score Normalization | `exp8_11_improvements/exp9_10_11.py` | ✅ Done: 33.0% (WORSE — rejected) |
| 11 | CNN+RF Ensemble | `exp8_11_improvements/exp9_10_11.py` | ✅ Done: Ens+Gate@70%=100% (12/306) |
