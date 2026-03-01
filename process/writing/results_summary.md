# Rigorous ML Evaluation — Complete Results

**Generated:** 2026-02-28 16:00
**Device:** cuda

---
# PAPER 1: Study B (Chin + Throat, 1,500 samples)

## P1. 5-Fold CV: **48.9% ± 3.1%**

| Fold | Accuracy |
|------|----------|
| 1 | 52.0% |
| 2 | 47.3% |
| 3 | 43.7% |
| 4 | 50.0% |
| 5 | 51.3% |

## P2. LOPO: **44.7%**

| Phase | Acc | n |
|-------|-----|---|
| Phase_1_Overt | 50.3% | 300 |
| Phase_2_Whispered | 52.3% | 300 |
| Phase_3_Mouthing | 52.7% | 300 |
| Phase_5_Exaggerated | 35.0% | 300 |
| Phase_6_Covert | 33.3% | 300 |

## P3. Cross-Session: **22.8%**

- Train: 934 (Session A) | Test: 600 (B+C)
- Train acc: 73.9%

## P4. Combined Multi-Session CV: **58.2% ± 3.1%**

| Fold | Accuracy |
|------|----------|
| 1 | 60.3% |
| 2 | 52.4% |
| 3 | 61.2% |
| 4 | 58.0% |
| 5 | 59.2% |

## P5. Hyperparameter Sweep: **50.7%**

| Param | Value |
|-------|-------|
| do | 0.3 |
| hd | 128 |
| lr | 0.001 |
| wd | 0 |

### Top 10

| do | wd | hd | lr | Acc |
|----|----|----|-----|-----|
| 0.3 | 0 | 128 | 0.001 | 50.7% |
| 0.5 | 0 | 128 | 0.001 | 50.1% |
| 0.3 | 0.0001 | 128 | 0.001 | 47.9% |
| 0.3 | 0.001 | 128 | 0.001 | 47.7% |
| 0.5 | 0.0001 | 128 | 0.001 | 46.9% |
| 0.3 | 0 | 64 | 0.005 | 46.3% |
| 0.3 | 0.0001 | 128 | 0.0005 | 45.8% |
| 0.3 | 0.001 | 128 | 0.005 | 45.8% |
| 0.5 | 0 | 128 | 0.005 | 45.3% |
| 0.5 | 0.001 | 128 | 0.001 | 45.0% |

## P6. Architecture Comparison

| Model | Accuracy |
|-------|----------|
| CNN | 49.3% ± 1.8% |
| LSTM | 16.6% ± 0.0% |
| Transformer | 36.4% ± 1.0% |

## P7. Learning Curve

| % Data | n_train | Test Acc | Train Acc |
|--------|---------|----------|-----------|
| 10% | 120 | 24.3% | 78.3% |
| 20% | 240 | 42.0% | 79.6% |
| 40% | 480 | 39.7% | 56.9% |
| 60% | 720 | 51.7% | 78.3% |
| 80% | 960 | 52.3% | 74.7% |
| 100% | 1200 | 51.7% | 71.8% |

## P8. Per-Class Analysis (Held-Out)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| DOWN | 0.44 | 0.48 | 0.46 |
| LEFT | 0.38 | 0.35 | 0.36 |
| NOISE | 0.56 | 0.47 | 0.51 |
| RIGHT | 0.46 | 0.46 | 0.46 |
| SILENCE | 0.76 | 0.84 | 0.80 |
| UP | 0.50 | 0.51 | 0.50 |

## P9. Confidence Gating

| θ | Accuracy | Coverage | n |
|---|----------|----------|---|
| 0.50 | 57.4% | 79.6% | 1194 |
| 0.55 | 61.3% | 69.7% | 1045 |
| 0.60 | 64.1% | 62.1% | 932 |
| 0.65 | 66.5% | 54.8% | 822 |
| 0.70 | 69.3% | 48.3% | 724 |
| 0.75 | 71.1% | 41.7% | 626 |
| 0.80 | 74.0% | 36.4% | 546 |
| 0.85 | 76.7% | 30.3% | 455 |
| 0.90 | 79.4% | 25.6% | 384 |
| 0.95 | 82.5% | 18.3% | 274 |

## P10. Multi-Seed Stability: **51.5% ± 1.0%**

| Seed | Accuracy |
|------|----------|
| 0 | 50.7% |
| 1 | 52.3% |
| 2 | 52.3% |
| 3 | 52.0% |
| 4 | 50.0% |

---
# PAPER 2: Study A (Chin + Under-Chin, 900 samples)

## P11. 5-Fold CV: **51.8% ± 2.8%**

| Fold | Accuracy |
|------|----------|
| 1 | 54.4% |
| 2 | 50.0% |
| 3 | 55.6% |
| 4 | 50.6% |
| 5 | 48.3% |

## P12. LOPO: **41.8%**

| Phase | Acc | n |
|-------|-----|---|
| Phase_1_Overt | 31.0% | 300 |
| Phase_3_Mouthing | 50.3% | 300 |
| Phase_5_Exaggerated | 44.0% | 300 |

## P13. Per-Class Analysis

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| DOWN | 0.38 | 0.33 | 0.36 |
| LEFT | 0.38 | 0.41 | 0.39 |
| NOISE | 0.57 | 0.51 | 0.54 |
| RIGHT | 0.33 | 0.33 | 0.33 |
| SILENCE | 0.73 | 0.75 | 0.74 |
| UP | 0.62 | 0.72 | 0.67 |

## P14. Architecture Comparison

| Model | Accuracy |
|-------|----------|
| CNN | 49.6% ± 2.2% |
| LSTM | 16.7% ± 0.0% |
| Transformer | 46.0% ± 7.1% |

## P15. Cross-Study Comparison

| Train On | Test On | Accuracy |
|----------|---------|----------|
| Study A (chin+underchin) | Study B (chin+throat) | 31.3% |
| Study B (chin+throat) | Study A (chin+underchin) | 25.2% |

---
# P16. Grand Summary

| # | Protocol | Study | Result |
|---|----------|-------|--------|
| 1 | 5-Fold CV | B | 48.9% ± 3.1% |
| 2 | LOPO | B | 44.7% |
| 3 | Cross-Session | B | 22.8% |
| 4 | Combined CV | B | 58.2% ± 3.1% |
| 5 | Best Hyperparams | B | 50.7% |
| 6 | Best Architecture | B | CNN (49.3%) |
| 7 | Learning Curve @100% | B | 51.7% |
| 10 | Multi-Seed | B | 51.5% ± 1.0% |
| 11 | 5-Fold CV | A | 51.8% ± 2.8% |
| 12 | LOPO | A | 41.8% |
| — | Original (train-set) | B | 99.7% |
| — | Paper 2 reported (held-out) | A | ~45% |

## Prior Experiments (from `022426_AccuracyExperiments/`)

| Experiment | Result | Verdict |
|------------|--------|---------|
| SpecAugment | 59.4% | ❌ Hurts |
| Feature Importance | 100% onset | ⚠️ Only first 80ms |
| Permutation Integrity | All pass | ✅ Real signal |
| Raw Signal CNN | 62.0% | = MFCCs |
| Phase-Weighted | 43.3% | ❌ Worse |
| Onset Masking | 17.6% | ❌ Chance |
| Multi-session | 49.7% | ↑ from 21.9% |
| 4-class | 57.0% | ↑ improvement |
| Gate @60% | 77.9% | ✅ Strong |
| Gate @80% | 91.4% | ✅ Very strong |
