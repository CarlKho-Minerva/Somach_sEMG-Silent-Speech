# Cross-Paper Consistency Audit: Paper 1 vs Paper 2

**Date:** March 1, 2026
**Auditor:** AI Review Agent (Claude)
**Scope:** Line-by-line comparison of `paper1_UPDATED/main.tex` (P1, 832 lines) and `paper2_UPDATED/main.tex` (P2, 619 lines) against actual codebase parameters.

---

## CRITICAL: Parameters That Must Be Fixed in BOTH Papers

These values appear in the papers but **do not match the actual code** (`6-extract-features.py`, `5-preprocess.py`, `exp1_specaugment/run.py`, `exp7_multisession/precompute.py`):

### 1. `n_fft` — FFT Window Size

| Source | Value | Window Duration | Overlap |
|--------|-------|----------------|---------|
| **Actual Code** (`6-extract-features.py` L22) | **N_FFT = 128** | **512 ms** | **80.5%** |
| P1 Table 5 (L263) | n_fft = 64 | 256 ms | ~60% |
| P2 §2.4 (L193) | FFT window = 64 | 256 ms | ~60% |
| Extension.md Phase 2 MFCC code | n_fft = 64 | 256 ms | — |

**Impact:** ALL reported results (48.9%, 51.8%, 58.2%, confidence gating, LOPO, etc.) were generated with N_FFT=128, not 64. The papers misreport the fundamental feature extraction parameter.

**Fix:** Change to `n_fft = 128` in both papers. Update window duration to 512 ms and overlap to ~80%.

### 2. Bandpass Filter Low Cutoff

| Source | Value |
|--------|-------|
| **Actual Code** (`5-preprocess.py` L19) | **LOW_CUT = 1.0 Hz** |
| P1 §3.3 (L242) | "Bandpass (1.3–50 Hz)" |
| P2 §2.4 (L189) | "Bandpass filter: 1.3–50 Hz" |
| AlterEgo (Kapur 2018) | 1.3–50 Hz |

**Impact:** Papers state 1.3 Hz (copied from AlterEgo), but code uses 1.0 Hz. The difference is minor but the mismatch is a factual error. The lowered cutoff slightly changes the DC drift removal behavior.

**Fix:** Change to "1.0–50 Hz" in both papers, or add a note explaining the deviation from AlterEgo's parameters.

### 3. Time Frames (T)

| Source | Value |
|--------|-------|
| **Actual Code** (`6-extract-features.py` L23) | **TARGET_TIMESTEPS = 100** |
| P1 §3.4 text (L268) | "~15 time frames × 13 coefficients" |
| P2 §2.4 text (L194) | "Output: X ∈ ℝ^{T × 26} per sample" |

**Impact:** With N_FFT=128 and HOP=25 on a 375-sample (1.5s) signal: `T = 1 + (375 - 128) / 25 ≈ 10.88 → 11 frames`, then padded to **100** via `pad_or_truncate()`. Papers say T ≈ 15 (which corresponds to n_fft=64: `1 + (375 - 64) / 25 ≈ 13.44`). Neither matches reality.

**Fix:** State that raw MFCC extraction yields ~11 frames per channel, **padded/truncated to 100 time steps** for fixed CNN input. This is actually an important detail — the zero-padding to 100 means 90% of the input is padding.

### 4. Overlap Percentage

| Source | Value |
|--------|-------|
| **Actual** (N_FFT=128, HOP=25) | **(128−25)/128 = 80.5%** |
| P1 Table 5 (L263) | "~60% overlap" |
| P2 §2.4 (L191) | "~60% overlap" (implicit from n_fft=64, hop=25) |

**Fix:** Change overlap figure to ~80%.

### 5. MacBook Air RAM (P1 Only)

| Source | Value |
|--------|-------|
| **Actual hardware** (user confirmed) | **8 GB** |
| P1 Table 8, Training Configuration (L413) | "Apple M2 MacBook Air, 16 GB" |

**Fix:** Change to "8 GB" in P1 Table 8.

---

## P1-to-P2 Cross-Paper Inconsistencies

### 6. P2 Still Has "80/20 Stratified" in Training Config Table

- **P2 Table 5** (L210, `\label{tab:training}`): Lists `Train/test split` as `80/20 stratified`
- **P2 text** (L223): "Note on evaluation: Initial experiments for Study A used a 20% held-out test split (180 test samples)... Subsequently, both studies were evaluated under identical 5-fold stratified CV..."
- **P1**: Does NOT have `80/20 stratified` in its training config table (correctly removed in prior fix round)

**Fix:** Remove the `80/20 stratified` row from P2's Table 5, or clearly label it as "Initial (superseded)" and add a `5-fold stratified CV` row as the primary protocol.

### 7. P2 Figure Captions Reference Obsolete 80/20 Split

- **P2 Figure 2 caption** (L284): "Confusion matrix on the 20% held-out test set (n = 180)"
- **P2 Figure 3 caption** (L230): "training vs. test accuracy across epochs"

The confusion matrix figure still references the old 80/20 evaluation. If the figure was generated from the 5-fold CV results, the caption is wrong. If the figure is from the old 80/20 split, it's inconsistent with the paper's primary evaluation protocol.

**Fix:** Either regenerate the confusion matrix from 5-fold CV aggregated predictions, or clearly label the figure as "initial 80/20 evaluation shown for visual illustration; quantitative results use 5-fold CV."

### 8. Per-Phase Test Accuracy Table (P2) Uses 80/20 Split

- **P2 Table 9** (L299): "Per-phase test accuracy" with `Test n = 60` per phase (consistent with 20% of 300 = 60)
- This table uses the OLD 80/20 split convention, not 5-fold CV

**Fix:** Either regenerate per-phase results using 5-fold CV held-out predictions, or add a note: "Per-phase breakdown reported from initial 80/20 evaluation; primary quantitative results use 5-fold CV."

### 9. Receptive Field Calculation (P1)

- **P1** §5.1 (L399): "receptive field spans ~7 time frames (~700 ms)"
- With T=100 (padded), this is plausible for the non-padded portion
- But with T≈15 (as papers claim), 7/15 = 46% of the input is covered by RF — different conclusion than 7/100 = 7%
- **The receptive field calculation should use the actual TIME interpretation, not frame count.** With N_FFT=128 and HOP=25: each frame spans 512ms, each hop is 100ms. RF of 7 frames = first frame starts at t=0 (covers 0–512ms), 7th frame starts at t=600ms (covers 600–1112ms). True RF in time = 0–1112ms ≈ 1.1 seconds.

**Fix:** Correct to "~7 time frames (~1.1 seconds with 512ms windows and 100ms hop), effectively spanning the entire ~1.5-second recording."

### 10. N_MELS Not Explicitly Stated in P1 Table 5

- **P1 Table 5**: Lists n_mfcc=13, n_fft=64, hop=25, sr=250 but **does not list n_mels=26**
- **P2 §2.4**: Lists "26 mel bands" — correct
- **Actual code**: `N_MELS=26` (in experiment scripts), librosa default is 128

**Fix:** Add `n_mels = 26` row to P1's MFCC parameter table for completeness.

### 11. Technical Constraint Note (P2 Only)

- **P2** §2.4 (L192–193): "At 250 Hz, a 256 ms window (n_fft = 64) contains only 64 samples. Extracting 26 mel bands from such sparse data introduces spectral leakage and aliasing..."
- This note is **wrong twice**: (a) actual window is 512ms/128 samples, and (b) 128 samples gives 65 FFT bins (N/2+1), which IS sufficient for 26 mel bands with much less leakage than 64 samples (33 bins).

**Fix:** Correct to: "At 250 Hz, a 512 ms window (n_fft = 128) yields 65 frequency bins. With 26 mel bands, each band averages ~2.5 bins — a sparse but functional resolution. This is substantially better than n_fft=64 (33 bins, ~1.3 bins/band), which would introduce significant spectral leakage."

---

## Minor Cross-Paper Consistency Issues

### 12. Citation Style for Kapur et al.

- **P1 §1.2**: "AlterEgo (Kapur, Kapur, & Maes, 2018)" — expanded form
- **P2 §1.1**: "AlterEgo system (Kapur et al., 2018)" — abbreviated
- Both are acceptable within their own papers, but ideally should be consistent across the companion paper pair.

### 13. Table Numbering Collision

- Both papers have `\label{tab:curriculum}` — P1 uses it for the 5-phase design (Table 4), P2 uses it for the 3-phase protocol (Table 3). Not an issue unless they're ever compiled together, but worth noting.

### 14. Study B Recording Time

- **P1** §4.4 Table 6: "Feb 11, 22:12–23:23, ~71 minutes"
- **P2** §1.3 (L127): "Study B's initial 5-phase session was recorded on the evening of the same day (22:12–23:23, ~71 minutes)"
- ✅ Consistent

### 15. 5-Fold CV Results Cross-Referenced

- **P1**: "48.9% ± 3.1%"
- **P2** Table 7 (L316): "48.9% ± 3.1%" for Study B
- ✅ Consistent

### 16. Cross-Study Transfer Numbers

- **P1** Abstract: "cross-study transfer of only 25–31%"
- **P2** Table 8 (L349): "Train A → Test B: 31.3%, Train B → Test A: 25.2%"
- ✅ Consistent

---

## Summary: Required Fixes

| # | Paper(s) | Severity | Fix |
|---|----------|----------|-----|
| 1 | **P1 + P2** | 🔴 CRITICAL | n_fft: 64 → 128 (512ms window) |
| 2 | **P1 + P2** | 🟡 HIGH | Bandpass: 1.3 → 1.0 Hz |
| 3 | **P1 + P2** | 🟡 HIGH | T ≈ 15 → "padded to 100 time steps" |
| 4 | **P1 + P2** | 🟡 HIGH | Overlap: ~60% → ~80% |
| 5 | **P1 only** | 🟡 MEDIUM | RAM: 16 GB → 8 GB |
| 6 | **P2 only** | 🟡 MEDIUM | Remove/update 80/20 from training table |
| 7 | **P2 only** | 🟡 MEDIUM | Fix confusion matrix caption |
| 8 | **P2 only** | 🟡 MEDIUM | Fix per-phase table split convention |
| 9 | **P1 only** | 🟢 LOW | Correct receptive field to ~1.1s |
| 10 | **P1 only** | 🟢 LOW | Add n_mels=26 to MFCC table |
| 11 | **P2 only** | 🟡 HIGH | Fix technical constraint note |

---

*Generated: March 1, 2026*
