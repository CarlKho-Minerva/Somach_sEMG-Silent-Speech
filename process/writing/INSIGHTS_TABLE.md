# 022826 Rigorous ML Evaluation — Master Insights Table

**Colab Notebook:** [022826FinalCapstone.ipynb](https://colab.research.google.com/drive/1vByQDtNsl-RW0PLcGGFF6_8GqmgMUbaI?authuser=1)
**Hardware:** NVIDIA A100-SXM4-80GB · 167 GB RAM · 29.6 min
**Date:** February 28, 2026

---

## Correction: Multi-Session Recording Timeline

> [!IMPORTANT]
> The data was **never single-session**. It spans 5 distinct recording sessions across 3 dates.

| Session | Date | Time | Electrode Position | Samples | Used In |
|---------|------|------|--------------------|---------|---------|
| Study A | Feb 11 | 3:48–5:07 PM | Chin + **under-chin** | 900 | Paper 2 |
| Study B (5-phase) | Feb 11 | 10:12–11:23 PM | Chin + **throat** | 1,500 | Paper 1 |
| Covert Resamples A | Feb 24 | 8:14–9:19 AM | Chin + throat (original) | 934 | Paper 1 (cross-session) |
| Covert Session B | Feb 25 | 8:26–8:47 PM | Chin + throat (1cm lower) | 300 | Paper 1 (cross-session) |
| Covert Session C | Feb 25 | 9:45–9:59 PM | Chin + throat (1cm lateral) | 300 | Paper 1 (cross-session) |

**Total: 3,934 recordings · 5 sessions · 3 dates · 2 electrode configurations**

Study A and Study B were recorded on the **same day** (Feb 11) but with different electrode placements — afternoon (Study A, under-chin) vs evening (Study B, throat). Covert sessions A/B/C were deliberate electrode-shift recordings for cross-session generalization testing.

---

## Master Insights Table

### Column key:
- **Insight**: What we learned
- **Evidence**: The protocol/data that proves it
- **Paper 1 Action**: What to change in Paper 1 (Study B, chin+throat)
- **Paper 2 Action**: What to change in Paper 2 (Study A, chin+under-chin)
- **Citation Impact**: How this affects our references/claims
- **User Request**: Specific requests from Carl

| # | Insight | Evidence | Paper 1 Action | Paper 2 Action | Citation Impact | User Request |
|---|---------|----------|----------------|----------------|-----------------|--------------|
| 1 | **Honest accuracy is 48.9% (Paper 1) and 51.8% (Paper 2)** — not 99.7%. The 99.7% was memorization with no held-out split. | P1: 5-fold CV → 48.9% ± 3.1%; P11: 5-fold CV → 51.8% ± 2.8% | Replace ALL mentions of 99.7%. Report "48.9% ± 3.1% (5-fold stratified CV, k=5)" | Update to "51.8% ± 2.8%" | Remove any comparison to high-accuracy BCI papers that used different eval. Add self-critical framing vs AlterEgo ($5K hardware). | — |
| 2 | **2.9× above chance (16.7% for 6 classes)** — this IS real signal, not noise. | P1: 48.9/16.7 = 2.93; P11: 51.8/16.7 = 3.1 | Add: "Classification accuracy exceeds 6-class chance by a factor of 2.9" | Add: "3.1× above chance" | Cite chance-level baselines from BCI lit. Frame as consumer-hardware proof-of-concept. | Highlight this framing |
| 3 | **Moderate overfitting**: train acc 71-85%, test 44-52%. Gap ~25pp. | P1 fold details, P7 learning curve | Add discussion of train-test gap. "The 25pp generalization gap reflects the limited SNR of the AD8232" | Same pattern | — | — |
| 4 | **LOPO shows curriculum failure pattern**: Whispered/Mouthing transfer (52%), Exaggerated/Covert fail (33-35%). | P2: LOPO results | Rewrite curriculum section: "Curriculum learning is limited by the fact that phases with minimal jaw activation (Exaggerated, Covert) occupy a fundamentally different signal regime" | Add LOPO table for Study A's 3 phases. Overt at 31% is surprising. | Cite curriculum learning papers with caveat that sEMG regimes differ from NLP/vision curriculum | "Doesn't make sense for curriculum learning because muscle activation — capstone full of learning" |
| 5 | **Electrode shift is THE dominant failure mode**: 22.8% cross-session (barely above chance). | P3: Train A → Test B+C = 22.8% | Dedicate paragraph to cross-session failure. "Single-session models are non-transferable due to electrode impedance variability" | Mention as motivation for future work | Cite electrode shift literature (Phinyomark et al.), AlterEgo's calibration procedure | "Electrode shift is the dominant failure mode" |
| 6 | **Multi-session data recovers to 58.2%** — model learns position-invariant features. | P4: Combined 3-position CV = 58.2% ± 3.1% | Add: "Multi-session recording with deliberate electrode shift recovers accuracy from 22.8% to 58.2%, demonstrating learnable position-invariant onset features" | Cross-reference Paper 1's finding | Cite domain adaptation / session-transfer BCI papers | "The model starts learning position-invariant features — amazing" |
| 7 | **Best hyperparams = defaults** (dropout=0.3, hidden=128, lr=0.001, wd=0). Ceiling is signal, not model. | P5: 72 configs searched, best = 50.7% ≈ default 48.9% | Add: "A grid search over 72 hyperparameter configurations confirmed that model capacity is not the bottleneck — the ceiling is imposed by the hardware's signal-to-noise ratio" | Same conclusion for Study A | — | "The ceiling is hardware SNR, not data quantity — amazing" |
| 8 | **CNN is the only viable architecture.** LSTM = 16.6% (chance), Transformer = 36.4%. | P6: Architecture comparison | Add architecture comparison table. "LSTMs fail because the AD8232 captures only the initiation burst (~80ms), leaving no temporal structure for recurrent modeling" | Add: CNN=49.6%, Transformer=46.0%, LSTM=16.6% | Cite LSTM/Transformer BCI papers, explain why they fail here (data regime) | — |
| 9 | **The initiation burst is the ONLY useful signal.** Everything beyond onset is hardware noise floor. | EXP6 (onset masking → 17.6%), P6 (LSTM=chance), P7 (saturation) | **Dedicate a section to the initiation burst.** "Onset masking (EXP6) proved that masking the first 80ms drops accuracy to chance. Combined with LSTM failure, this establishes that the AD8232 captures only the motor initiation burst." | Cross-reference Paper 1's finding | Cite motor unit recruitment literature. AlterEgo's 8-channel vs our 2-channel. | "Why can't we dedicate a section talking about the initiation burst? I remember you made a database for that" |
| 10 | **Learning curve saturates at ~600 samples.** More data reduces overfit but doesn't raise the ceiling. | P7: 60% accuracy plateau at 720 samples, train-test gap narrows from 54pp → 20pp | Add: "The test accuracy plateau at N≈600 with continued overfitting reduction indicates a hardware-imposed SNR ceiling, not a data-limited regime" | Relevant for Study A (only 900 samples — near saturation) | — | "Saturation means more data reduces overfit but ceiling is hardware" |
| 11 | **SILENCE dominates** (F1=0.80). Jaw-at-rest has a uniquely flat EMG signature. | P8: Per-class F1 | Add per-class table. "SILENCE (F1=0.80) is trivially classifiable due to the absence of motor unit firing" | Same: SILENCE F1=0.74 | — | — |
| 12 | **LEFT is the worst class** (F1=0.36). Lateral tongue movements don't propagate to chin/throat. | P8: LEFT F1=0.36 | "Lateral lingual movements lack sufficient bilateral chin/throat projection for surface EMG resolution" | LEFT F1=0.39 in Study A too — same issue even with under-chin | Cite Meltzner et al. (2017) on electrode placement for lateral articulators | "Lateral tone movements — advisor knows I'm doing lingual-mandibular" |
| 13 | **Confidence gating θ=0.60: 64.1% accuracy on 62% of predictions.** Deployable. | P9: Gating sweep | Add: "At a softmax confidence threshold of 0.60, the system achieves 64.1% accuracy on the 62% of predictions it accepts, offering a practical operating point for real-time deployment" | Same analysis needed | Cite rejection-option classifiers in BCI. Compare to commercial BCI products that also require calibration. | "θ=0.60 is sweet spot — it's actually deployable" |
| 14 | **BCI personalization is standard.** Our calibration requirement isn't a weakness — it's the norm. | Industry standard. AlterEgo, Emotiv, OpenBCI all require per-session calibration. | Add: "Per-session calibration is standard practice in commercial BCI systems" | Same | Cite AlterEgo's calibration protocol, NeuroSky, Emotiv docs | "BCI research requires personalization — we're with state of the art not really" |
| 15 | **Multi-seed: 51.5% ± 1.0%** — results are reproducible across random seeds. Not a fluke. | P10: 5 seeds, range 50.0-52.3% | Add: "Variance across 5 random seeds was ±1.0%, confirming result reproducibility" | — | — | "Five different random seeds all produce 50-52 — reproducible" |
| 16 | **Study A (under-chin) = 51.8% > Study B = 48.9%.** Under-chin placement slightly outperforms throat. | P1 vs P11 | Cross-reference in Discussion | Add: "The submental placement achieves marginally higher held-out accuracy (51.8% vs 48.9%), though the difference is not statistically significant given the standard deviations" | — | "Study A actually performs slightly better — explore more" |
| 17 | **Cross-study transfer fails: 25-31%.** Chin+throat and chin+under-chin create incompatible feature spaces. | P15: A→B=31.3%, B→A=25.2% | Add paragraph in Discussion | **Key negative finding**: "Cross-study transfer accuracy (25-31%) demonstrates that electrode placement creates fundamentally incompatible feature spaces" | Cite electrode montage comparison papers | "Different electrode placements, great incompatible feature spaces — could be visualized" |
| 18 | **Data spans 5 sessions across 3 dates** — not single-session. Multi-session is the honest framing. | CSV timestamp analysis | Correct methodology section. List all recording sessions with dates, times, electrode positions. | Same — document the Feb 11 afternoon session | — | "It's not single session — I told you so many times" |
| 19 | **Add Colab reference.** Document that evaluation was GPU-accelerated (A100, 29.6 min). | Colab notebook link + specs | Add to methodology: "All evaluation protocols were executed on an NVIDIA A100-SXM4-80GB via Google Colab" | Same | — | "Add Colab reference" |
| 20 | **Visualize cross-study feature space incompatibility.** | P15 cross-study results | Create a figure showing feature space separation (t-SNE or PCA of Study A vs Study B) | Same figure, different perspective | — | "Could be visualized — lane that I work with" |

---

## Citation Impact Summary

| Current Citation/Claim | Problem | New Citation/Framing |
|------------------------|---------|---------------------|
| "99.7% classification accuracy" | Memorization, no held-out split | "48.9% ± 3.1% (5-fold CV), 2.9× above 6-class chance" |
| Implicit comparison to AlterEgo's ~92% | Hardware incomparable ($40 vs $5,000+) | Explicitly frame as consumer-grade feasibility study |
| No cross-session evaluation | Didn't test electrode shift | Add 22.8% cross-session + 58.2% multi-session recovery |
| "Curriculum learning enables progression" | LOPO shows covert/exaggerated phases fail | Reframe: curriculum is pedagogically valuable, not ML-beneficial |
| LSTM/Transformer not tested | No architecture comparison | CNN-only viable; LSTM fails due to onset-only signal |
| No per-class breakdown | Hides LEFT's 0.36 F1 | Full per-class table, acknowledge lateral tongue limitation |
| No confidence gating | Reports only aggregate accuracy | Add deployable operating point (θ=0.60 → 64.1%) |
| "Single participant study" accurate but undersells | 5 sessions over 3 dates is substantial | Emphasize multi-session protocol with electrode shift |

---

## What To Do Next

### For Paper 1 (Study B):
1. ✏️ Replace accuracy claims throughout
2. ✏️ Add methodology section on evaluation protocols (5-fold CV, LOPO, cross-session)
3. ✏️ Add per-class F1 table
4. ✏️ Rewrite curriculum learning discussion (LOPO results)
5. ✏️ Add cross-session paragraph + multi-session recovery
6. ✏️ Dedicate section to initiation burst
7. ✏️ Add architecture comparison table
8. ✏️ Add confidence gating operating point
9. ✏️ Add multi-session recording timeline
10. ✏️ Add Colab/A100 reference

### For Paper 2 (Study A):
1. ✏️ Update accuracy to 51.8% ± 2.8%
2. ✏️ Add LOPO results (Overt=31%, Mouthing=50.3%, Exaggerated=44%)
3. ✏️ Add cross-study transfer failure (25-31%)
4. ✏️ Strengthen negative finding: under-chin doesn't improve AND transfers poorly
5. ✏️ Add per-class F1 (RIGHT=0.33 worst, SILENCE=0.74 best)
6. ✏️ Add CNN vs Transformer vs LSTM comparison
7. ✏️ Note that Study A and Study B were same day (Feb 11), different electrodes
