# Oral Defense Preparation Guide

**Carl Vincent Ladres Kho — Capstone Oral Defense**
**Date:** March 1, 2026
**Papers:**
- P1: "Curriculum Learning for Silent Speech Classification: A Proof-of-Concept $40 Two-Channel sEMG System"
- P2: "Lingual-Mandibular Electrode Configuration for Silent Speech: Throat Sensor Necessity on Consumer-Grade ADCs"

---

## 1. Your 90-Second Elevator Pitch

> "I built a $40 silent speech interface — a device that reads what you're saying without you making any sound — using two AD8232 ECG sensors and an ESP32 microcontroller. The total hardware cost is 25× less than research systems like MIT's AlterEgo. The key finding: on consumer-grade 12-bit ADCs, the system can only detect the initial 80-millisecond neuromuscular onset burst — not continuous articulation. Using curriculum learning across five speech intensity phases, I achieved 48.9% ± 3.1% held-out accuracy on 6 classes in a single session, 58.2% with multi-session training, and 64.1% with confidence gating — all rigorously evaluated with 5-fold stratified blocked CV on an A100 GPU. A concurrent control study proves that throat electrode placement is necessary — it provides orthogonal neuroanatomical information that chin-only placement cannot."

---

## 2. Key Numbers to Memorize

| Metric | Value | Context |
|--------|-------|---------|
| Hardware cost | **$40** | 25× less than AlterEgo (~$1K) |
| ADC resolution | **12-bit** | vs 24-bit in research (4096:1 resolution difference) |
| Channels | **2** | vs 7 in AlterEgo (3.5× fewer) |
| AD8232 gain | **~1000×** | vs 24× in AlterEgo (compensates for lower ADC res) |
| Bandpass | **1.0–50 Hz** | AD8232 hardware: 0.5–40 Hz |
| Onset burst duration | **~80 ms** | Contains 100% of discriminative information |
| **Single-session CV** | **48.9% ± 3.1%** | 5-fold stratified blocked CV, 2.9× above 16.7% chance |
| **Multi-session** | **58.2% ± 3.1%** | 3 sessions with deliberate electrode repositioning |
| **Confidence gating** | **64.1% @ θ=0.60** | On 62% of accepted predictions |
| Training accuracy | **99.7%** | CNN memorizes 1,500 samples — NOT the honest figure |
| Study A (chin+under-chin) | **51.8% ± 2.8%** | 3 phases, 900 samples |
| Cross-study transfer | **25–31%** | Proves incompatible feature spaces |
| Electrode shift degradation | **22.8%** | 1 cm shift → near-chance |
| Model parameters | **~47K** | Deliberately small to prevent overfitting |
| Training time | **~3 min** | 100 epochs on M2 MacBook Air |
| Vocabulary | **6 classes** | UP, DOWN, LEFT, RIGHT, SILENCE, NOISE |
| MFCC params (actual) | **n_fft=128, hop=25, n_mfcc=13, n_mels=26** | 512ms window, 100ms hop, ~80% overlap |
| AlterEgo accuracy | **92.01%** | On 10-digit vocabulary (NOT 20 words) |
| Nieto inner speech EEG | **~40%** | 4 classes, $5K hardware, 136 channels |

---

## 3. Anticipated Questions & Strong Answers

### Q1: "Why only 49%? That seems low."

**Strong answer:**
> "49% is the honest figure — 2.9× above the 16.7% chance baseline for 6 classes. The previous 99.7% training accuracy was memorization, which is why rigorous held-out evaluation was essential. The ceiling is imposed by hardware signal-to-noise ratio, not model capacity. I proved this three ways:
> 1. A 72-configuration hyperparameter sweep found the default near-optimal — no tuning can overcome the hardware limit.
> 2. The learning curve saturates at ~600 samples — more data doesn't help.
> 3. Three architecture comparisons (CNN, LSTM, Transformer) all hit the same ceiling.
>
> Importantly, with multi-session training (58.2%) and confidence gating (64.1%), the system reaches a deployable operating point for command interfaces. And at $40, this is the cheapest SSI ever reported that's above chance with proper evaluation."

### Q2: "Why not just buy better hardware?"

**Strong answer:**
> "That's exactly the point of this work — to establish the operational boundaries of consumer-grade hardware. The $40 price point matters for accessibility: research hardware costs $1K–$5K, making SSI technology inaccessible to most potential users (e.g., people with speech disabilities in developing countries). This work maps exactly where the ceiling is, so future designers know what's achievable at what price point. The contribution isn't beating AlterEgo — it's a quantitative analysis of how far you can push cheap components."

### Q3: "How do you know it's the onset burst and not something else?"

**Strong answer:**
> "Three converging pieces of evidence:
> 1. **Onset masking experiment (EXP6):** I zeroed out the first 80ms and classified the rest. Accuracy dropped to 17.6% — chance level. 100% of the information is in that burst.
> 2. **LSTM failure:** The LSTM achieved exactly 16.6% (chance). LSTMs model sequential dependencies, but there's no sequence to model — after 80ms the signal returns to the noise floor.
> 3. **Feature importance analysis (EXP2):** The model's predictive power maps entirely to the first ~80ms.
>
> Physiologically, this is the motor unit recruitment burst — the synchronous firing at the start of voluntary contraction. Research hardware can capture the sustained articulatory trajectory that follows; our 12-bit ADC cannot."

### Q4: "Why is electrode placement such a big deal? Can't you compensate in software?"

**Strong answer:**
> "No — and I proved it quantitatively. Training on Study A (chin+under-chin) and testing on Study B (chin+throat) gives only 25–31% accuracy. The two placements capture fundamentally different physiological signals — chin+throat detects laryngeal elevation via CN X innervation, while chin+under-chin detects digastric/mylohyoid via CN V. These are different neural pathways.
>
> On research hardware, you can compensate with more channels (7+ electrodes covering multiple regions). With only 2 channels and 12-bit resolution, you're locked into whichever feature axis your electrode placement defines. The contribution is proving this empirically and providing design guidance: throat sensors are necessary for multi-session robustness."

### Q5: "Why did you use MFCCs on EMG? That's designed for audio."

**Strong answer:**
> "You're right — MFCCs were designed for acoustic speech and encode spectral envelope shape, not muscle activation patterns. It's a domain mismatch. But Kapur et al. empirically validated MFCCs as effective features for sEMG classification in AlterEgo, and the key property I need is **amplitude invariance** — the same word at different volumes (overt vs. covert) should produce similar features. That's exactly what MFCCs provide. Also, at 250 Hz sampling (125 Hz Nyquist), the Mel scale is essentially linear, so the extraction is effectively a linear filterbank analysis."

### Q6: "The curriculum learning — does it actually help?"

**Strong answer:**
> "The LOPO (leave-one-phase-out) results are nuanced. Middle phases (Mouthing, Whispered) transfer well to each other (~52%). But the extreme phases fail: Exaggerated (35%) causes ADC saturation, and Covert (33%) is below the noise floor. So curriculum learning doesn't 'compensate for' noise in a physics sense — it enables the CNN to learn a shared onset representation across SNR regimes. I'd argue its pedagogical value (helping me understand the system's failure modes) exceeds its ML accuracy improvement. The honest finding is that the 'Palm Pilot strategy' — teaching the USER to be consistent — is more effective than any model architecture change."

### Q7: "n=1. Can you generalize?"

**Strong answer:**
> "No — and I say so explicitly. All data is from one male, 22-year-old participant. The multi-session protocol (5 sessions, 3 days) provides temporal diversity but not anatomical generalization. This is a proof-of-concept study establishing methodology and boundaries. The replication package (open-source at somach.vercel.app) enables others to reproduce and extend. The contribution is the framework and the quantitative analysis, not claims about population-level performance."

### Q8: "Why did you choose n_fft=128 (512ms) window size?"

**Strong answer:**
> "I optimized the window size to balance frequency resolution with temporal stationarity. A 128-sample window (512ms at 250Hz) provides 65 FFT bins, which is sufficient to extract 26 distinct Mel bands. A smaller window like 64 samples (used in some preliminary tests) only yielded 33 bins, which wasn't enough resolution for the filterbank. The 512ms window with ~80% overlap (hop=25) captures the sustained muscle activation while smoothing out transient noise, which proved optimal during signal processing experimentation."

### Q9: "What would you do differently?"

**Strong answer:**
> "Three things. First, I'd include a third electrode channel. My original design had 3 channels — mental, inner laryngeal, outer laryngeal — the top 3 from AlterEgo's ranking. The third module failed, which actually led to a productive controlled comparison (Study A vs B), but a working 3-channel system would likely exceed 60% held-out accuracy.
>
> Second, I'd implement a per-session calibration protocol from day one. The biggest accuracy jump came from multi-session training with deliberate electrode repositioning — that's the single most effective intervention.
>
> Third, I'd explore oversampling: sample at 1000 Hz and downsample to 250 Hz by averaging 4 samples, gaining ~2 bits of effective ADC resolution (12 → 14 bit equivalent). This could push beyond the onset burst into partial articulatory trajectory detection."

### Q10: "What's the real-world deployment potential?"

**Strong answer:**
> "At 64% accuracy (with confidence gating), the system is usable for low-stakes command interfaces where repetition is acceptable — think accessibility devices for people who can't speak, or private commands in public spaces. The confidence gating creates an honest 'I don't know' behavior that prevents catastrophic miscommands. The $40 price point makes it accessible in resource-constrained settings. I've built a real-time demo that runs on an M2 MacBook Air with <1ms inference latency and a terminal-based dashboard. The path to a practical device is: better ADC (16-bit, ~$15 more), 3 channels, and a dedicated neckband form factor."

---

## 4. Your Narrative Arc

**Opening:** Start with the $40 price — it's the hook. "Can you build a silent speech interface for $40?"

**Middle:**
1. The honest answer is "partially." Show the 99.7% → 48.9% story. "Here's what happens when you test honestly."
2. The onset burst discovery — the 80ms finding is your most novel scientific contribution.
3. The electrode placement finding — incompatible feature spaces is the strongest result from Study A.
4. Confidence gating as the path to deployment — 64.1% is your best realistic operating point.

**Closing:** Frame it as "a quantitative map of consumer sEMG's boundaries" rather than "a working product." The contribution is knowledge, not a device.

---

## 5. Figures to Have Ready

1. **The 99.7% vs 48.9% comparison** — dramatic visual showing training vs held-out gap
2. **Confidence gating curve** — the accuracy-coverage tradeoff (Figure 5 in P1)
3. **Cross-study transfer table** — 25–31% proves incompatible feature spaces
4. **Electrode placement photo** — shows the physical setup
5. **Live demo** — run `live_demo.py --no-serial` to show the terminal dashboard in real-time

---

## 6. Potential Gotcha Questions

| Question | Trap | Good Response |
|----------|------|---------------|
| "Why not use deep learning state-of-art like Transformers?" | Assuming bigger = better | "I tested Transformers. They got 36.4% — WORSE than CNN. With only 15 non-padded time steps and 1,500 samples, self-attention can't learn meaningful relationships. CNN's inductive bias (local temporal patterns matter) matches the physics of onset detection." |
| "How does this compare to commercial BCI products?" | Comparing apples to oranges | "Commercial BCIs (Emotiv, Neurable) use 14+ channels, 14-bit+ ADCs, and cost $300–$800. They also require per-session calibration. My system operates at 1/7th the cost of the cheapest commercial option." |
| "What about privacy/ethics?" | Expecting a thoughtful answer | "All data was auto-experimental (self-collected). No external subjects, so IRB exemption applies. For deployment, silent speech devices inherently raise privacy concerns — they could detect thoughts. I document this in the ethics section." |
| "The Jou 2006 citation — is subvocal EMG really NASA tech?" | Checking citation accuracy | "Yes — NASA Ames Research Center. Chuck Jou and colleagues demonstrated subvocal speech recognition at ~92% accuracy, published at Interspeech 2006 and ICASSP 2007. They used a different approach (multi-stream decoding) but established the fundamental viability." |

---

## 7. One-Page Cheat Sheet (Print This)

```
HARDWARE:  $40 | 2ch AD8232 + ESP32 | 12-bit, 250 Hz, ~1000× gain
SIGNAL:    1.0–50 Hz BPF | 60 Hz notch | MFCC (n_fft=128, hop=25, 13 coeff, 26 mels)
MODEL:     1D CNN | 47K params | Conv1d(26→64→128) + AdaptiveAvgPool + FC(128→6)
VOCAB:     UP, DOWN, LEFT, RIGHT, SILENCE, NOISE (6 classes)

KEY RESULTS (all 5-fold blocked CV):
  Single-session:   48.9% ± 3.1%  (2.9× chance)
  Multi-session:    58.2% ± 3.1%  (3 electrode positions)
  Conf. gating:     64.1% @ θ=0.60  (62% coverage)
  Training:         99.7%  (memorization, NOT honest)

STUDY A (chin+under-chin): 51.8% ± 2.8%
CROSS-TRANSFER: 25–31%  → incompatible feature spaces
ELECTRODE SHIFT: 22.8%  → 1cm destroys accuracy

CONVERGENT EVIDENCE FOR ONSET-ONLY:
  1. Onset mask → 17.6% (chance)
  2. LSTM → 16.6% (chance)
  3. Feature importance → 100% in first 80ms

ARCHITECTURE COMPARISON:
  CNN: 49.3% | Transformer: 36.4% | LSTM: 16.6%
```

---

*Prepared: March 1, 2026*
