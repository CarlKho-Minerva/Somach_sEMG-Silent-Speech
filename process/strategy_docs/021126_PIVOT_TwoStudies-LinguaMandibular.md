# PIVOT: Two Studies — Lingual-Mandibular + Subvocal

**Date:** 2026-02-10
**Trigger:** AD8232 #3 confirmed broken (both spare units dead)
**Decision:** Proceed with 2-channel system, split into two parallel studies

---

## Why Two Sensors Is Enough

Per Kapur et al. (AlterEgo, 2018):
- **Chin (Mentalis)** ranked #1 for articulation detection
- **Laryngeal (Throat)** ranked #2–#3 for voicing/phonation
- Left and Right throat signals are **~90% collinear** for digits — one side is sufficient

The third channel provided redundant feature data for a 10-word vocabulary. Dropping to 2 channels is a **dimensionality reduction**, not a degradation.

---

## Shared Vocabulary: UP, DOWN, LEFT, RIGHT

Both studies use the **same four commands** (+ SILENCE, NOISE). The AI label stays the same — what changes is the **sensor placement** and the **physical action strategy**.

| COMMAND | PHONETIC TRIGGER | THE ACTION | WHAT YOU FEEL |
|:--------|:-----------------|:-----------|:-------------|
| **UP** | "Nnnn" / "Ceiling" | Tongue flat → hard palate (roof of mouth) | Strong upward jaw-floor pressure |
| **DOWN** | "Gah" / "Floor" | Tongue back/down, gag/yawn motion | Deep tension at tongue root |
| **LEFT** | "Llll" / "Cheek" | Tongue → left cheek/molars | One-sided jaw pressure (left) |
| **RIGHT** | "Errr" / "Curl" | Tongue tip curled back toward soft palate | Tight tip tension, hollow underneath |

---

## Study A: Lingual-Mandibular Interface (Transfer Learning)

**Goal:** Train on open-mouth articulation, then transfer to closed-mouth — proving the signal transfers.

| Sensor | Placement | Muscle Target |
|--------|-----------|---------------|
| CH1 (GPIO 34) | Chin center | Mentalis |
| CH2 (GPIO 36) | Under-chin (jaw floor) | Mylohyoid / Digastric |
| Reference | Left earlobe | — |

**Transfer Learning Phases:**
1. **Phase 1 — Overt (Open Mouth):** Say the commands aloud while doing tongue gestures. Full jaw + tongue movement. Strongest signal.
2. **Phase 3 — Mouthing (Open Mouth, Silent):** Lip sync the commands, tongue active, no voice. Jaw still moves.
3. **Phase 5 — Exaggerated (Closed Mouth):** Mouth shut. Exaggerate the tongue press hard. No jaw/lip movement visible.
4. **Phase 6 — Covert (Closed Mouth, Minimal):** Mouth shut. Think the tongue gesture with minimal muscular effort.

**The story for your professor:** "Model trained on open-mouth data (Phase 1+3), fine-tuned/evaluated on closed-mouth data (Phase 5+6). Accuracy degrades from X% → Y%, proving transfer learning viability."

---

## Study B: Subvocal Interface (Curriculum Learning)

**Goal:** Classify the same UP/DOWN/LEFT/RIGHT via laryngeal (throat) activity, using curriculum learning to progressively reduce signal intensity.

| Sensor | Placement | Muscle Target |
|--------|-----------|---------------|
| CH1 (GPIO 34) | Chin center | Mentalis |
| CH2 (GPIO 36) | Left throat (1.5cm from Adam's apple) | Thyrohyoid / Sternohyoid |
| Reference | Left earlobe | — |

**Curriculum Phases:**
1. **Phase 1 — Overt:** Speak UP/DOWN/LEFT/RIGHT aloud — establishes signal shape baseline
2. **Phase 2 — Whispered:** Whisper the commands — reduces amplitude, tests placement
3. **Phase 3 — Mouthing:** Lip sync only — the practical demo target
4. **Phase 6 — Covert:** Pure internal, mouth closed — the scientific stretch goal

**The story for your professor:** "Same vocabulary, same model architecture. When Sensor 2 moves from under-chin to throat, the signal source shifts from articulatory to laryngeal. Curriculum learning shows how accuracy degrades as phonation becomes internal."

**Electrode Placement:**
1. Find your Adam's apple (Thyroid Cartilage).
2. Slide 1.5 cm to the **LEFT** of the peak.
3. **Yellow (LO+):** On the flat "wing" of the thyroid lamina.
4. **Green (LO-):** 1.5 cm directly below Yellow.

---

## Technical Notes

- **N-channel refactor** is already complete in both Arduino firmware and Python pipeline
- All scripts auto-detect 2 channels from the serial stream
- The same `code/python/` folder serves both studies — separation is at the session/data level
- `10-curriculum-eval.py` trains on ALL phases combined, evaluates EACH separately
- Pilot session (020926) achieved **87.9% accuracy** on 5 classes with just 20 samples/class