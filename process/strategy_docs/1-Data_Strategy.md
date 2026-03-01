# The "Very Real" Data Strategy

You asked for the hard truth. Here is the breakdown based on your current results and code.

## The Bottom Line

**Study B (Throat) is your winner.** It is highly efficient and "salvageable" with minimal effort.
**Study A (Chin) is a grind.** To make it work, you need 3-4x the effort of Study B.

## 1. Study B: The "Subvocal" Path (Recommended)
This is the one that gave you **99% accuracy**. It works because the signal is stronger and simpler.

*   **Target:** 50 samples per label (Total: 300 samples).
*   **Why:** Your results showed 99% accuracy with this amount.
*   **Time Cost:** ~1 Hour total recording time.
*   ** "100 Samples Per Day" Plan:**
    *   **Day 1:** Phase 1 (Overt) & Phase 3 (Mouthing) - 100 samples.
    *   **Day 2:** Phase 5 (Exaggerated) & Phase 6 (Covert) - 100 samples.
    *   **Day 3:** Makeup / Extra data for weak classes - 100 samples.
    *   **Result:** Done in 3 days. High success probability.

## 2. Study A: The "Lingual" Path (Hard Mode)
This one is failing (**45% accuracy**) because the signal is nuanced and you don't have enough data.

*   **Target:** 150-200 samples per label (Total: 900-1200 samples).
*   **Why:** Deep learning rules of thumb—noisy data needs more examples. 50 was not enough.
*   **Time Cost:** ~4-6 Hours total recording time.
*   **"100 Samples Per Day" Plan:**
    *   This would take you **9-12 days**.
    *   **Risk:** "Sensor Drift". Your electrode placement will change slightly every day, confusing the model.
    *   **Verdict:** Unless this is critical, **skip it** or reduce the scope (e.g., fewer phases).

## Can I do "100 per day"?
**YES.**
I checked your code (`4-curriculum-recorder.py`).
1.  It **appends** new files to the folder. It does NOT overwrite.
2.  You can record Monday, Tuesday, Wednesday.
3.  When you run the training script, it will grab **all of it**.

**Critical Tip for Multi-Day:**
Since you are spreading it out, **Mark your skin** or take a photo of your electrode placement. If you move the sensor by 1cm, clean data from Day 1 becomes noise for Day 2.

## Summary Checklist
- [ ] **Commit to Study B** as your primary deliverable.
- [ ] **Goal:** 50 samples per class per phase.
- [ ] **Daily Habit:** 20 mins (100 samples) is a perfect pace. Just keep the sensor placement consistent.
