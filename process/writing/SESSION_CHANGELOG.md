# Session Changelog — Citation Audit & NOTAI-SKILL Humanizer

## 1. Citation Audit Integration

### Paper 1 (Study B — Chin + Throat, Curriculum Learning)

| # | Location | Before | After | Reason (from audit) |
|---|----------|--------|-------|---------------------|
| 1 | Comparison table, AlterEgo row | "20 cmd" | "10 digits†" + footnote: "Kapur reports ~92 % on a 10-digit vocabulary; separate tasks used ~20 words total" | Audit found 92 % was on 10-digit subset, not 20-command set |
| 2 | Comparison table, Meltzner row | "Cross-session" | "Laryngectomy patients" | 83 % figure was on laryngectomy cohort, not healthy cross-session |
| 3 | Related Work, Jou paragraph | Attributed 92 % to 2006 Interspeech; described as "covert" speech | Rewrote: 2007 ICASSP paper; clarified "mouthed" (visible articulation) vs "imagined" (no movement); noted failure to decode mentally rehearsed speech | Audit found Jou 92 % is from 2007, used mouthed speech, not silent/imagined |

### Paper 2 (Study A — Chin + Under-Chin, Electrode Comparison)

| # | Location | Before | After | Reason (from audit) |
|---|----------|--------|-------|---------------------|
| 1 | After electrode config table | (no table) | Added **Table: innervation** — 5 muscles × {nerve, depth, volume-conduction} | Audit provided biophysical innervation data explaining shared CN V3 pathway |
| 2 | Related Work | No Jou entry | Added Jou et al. (2006; 2007) with mouthed/imagined distinction | Aligns with audit correction |
| 3 | AD8232 bandwidth discussion | "limiting the system to an onset-detection paradigm" | Strengthened to "onset-only detection paradigm"; added 50–500 Hz MUAP frequency context | Audit highlighted bandwidth ceiling as hardware-mandated |
| 4 | Bibliography | No Jou entries | Added `jou2006` and `jou2007` bibitem entries | Required by new Related Work citations |

---

## 2. NOTAI-SKILL Humanizer — Per-Pattern Changelog

### NOTAI-SKILL Pattern Key

| Pattern ID | Category | Description |
|------------|----------|-------------|
| NS-01 | Filler adjectives | "Key", "Crucial", "Critical", "Vital", "Essential" |
| NS-02 | Em-dash overuse | "—" used as parenthetical, pivot, or appositive |
| NS-03 | Weak copula | "represents" as hedge instead of direct verb |
| NS-04 | Boldface mechanical | `\textbf{...}` used as sentence-starter pattern |
| NS-05 | Conjunctive adverbs | "Furthermore", "Consequently", "Moreover" |
| NS-06 | Participial -ing | Dangling or chained -ing clauses |
| NS-07 | Intensifiers | "drastically", "profoundly", "critically" |
| NS-08 | Filler phrases | "has implications for", "delivers actionable knowledge", "refines the understanding" |
| NS-09 | Negative parallelism | "not just X, but Y" |
| NS-10 | Word choice | "foundational", "encompassing", "nuanced", "highlights" |

---

### Paper 1 — Humanizer Fixes (14 edits)

| # | Pattern | Line area | Before → After |
|---|---------|-----------|----------------|
| 1 | NS-07 | Abstract | "a drastically cost-reduced alternative" → "a low-cost alternative" |
| 2 | NS-02 | Abstract | "demonstrating…---though" → "which demonstrates…, though" |
| 3 | NS-10 | Intro | "foundational" → "central" |
| 4 | NS-10 | Intro | "highlights" → "shows" |
| 5 | NS-03 | Methodology | "represents a domain mismatch" → "is a domain mismatch" |
| 6 | NS-08 | Discussion | "A profound insight" → "A practical realization" |
| 7 | NS-07 | Discussion | "drastically reduces" → "sharply reduces" |
| 8 | NS-06 | Results | "demonstrating effective" → "confirming effective" |
| 9 | NS-03 | Discussion | "represents a significant cognitive load" → "is a significant cognitive load" |
| 10 | NS-06 | Discussion | "reinforcing the need" → "which supports the case" |
| 11 | NS-02 | Discussion | "features---the spectral" → "features; the spectral" |
| 12 | NS-04 | Discussion | "**This is the strongest practical finding:**" → removed bold lead-in |
| 13 | NS-05 | Discussion | "Furthermore, the necessity" → "Per-session calibration also remains" |
| 14 | NS-10+08 | Conclusion | "encompassing…delivers actionable knowledge" → "including…these results inform the design" |

### Paper 2 — Humanizer Fixes (20 edits)

| # | Pattern | Line area | Before → After |
|---|---------|-----------|----------------|
| 1 | NS-10 | Abstract | "nuanced negative result" → "negative result" |
| 2 | NS-02 | §1 Motivation | "12-bit ADCs---where…---remains" → "12-bit ADCs, where…, remains" |
| 3 | NS-01 | Related Work | "Key references" → "References" |
| 4 | NS-01 | §3 Setup | "Key difference from Study B:" → "Difference from Study B:" |
| 5 | NS-02 | §4 Held-out | "and---surprisingly---marginally" → "and, surprisingly, marginally" |
| 6 | NS-01+04 | §4 Per-phase | "**Key finding:** Accuracy does not…weakness---the problem is" → "Accuracy does not…weakness; it is" |
| 7 | NS-01 | §4 Per-phase | "Key observations:" → "Observations:" |
| 8 | NS-02 | §5 Cross-study | "other---they are not" → "other. They are not" |
| 9 | NS-02 | §5 Cross-study | "for directional commands---they" → "for directional commands: they" |
| 10 | NS-02 | §5 Biophysics | "anomaly---it is" → "anomaly; it is" |
| 11 | NS-06 | Related Work | "establishing that" → "finding that" |
| 12 | NS-05 | §5 Biophysics | "Consequently, my under-chin" → "My under-chin electrodes therefore" |
| 13 | NS-03 | §5 Biophysics | "represents the CNN memorizing" → "shows the CNN memorizing" |
| 14 | NS-06 | §5 Biophysics | "demonstrating that throat sensors…introducing a statistically" → "Their results show that throat sensors…adding a statistically" |
| 15 | NS-09 | §5 Design | "not just for instantaneous accuracy, but for multi-session stability" → "for multi-session stability over instantaneous accuracy" |
| 16 | NS-05+06 | §5 Limitations | "Furthermore, Study B followed…introducing…improving…decreasing" → "Study B also followed…which may have created…offset by" |
| 17 | NS-07 | §6 Conclusion | "critically underreported" → "underreported" |
| 18 | NS-08 | §6 Conclusion | "has implications for wearable form factors:" → "constrains wearable form factors:" |
| 19 | NS-08+10 | §6 Conclusion | "refines the understanding of electrode placement" → "establishes electrode placement" |
| 20 | NS-08 | §6 Conclusion | "reinforces the case for" → "supports" |

---

## 3. Build Summary

| Paper | Pages | Size | Status |
|-------|-------|------|--------|
| Paper 1 (Study B) | 20 | 1.07 MB | ✅ Compiled clean |
| Paper 2 (Study A) | 15 | 1.02 MB | ✅ Compiled clean |

---

## 4. Deliverables This Session

| Deliverable | Path |
|-------------|------|
| Paper 1 PDF | `paper1_UPDATED/main.pdf` |
| Paper 2 PDF | `paper2_UPDATED/main.pdf` |
| This changelog | `SESSION_CHANGELOG.md` |
