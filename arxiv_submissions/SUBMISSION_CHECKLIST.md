# arXiv Submission Checklist
**SOMACH Companion Papers — cs.HC**
_Last updated: 2026-03-21_

---

## PRE-UPLOAD: Citation Verification (Do These First)

These items are flagged in the `.tex` files with `% CITATION CHECK REQUIRED` comments. You MUST manually verify or remove each before uploading.

### Paper 1 (`paper1_studyB_curriculum.tar.gz`)

- [x] **`hristov2026`** — **VERIFIED.** arXiv:2602.01855 exists (Hristov, Gjoreski, Ojleska Latkoska, Nadzinski). Authors corrected from fabricated names. Title confirmed: "Time2Vec Transformer for Robust Gesture Recognition from Low-Density sEMG."

- [x] **`biosense2016`** — **VERIFIED.** Correct authors: M.T. Curran, J.-K. Yang, N. Merrill, J. Chuang. Pages corrected 1834→1979-1982. DOI corrected to 10.1109/EMBC.2016.7591112.

- [x] **`palmgraffiti`** — Acceptable as product reference. No change needed.

### Paper 2 (`paper2_studyA_electrode.tar.gz`)

- [x] **`saejong2018`** — **FIXED.** Original entry was fabricated (no paper in J. Oral Rehab vol.45 no.8 by "Y. Sae Jong"). Replaced with real paper: N. Sae Jong, P. Phukpattaranont, C. Limsakul, "Channel reduction in speech recognition system based on surface electromyography," IEEE ECTI-CON 2018. DOI: 10.1109/ECTICon.2018.8619947. Related Work description updated to match.

- [x] **`ala2018`** — Kept as-is (gray literature, acceptable for clinical context). Comment in .tex notes Blitzer 2015 as peer-reviewed alternative.

- [x] **`fessenden2019`** — **FIXED.** Actual authors: S. Nimpf and D.A. Keays (not T. Fessenden). Title: "Why (and how) we should publish negative data." Online Dec 2019, print Jan 2020 (vol.21 no.1). DOI: 10.15252/embr.201949775 confirmed. Year updated to 2020. In-text attribution changed to "Nimpf and Keays (2020)."

---

## POST-PAPER-1 SUBMISSION: Update Companion Citations

After Paper 1 is accepted and you receive its arXiv ID (format: `XXXX.XXXXX`):

1. **Update Paper 2** — In `paper2_UPDATED/main.tex`, find `\bibitem{kho2026companion}` and replace `XXXX.XXXXX` with the actual arXiv ID.
2. **Rebuild Paper 2 archive:**
   ```bash
   cd /Users/cvk/Downloads/carl/phase1-5/CP-PHASE4-2-sEMG_arXiv_Papers_25TPE
   tar -czf arxiv_submissions/paper2_studyA_electrode.tar.gz \
     -C paper2_UPDATED main.tex arxiv.sty figures
   ```
3. **Submit Paper 2** to arXiv.
4. Get Paper 2 arXiv ID → **update Paper 1** via arXiv "Replace" (v2): find `\bibitem{kho2026companion}` in `paper1_UPDATED/main.tex` and fill in Paper 2's arXiv ID.

---

## SUBMISSION: Paper 1 Upload

**File to upload:** `arxiv_submissions/paper1_studyB_curriculum.tar.gz` (968K)

| Field | Value |
|-------|-------|
| **Title** | Curriculum Learning for Silent Speech Classification: A Proof-of-Concept $40 Two-Channel sEMG System |
| **Authors** | Carl Vincent Kho |
| **Abstract** | Use abstract from `main.tex` — first `\begin{abstract}` block |
| **Primary category** | `cs.HC` (Human-Computer Interaction) |
| **Cross-list** | `eess.SP` (Signal Processing) — optional but recommended |
| **License** | CC BY 4.0 |
| **Comments field** | `20 pages, 5 figures. Companion to arXiv:XXXX.XXXXX (Study A). Code: [your repo if public]` |
| **Journal-ref** | Leave blank |
| **DOI** | Leave blank |
| **Endorsement** | Already endorsed in cs.HC via arXiv:2601.06516 ✅ |

---

## SUBMISSION: Paper 2 Upload

**File to upload:** `arxiv_submissions/paper2_studyA_electrode.tar.gz` (712K) — only after Paper 1 ID is filled in.

| Field | Value |
|-------|-------|
| **Title** | Lingual-Mandibular Electrode Configuration for Silent Speech: Throat Sensor Necessity on Consumer-Grade ADCs |
| **Authors** | Carl Vincent Kho |
| **Abstract** | Use abstract from `main.tex` |
| **Primary category** | `cs.HC` |
| **Cross-list** | `eess.SP` — optional |
| **License** | CC BY 4.0 |
| **Comments field** | `15 pages, 4 figures. Companion to arXiv:[PAPER1_ID] (Study B). Negative result paper.` |
| **Journal-ref** | Leave blank |

---

## FILE LOCATIONS

```
Canonical sources (submit from here):
  paper1_UPDATED/main.tex
  paper2_UPDATED/main.tex

Archives to upload:
  arxiv_submissions/paper1_studyB_curriculum.tar.gz   ← 905K ✅ rebuilt 2026-03-21 (citations fixed)
  arxiv_submissions/paper2_studyA_electrode.tar.gz    ← 697K ✅ rebuilt 2026-03-21 (citations fixed)

Mirror copies (reference only):
  md-capstonefall25_25TPE/.../paper1_UPDATED/main.tex  ✅ synced
  md-capstonefall25_25TPE/.../paper2_UPDATED/main.tex  ⚠️  NOT synced (paper2 fixes not applied yet)
```

---

## KNOWN ISSUES RESOLVED

- ✅ Broken `Figure~\ref{fig:lcurve_note}` → was compiling to `(??)` — fixed to plain text
- ✅ Abstract `±1.0%` → `±1.0 pp` (percentage points — precision for reviewers)
- ✅ Companion citation placeholder `XXXX.XXXXX` annotated with TODO in both papers
- ✅ All gray-literature / high-risk citations flagged with comments

---

## TIMING

**Submit after:** March 20, 2026
**Paper 1 first**, then Paper 2 (within same week if possible so they appear together).

arXiv processes Sunday–Thursday; submissions before 14:00 ET appear next business day.
