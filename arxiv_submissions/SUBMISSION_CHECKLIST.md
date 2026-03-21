# arXiv Submission Checklist
**SOMACH Companion Papers — cs.HC**
_Last updated: 2026-03-17_

---

## PRE-UPLOAD: Citation Verification (Do These First)

These items are flagged in the `.tex` files with `% CITATION CHECK REQUIRED` comments. You MUST manually verify or remove each before uploading.

### Paper 1 (`paper1_studyB_curriculum.tar.gz`)

- [ ] **`hristov2026`** — Search arXiv for `2602.01855`. If it doesn't exist:
  - Remove the `\cite{hristov2026}` in §1.2 Related Work (the sentence ending "...CNN-over-Transformer architectures for biosignal classification~\cite{hristov2026}").
  - Delete the entire `\bibitem{hristov2026}` block.
  - The claim is self-supported by your Table 3 results; removal doesn't weaken the paper.

- [ ] **`biosense2016`** — Verify: N. Merrill, A. Chuang, T. Chuang, "BioSense: A natural input device for biometric identification," EMBC 2016. Check page numbers. Suggested DOI in the `.tex` comment.

- [ ] **`palmgraffiti`** — Acceptable as a product reference; no action required unless you want to replace with a peer-reviewed cite.

### Paper 2 (`paper2_studyA_electrode.tar.gz`)

- [ ] **`saejong2018`** — Verify author name romanization in: Journal of Oral Rehabilitation, vol. 45, no. 8, 2018. Author listed as "K. Saejong et al." — confirm spelling and full first name in the journal.

- [ ] **`ala2018`** — This is an ALA clinical practice guideline (gray literature, not peer-reviewed). Either:
  - Keep as-is (acceptable for clinical context), OR
  - Replace with: Blitzer et al., "Laryngoscope," 2015 — a peer-reviewed alternative noted in the `.tex` comment.

- [ ] **`fessenden2019`** — Verify print publication year of EMBO Reports e49775 (electronic pub date may be 2019 but print volume may be 2020). Check the journal page and confirm year matches your `\bibitem`.

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
  arxiv_submissions/paper1_studyB_curriculum.tar.gz   ← 968K ✅ rebuilt 2026-03-17
  arxiv_submissions/paper2_studyA_electrode.tar.gz    ← 712K ✅ rebuilt 2026-03-17

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
