# COMP 565 Fall 2026 Course Project: RegExplain

Everything in this folder supports the project described in
`COMP565_Fall2026_Course_Project_RegExplain.pdf`. **Read the PDF first.**
Deadline: 15 December 2026.

## What is here

| File | What it is |
|---|---|
| `COMP565_Fall2026_Course_Project_RegExplain.pdf` | The project description. Everything you need to do is in here. |
| `tier1_CTCF_K562.npz` | The Tier 1 dataset. Real CTCF ChIP-seq peaks in K562, already processed. |
| `tier1_CTCF_K562.meme` | The CTCF motif (JASPAR MA0139.2) in MEME format, for FIMO. |
| `tier1_CTCF_K562.json` | Provenance: ENCODE accession, source URLs, counts, GC statistics. |
| `tier1_loader.py` | Loads the dataset. Use this rather than opening the `.npz` yourself. |
| `build_tier1.py` | The script that produced the dataset. For reference. **You do not need to run it.** |
| `requirements.txt` | Pinned package versions. Use these. |

You write the Tier 0 simulation yourself; Section 3 of the PDF gives you the
motif matrix to use and tells you how to check it.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The versions in `requirements.txt` are the ones the project description was
written and tested against. `dice-ml` in particular behaves differently across
releases, and Sections 4 and 6 of the PDF describe the pinned version. If you
upgrade it, you are on your own.

## Loading the Tier 1 data

```python
from tier1_loader import load_tier1, onehot

train, val, test = load_tier1("tier1_CTCF_K562.npz")
X, y = train                 # X is (n, 101) int8 with A,C,G,T = 0,1,2,3
Xoh = onehot(X)              # (n, 4, 101) float32, ready for a Conv1d
```

Running `python3 tier1_loader.py tier1_CTCF_K562.npz` prints the split sizes.

## What is in the dataset

40,000 sequences of 101 bp: 20,000 positives and 20,000 negatives.

- **Positives** are windows centred on the summits of the top 20,000 IDR
  thresholded CTCF peaks from ENCODE experiment file `ENCFF042WXA`, GRCh38.
- **Negatives** are dinucleotide-preserving shuffles of those same positives.
  This keeps GC and CpG content identical while destroying motif syntax. The
  mean GC of the positives and of the negatives agree to six decimal places
  (0.574395 both), which you can verify from the `.json`. Section 3 of the PDF
  explains why that matters more than it looks like it should.
- **Splits are fixed and held out by chromosome**: chr1 and chr2 are test,
  chr8 and chr9 are validation, the rest is train. 29,388 train, 3,174
  validation, 7,438 test. **Do not re-split.** A random split leaks through
  repeats and paralogous regions and will inflate your AUROC, and keeping the
  given split makes every group's numbers comparable.

A reasonable CNN reaches test AUROC around 0.99 on this dataset. CTCF has an
unusually long and information-rich motif, so a high number here is expected
and is not evidence that something is wrong. Your Tier 0 numbers will be lower.

## Before you ask a question

Section 6 of the PDF documents three failure modes in DiCE that look like your
bug but are not. If a counterfactual method silently returns nothing, read that
section before changing any hyperparameter.
