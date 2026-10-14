"""Load the prebuilt Tier 1 dataset. Students use this; nobody downloads a genome.

    from tier1_loader import load_tier1
    tr, va, te = load_tier1("tier1_CTCF_K562.npz")
    X, y = tr                      # X int8 (n, 101) with A,C,G,T = 0,1,2,3
"""
import numpy as np


def onehot(seqs):
    """(n, L) int8 -> (n, 4, L) float32, channel order ACGT."""
    n, L = seqs.shape
    X = np.zeros((n, 4, L), dtype=np.float32)
    X[np.arange(n)[:, None], seqs.astype(int), np.arange(L)[None, :]] = 1.0
    return X


def load_tier1(path, negatives="dinuc"):
    """Return (train, val, test), each a tuple (seqs int8, labels float32).

    negatives: "dinuc"   dinucleotide-shuffled negatives (the default, and the
                         one the handout requires)
               "gcmatch" GC-matched genomic negatives, if the file has them
               "both"    use both negative sets
    Splits are by chromosome and were fixed when the file was built, so they
    are identical for every group. Do not re-split.
    """
    d = np.load(path, allow_pickle=False)
    seqs, y, split, kind = d["seqs"], d["y"], d["split"], d["neg_kind"]
    want = {"dinuc": {"pos", "dinuc"},
            "gcmatch": {"pos", "gcmatch"},
            "both": {"pos", "dinuc", "gcmatch"}}[negatives]
    keep = np.isin(kind, list(want))
    if not keep.any():
        raise ValueError(f"no sequences of kind {want} in {path}")
    out = []
    for s in ("train", "val", "test"):
        m = keep & (split == s)
        out.append((seqs[m], y[m]))
    return tuple(out)


if __name__ == "__main__":
    import sys
    tr, va, te = load_tier1(sys.argv[1])
    for name, (X, y) in zip(("train", "val", "test"), (tr, va, te)):
        print(f"{name:6s} n={len(y):6d}  positives={int(y.sum()):6d}  "
              f"length={X.shape[1]}")
