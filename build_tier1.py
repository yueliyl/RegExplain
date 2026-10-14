#!/usr/bin/env python3
"""
build_tier1.py - build the COMP 565 RegExplain Tier 1 dataset once, so that
students do not each have to navigate ENCODE and download a 1 GB genome.

Run this ONCE (instructor or TA), on a machine with normal internet access.
Ship the resulting .npz and .meme files with the handout.

Typical use
-----------
    python3 build_tier1.py --tf CTCF --biosample K562 --out tier1_CTCF_K562

Everything it needs is downloaded into --cache (default ./tier1_cache) and
reused on later runs. The human genome download is about 1 GB and is the
slow step; the script never decompresses it to disk, it streams it.

Offline / manual mode
---------------------
If you already have the files, point at them directly:

    python3 build_tier1.py --peaks ENCFF796WRU.bed.gz \\
        --genome hg38.fa.gz --motif MA0139.2.meme --out tier1_CTCF_K562

Outputs
-------
    <out>.npz    seqs int8 (n, L) with A,C,G,T = 0,1,2,3
                 y     float32 (n,)   1 = peak-derived positive
                 chrom <U6 (n,)       source chromosome
                 split <U5 (n,)       "train" / "val" / "test"
                 neg_kind <U8 (n,)    "pos", "dinuc", or "gcmatch"
    <out>.meme   the TF's JASPAR matrix, for FIMO
    <out>.json   provenance: accessions, URLs, counts, parameters

Dependencies: numpy only (plus the standard library).
"""
import argparse, gzip, io, json, os, random, sys, urllib.request, urllib.parse
import numpy as np

UA = {"User-Agent": "COMP565-RegExplain/1.0 (course dataset builder)"}
GENOME_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
ENCODE_SEARCH = "https://www.encodeproject.org/search/"
JASPAR_MEME = "https://jaspar.elixir.no/api/v1/matrix/{mid}.meme"
STANDARD = {f"chr{c}" for c in list(range(1, 23)) + ["X"]}
VAL_CHROMS, TEST_CHROMS = {"chr8", "chr9"}, {"chr1", "chr2"}
IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


# ----------------------------------------------------------------- utilities
def log(*a):
    print("[tier1]", *a, file=sys.stderr, flush=True)


def fetch(url, dest, cache):
    """Download url to cache/dest unless already present. Returns the path."""
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, dest)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        log(f"using cached {path}")
        return path
    log(f"downloading {url}")
    tmp = path + ".part"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                print(f"\r[tier1]   {done >> 20} / {total >> 20} MiB ({pct:.0f}%)",
                      end="", file=sys.stderr, flush=True)
    if total:
        print("", file=sys.stderr)
    os.replace(tmp, path)
    return path


def encode_find_peaks(tf, biosample, cache):
    """Query the ENCODE portal for an IDR thresholded peak file. Returns (url, accession)."""
    q = {
        "type": "File", "file_format": "bed", "file_type": "bed narrowPeak",
        "output_type": "IDR thresholded peaks", "assembly": "GRCh38",
        "status": "released", "assay_title": "TF ChIP-seq",
        "target.label": tf, "biosample_ontology.term_name": biosample,
        "limit": "25", "format": "json",
    }
    url = ENCODE_SEARCH + "?" + urllib.parse.urlencode(q)
    log(f"querying ENCODE for {tf} in {biosample}")
    req = urllib.request.Request(url, headers={**UA, "Accept": "application/json"})
    with urllib.request.urlopen(req) as r:
        data = json.load(r)
    rows = data.get("@graph", [])
    if not rows:
        raise SystemExit(
            f"No IDR thresholded peak file found for {tf} in {biosample}.\n"
            f"Browse the portal yourself and pass --peaks <url-or-path>:\n  {url}")
    # prefer 'conservative IDR thresholded peaks' when several are returned
    rows.sort(key=lambda f: (0 if "conservative" in (f.get("output_type") or "") else 1,
                             f.get("accession", "")))
    best = rows[0]
    log(f"found {len(rows)} candidate file(s); using {best['accession']} "
        f"({best.get('output_type')})")
    for f in rows[:10]:
        log(f"    {f['accession']}  {f.get('output_type')}  {f.get('dataset')}")
    return "https://www.encodeproject.org" + best["href"], best["accession"]


def opener(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


# ------------------------------------------------------------------- parsing
def read_narrowpeak(path, top_n, half):
    """Return {chrom: [(center, signal), ...]} for the top_n peaks by signal."""
    peaks = []
    with opener(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 10:
                continue
            chrom, start, end = f[0], int(f[1]), int(f[2])
            if chrom not in STANDARD:
                continue
            try:
                signal, summit = float(f[6]), int(f[9])
            except ValueError:
                continue
            center = start + summit if summit >= 0 else (start + end) // 2
            if center - half < 0:
                continue
            peaks.append((chrom, center, signal))
    if not peaks:
        raise SystemExit("No usable peaks parsed. Is this a narrowPeak file?")
    peaks.sort(key=lambda p: -p[2])
    peaks = peaks[:top_n]
    log(f"kept {len(peaks)} peaks on standard chromosomes")
    by = {}
    for chrom, center, signal in peaks:
        by.setdefault(chrom, []).append(center)
    for c in by:
        by[c].sort()
    return by


def stream_extract(genome_path, wanted, half, extra=None):
    """Stream a (gzipped) FASTA once, pulling 2*half+1 bp windows.

    wanted: {chrom: [center, ...]}   extra: {chrom: [center, ...]} or None
    Returns (seqs, extras) as dicts {chrom: [str, ...]} aligned with the inputs.
    Holds at most one chromosome in memory.
    """
    out = {c: [] for c in wanted}
    out_extra = {c: [] for c in (extra or {})}
    need = set(wanted) | set(extra or {})
    cur, buf = None, []

    def flush():
        if cur is None or cur not in need:
            return
        seq = "".join(buf).upper()
        for centers, sink in ((wanted.get(cur), out.get(cur)),
                              ((extra or {}).get(cur), out_extra.get(cur))):
            if centers is None:
                continue
            for c in centers:
                sink.append(seq[c - half: c + half + 1])
        log(f"  {cur}: {len(seq):,} bp scanned")

    with opener(genome_path) as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                cur = line[1:].split()[0]
                buf = []
            elif cur in need:
                buf.append(line.strip())
    flush()
    return out, out_extra


# ------------------------------------------------------- dinucleotide shuffle
def dinuc_shuffle(s, rng):
    """Altschul-Erikson dinucleotide-preserving shuffle of a string."""
    n = len(s)
    if n < 3:
        return s
    last = s[-1]
    edges = {}
    for a, b in zip(s[:-1], s[1:]):
        edges.setdefault(a, []).append(b)
    verts = list(edges)
    for _ in range(100):
        # 1. pick a last edge per vertex so that the last-edge graph is a tree to `last`
        lastedge = {}
        ok = True
        for v in verts:
            if v == last:
                continue
            lastedge[v] = rng.choice(edges[v])
        for v in verts:                       # connectivity check
            seen, cur = set(), v
            while cur != last:
                if cur in seen or cur not in lastedge:
                    ok = False
                    break
                seen.add(cur)
                cur = lastedge[cur]
            if not ok:
                break
        if not ok:
            continue
        # 2. shuffle the remaining edges, append the chosen last edge
        pool = {}
        for v in verts:
            rest = list(edges[v])
            if v in lastedge:
                rest.remove(lastedge[v])
            rng.shuffle(rest)
            pool[v] = rest + ([lastedge[v]] if v in lastedge else [])
        # 3. Eulerian walk
        outseq, cur = [s[0]], s[0]
        for _ in range(n - 1):
            if not pool.get(cur):
                outseq = None
                break
            nxt = pool[cur].pop(0)
            outseq.append(nxt)
            cur = nxt
        if outseq and len(outseq) == n:
            return "".join(outseq)
    return s                                   # give up, keep the original


def gc(s):
    return (s.count("G") + s.count("C")) / max(len(s), 1)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", help="TF name, e.g. CTCF (used to query ENCODE)")
    ap.add_argument("--biosample", default="K562", help="cell type, default K562")
    ap.add_argument("--peaks", help="narrowPeak file or URL (skips the ENCODE query)")
    ap.add_argument("--genome", help="hg38 FASTA, .fa or .fa.gz (skips the download)")
    ap.add_argument("--motif", help="MEME file for the TF (skips the JASPAR download)")
    ap.add_argument("--jaspar-id", help="JASPAR matrix id, e.g. MA0139.2")
    ap.add_argument("--length", type=int, default=101, help="window length, default 101")
    ap.add_argument("--top-n", type=int, default=20000, help="peaks to keep, default 20000")
    ap.add_argument("--gc-matched", action="store_true",
                    help="also emit GC-matched genomic negatives (second negative set)")
    ap.add_argument("--cache", default="tier1_cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output prefix")
    a = ap.parse_args()

    if a.length % 2 == 0:
        raise SystemExit("--length must be odd so the summit sits in the middle")
    half = a.length // 2
    rng = random.Random(a.seed)
    nprng = np.random.default_rng(a.seed)
    prov = {"length": a.length, "top_n": a.top_n, "seed": a.seed,
            "val_chroms": sorted(VAL_CHROMS), "test_chroms": sorted(TEST_CHROMS)}

    # --- peaks
    if a.peaks:
        peaks_path = a.peaks
        if peaks_path.startswith("http"):
            peaks_path = fetch(peaks_path, os.path.basename(peaks_path), a.cache)
        prov["peaks"] = a.peaks
    else:
        if not a.tf:
            raise SystemExit("give either --peaks or --tf")
        url, acc = encode_find_peaks(a.tf, a.biosample, a.cache)
        peaks_path = fetch(url, f"{acc}.bed.gz", a.cache)
        prov["peaks"] = url
        prov["encode_accession"] = acc
    by_chrom = read_narrowpeak(peaks_path, a.top_n, half)

    # --- optional GC-matched background locations: random centres on the same
    #     chromosomes, far from any kept peak. Filtered for GC after extraction.
    extra = None
    if a.gc_matched:
        extra = {}
        for c, centers in by_chrom.items():
            occupied = np.array(centers)
            lo, hi = half + 1, int(occupied.max())
            if hi <= lo:
                extra[c] = []
                continue
            cand = []
            for gap in (10_000, 2_000, 500):      # relax if the chromosome is crowded
                cand, tries = [], 0
                while len(cand) < len(centers) * 8 and tries < len(centers) * 400:
                    tries += 1
                    x = int(nprng.integers(lo, hi))
                    if np.min(np.abs(occupied - x)) > gap:
                        cand.append(x)
                if len(cand) >= len(centers) * 2:
                    break
            if gap != 10_000:
                log(f"  {c}: peaks are dense, background kept {gap} bp from any peak")
            extra[c] = sorted(set(cand))

    # --- genome
    genome = a.genome or fetch(GENOME_URL, "hg38.fa.gz", a.cache)
    prov["genome"] = a.genome or GENOME_URL
    log("streaming the genome (one pass)")
    pos_by, bg_by = stream_extract(genome, by_chrom, half, extra)

    # --- assemble
    seqs, ys, chroms, kinds = [], [], [], []
    n_dropped = 0
    for c, lst in pos_by.items():
        for s in lst:
            if len(s) != a.length or "N" in s or set(s) - set("ACGT"):
                n_dropped += 1
                continue
            seqs.append(s); ys.append(1.0); chroms.append(c); kinds.append("pos")
            seqs.append(dinuc_shuffle(s, rng)); ys.append(0.0)
            chroms.append(c); kinds.append("dinuc")
    log(f"{n_dropped} positives dropped for N or short window")

    if a.gc_matched:
        want_gc = np.array([gc(s) for s, k in zip(seqs, kinds) if k == "pos"])
        pool = [(c, s) for c in bg_by for s in bg_by[c]
                if len(s) == a.length and not (set(s) - set("ACGT"))]
        rng.shuffle(pool)
        used, bins = 0, np.histogram(want_gc, bins=20, range=(0, 1))[0]
        quota = {i: int(bins[i]) for i in range(20)}
        for c, s in pool:
            b = min(int(gc(s) * 20), 19)
            if quota.get(b, 0) > 0:
                quota[b] -= 1
                seqs.append(s); ys.append(0.0); chroms.append(c); kinds.append("gcmatch")
                used += 1
        unfilled = sum(quota.values())
        log(f"{used} GC-matched genomic negatives added ({unfilled} slots unfilled)")
        if used < 0.5 * len(want_gc):
            log("WARNING: fewer than half the GC-matched slots could be filled. "
                "The genomic background does not cover the positives' GC range. "
                "Report this, or drop --gc-matched and use the dinucleotide "
                "shuffles alone.")

    arr = np.array([[IDX[ch] for ch in s] for s in seqs], dtype=np.int8)
    y = np.array(ys, dtype=np.float32)
    chrom = np.array(chroms)
    split = np.where(np.isin(chrom, list(TEST_CHROMS)), "test",
                     np.where(np.isin(chrom, list(VAL_CHROMS)), "val", "train"))
    kind = np.array(kinds)

    np.savez_compressed(a.out + ".npz", seqs=arr, y=y, chrom=chrom,
                        split=split, neg_kind=kind)
    prov["counts"] = {k: int((kind == k).sum()) for k in np.unique(kind)}
    prov["splits"] = {k: int((split == k).sum()) for k in np.unique(split)}
    prov["gc_positive_mean"] = float(np.mean([gc(s) for s, k in zip(seqs, kinds) if k == "pos"]))
    prov["gc_dinuc_mean"] = float(np.mean([gc(s) for s, k in zip(seqs, kinds) if k == "dinuc"]))

    # --- motif
    mid = a.jaspar_id
    if a.motif:
        with open(a.motif) as f:
            open(a.out + ".meme", "w").write(f.read())
        prov["motif"] = a.motif
    elif mid:
        p = fetch(JASPAR_MEME.format(mid=mid), f"{mid}.meme", a.cache)
        with open(p) as f:
            open(a.out + ".meme", "w").write(f.read())
        prov["motif"] = JASPAR_MEME.format(mid=mid)
    else:
        log("no --motif or --jaspar-id given; fetch the matrix yourself from "
            "https://jaspar.elixir.no and save it as " + a.out + ".meme")

    json.dump(prov, open(a.out + ".json", "w"), indent=2)
    log("wrote " + a.out + ".npz / .meme / .json")
    log(json.dumps(prov["counts"]) + "  " + json.dumps(prov["splits"]))
    log(f"GC: positives {prov['gc_positive_mean']:.3f}, "
        f"dinuc negatives {prov['gc_dinuc_mean']:.3f} (these must match)")


if __name__ == "__main__":
    main()
