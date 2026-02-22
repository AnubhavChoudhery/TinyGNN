#!/usr/bin/env python3
"""
fetch_datasets.py  --  Download Cora and Reddit datasets, convert to TinyGNN CSV.

Usage:
    python3 scripts/fetch_datasets.py              # fetch both
    python3 scripts/fetch_datasets.py --cora-only   # Cora only  (no extra deps)
    python3 scripts/fetch_datasets.py --reddit-only  # Reddit only (needs numpy + scipy)

Output:
    datasets/cora/edges.csv            src,dst  per line
    datasets/cora/node_features.csv    node_id,f0,...,f1432  per line
    datasets/reddit/edges.csv          src,dst  per line
    datasets/reddit/node_features.csv  node_id,f0,...,f601  per line

Requirements:
    Python 3.6+
    numpy       (pip install numpy)       -- Reddit only
    scipy       (pip install scipy)       -- Reddit only

Sources:
    Cora   : LINQS  https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
    Reddit : DGL    https://data.dgl.ai/dataset/reddit.zip
"""

import argparse
import io
import os
import sys
import tarfile
import tempfile
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATASETS   = os.path.join(ROOT_DIR, "datasets")


# ── Download helper ─────────────────────────────────────────────────────────

def _progress(label):
    """Return a urllib reporthook that prints download percentage."""
    last = [-1]
    def hook(count, block, total):
        if total > 0:
            pct = min(100, count * block * 100 // total)
        else:
            pct = -1
        if pct != last[0]:
            last[0] = pct
            if pct >= 0:
                print(f"\r  [{label}] Downloading: {pct}%", end="", flush=True)
            else:
                mb = count * block / 1048576
                print(f"\r  [{label}] Downloaded: {mb:.1f} MB", end="", flush=True)
    return hook


# ── Cora ────────────────────────────────────────────────────────────────────

CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"

def fetch_cora():
    """Download the Cora citation dataset and convert to TinyGNN CSV."""
    cora_dir     = os.path.join(DATASETS, "cora")
    edges_csv    = os.path.join(cora_dir, "edges.csv")
    features_csv = os.path.join(cora_dir, "node_features.csv")

    if os.path.isfile(edges_csv) and os.path.isfile(features_csv):
        print("[Cora] Already present in datasets/cora/ -- skipping.\n")
        return

    os.makedirs(cora_dir, exist_ok=True)
    print(f"[Cora] Downloading {CORA_URL}")

    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "cora.tgz")
        urllib.request.urlretrieve(CORA_URL, archive, _progress("Cora"))
        print()

        print("[Cora] Extracting archive...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp)

        content_path = os.path.join(tmp, "cora", "cora.content")
        cites_path   = os.path.join(tmp, "cora", "cora.cites")
        if not os.path.isfile(content_path) or not os.path.isfile(cites_path):
            raise FileNotFoundError(
                f"Expected cora.content and cora.cites inside archive")

        # ── Parse cora.content ──────────────────────────────────────
        # Format: <paper_id>\t<word_0>\t...\t<word_1432>\t<class_label>
        print("[Cora] Parsing cora.content...")
        paper_feats = {}
        with open(content_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                pid   = int(parts[0])
                feats = [int(x) for x in parts[1:-1]]   # drop class label
                paper_feats[pid] = feats

        sorted_pids = sorted(paper_feats)
        pid2idx     = {p: i for i, p in enumerate(sorted_pids)}
        num_nodes   = len(sorted_pids)
        num_feats   = len(next(iter(paper_feats.values())))

        # ── Parse cora.cites ────────────────────────────────────────
        # Format: <cited_paper_id>\t<citing_paper_id>
        print("[Cora] Parsing cora.cites...")
        edges   = []
        skipped = 0
        with open(cites_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                cited, citing = int(parts[0]), int(parts[1])
                if cited in pid2idx and citing in pid2idx:
                    # Direction: citing -> cited  (paper X cites paper Y)
                    edges.append((pid2idx[citing], pid2idx[cited]))
                else:
                    skipped += 1

        if skipped:
            print(f"  (skipped {skipped} edges referencing unknown paper IDs)")

        # ── Write edges.csv ─────────────────────────────────────────
        num_edges = len(edges)
        print(f"[Cora] Writing edges.csv  ({num_edges:,} edges)")
        with open(edges_csv, "w") as f:
            f.write("src,dst\n")
            for s, d in edges:
                f.write(f"{s},{d}\n")

        # ── Write node_features.csv ─────────────────────────────────
        print(f"[Cora] Writing node_features.csv  ({num_nodes} x {num_feats})")
        with open(features_csv, "w") as f:
            hdr = "node_id," + ",".join(f"f{j}" for j in range(num_feats))
            f.write(hdr + "\n")
            for idx, pid in enumerate(sorted_pids):
                feat = paper_feats[pid]
                f.write(str(idx))
                for v in feat:
                    f.write(f",{v}")
                f.write("\n")

    print(f"[Cora] Done: {num_nodes} nodes, {num_edges:,} edges, "
          f"{num_feats} features")
    print(f"  -> {edges_csv}")
    print(f"  -> {features_csv}\n")


# ── Reddit ──────────────────────────────────────────────────────────────────

REDDIT_URL = "https://data.dgl.ai/dataset/reddit.zip"

def fetch_reddit():
    """Download the Reddit GraphSAGE dataset and convert to TinyGNN CSV."""
    reddit_dir   = os.path.join(DATASETS, "reddit")
    edges_csv    = os.path.join(reddit_dir, "edges.csv")
    features_csv = os.path.join(reddit_dir, "node_features.csv")

    if os.path.isfile(edges_csv) and os.path.isfile(features_csv):
        print("[Reddit] Already present in datasets/reddit/ -- skipping.\n")
        return

    try:
        import numpy as np
        from scipy import sparse as sp
    except ImportError as exc:
        print(f"[Reddit] Missing dependency: {exc}")
        print("  Install with:  pip install numpy scipy")
        sys.exit(1)

    import zipfile
    os.makedirs(reddit_dir, exist_ok=True)

    print(f"[Reddit] Downloading {REDDIT_URL}  (~200 MB)")

    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "reddit.zip")
        urllib.request.urlretrieve(REDDIT_URL, archive, _progress("Reddit"))
        print()

        print("[Reddit] Extracting archive...")
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)

        # Locate .npz files (may be nested)
        data_npz = graph_npz = None
        for root, _, files in os.walk(tmp):
            for fname in files:
                if fname == "reddit_data.npz":
                    data_npz = os.path.join(root, fname)
                elif fname == "reddit_graph.npz":
                    graph_npz = os.path.join(root, fname)

        if not data_npz or not graph_npz:
            print("[Reddit] ERROR: reddit_data.npz or reddit_graph.npz not found")
            sys.exit(1)

        # ── Load features ───────────────────────────────────────────
        print("[Reddit] Loading features from reddit_data.npz...")
        blob     = np.load(data_npz)
        features = blob["feature"]                       # (N, 602) float32
        num_nodes, num_feats = features.shape
        print(f"  Features shape: ({num_nodes}, {num_feats})")

        # ── Load adjacency ──────────────────────────────────────────
        print("[Reddit] Loading adjacency from reddit_graph.npz...")
        adj       = sp.load_npz(graph_npz)               # scipy CSR
        num_edges = adj.nnz
        print(f"  Adjacency: {adj.shape[0]}x{adj.shape[1]}, nnz={num_edges:,}")

        # ── Write edges.csv (chunked for 100M+ edges) ──────────────
        print(f"[Reddit] Writing edges.csv  ({num_edges:,} edges) ...")
        coo   = adj.tocoo()
        CHUNK = 2_000_000
        with open(edges_csv, "w", buffering=8 * 1024 * 1024) as f:
            f.write("src,dst\n")
            written = 0
            for start in range(0, num_edges, CHUNK):
                end = min(start + CHUNK, num_edges)
                buf = io.BytesIO()
                np.savetxt(
                    buf,
                    np.column_stack([coo.row[start:end], coo.col[start:end]]),
                    fmt="%d", delimiter=",",
                )
                f.write(buf.getvalue().decode("ascii"))
                written = end
                if end % 10_000_000 < CHUNK or end == num_edges:
                    print(f"    {written:,} / {num_edges:,} edges", flush=True)

        # ── Write node_features.csv ─────────────────────────────────
        print(f"[Reddit] Writing node_features.csv  ({num_nodes} x {num_feats}) ...")
        hdr  = "node_id," + ",".join(f"f{j}" for j in range(num_feats))
        ids  = np.arange(num_nodes, dtype=np.float64).reshape(-1, 1)
        full = np.column_stack([ids, features.astype(np.float64)])
        fmt  = ["%d"] + ["%.6g"] * num_feats
        np.savetxt(features_csv, full, delimiter=",",
                   header=hdr, comments="", fmt=fmt)

    print(f"[Reddit] Done: {num_nodes} nodes, {num_edges:,} edges, "
          f"{num_feats} features")
    print(f"  -> {edges_csv}")
    print(f"  -> {features_csv}\n")


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Fetch graph datasets for TinyGNN")
    ap.add_argument("--cora-only",   action="store_true",
                    help="Download only the Cora dataset")
    ap.add_argument("--reddit-only", action="store_true",
                    help="Download only the Reddit dataset (needs numpy+scipy)")
    args = ap.parse_args()

    print(f"Output directory: {DATASETS}\n")

    if not args.reddit_only:
        fetch_cora()
    if not args.cora_only:
        fetch_reddit()

    print("All done.")

if __name__ == "__main__":
    main()
