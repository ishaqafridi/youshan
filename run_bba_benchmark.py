"""
Benchmark the corrected YouShanAligner (see youshan_aligner_finetuned.py)
against the six reference datasets in the youshan repo:
BBA0001.tfa ... BBA0006.tfa (BAliBASE-style reference protein families).

For each dataset this script:
  1. Downloads the .tfa file directly from the GitHub repo (raw content),
     so no manual file management is needed.
  2. Parses it with Biopython.
  3. Fine-tunes a FRESH copy of the pretrained encoder on that dataset's own
     sequences via masked language modeling (self-supervised, transductive
     fine-tuning -- appropriate here because each BBA file is a different
     protein family/superfamily, so a single globally fine-tuned model
     across all six would blur family-specific signal and risks
     catastrophic forgetting between unrelated families).
  4. Builds an embedding-similarity-based guide tree (progressive MSA
     ordering) with the fine-tuned model.
  5. As an ablation, repeats step 4 with the *frozen, non-fine-tuned*
     pretrained model, so you can directly report whether fine-tuning
     changes/improves the guide tree ordering and embedding separation
     -- this is the evidence a reviewer will want to see, since it
     demonstrates the fine-tuning step is doing something.
  6. Times every stage and writes a CSV summary.

Usage:
    pip install torch transformers biopython scipy scikit-learn pandas --break-system-packages
    python run_bba_benchmark.py --datasets BBA0001 BBA0002 BBA0003 BBA0004 BBA0005 BBA0006
    python run_bba_benchmark.py --epochs 3 --batch-size 4 --out results/bba_benchmark.csv

Notes:
  - BBA0004 (248 seqs) and BBA0003 (126 seqs) will take noticeably longer
    to fine-tune than BBA0001 (23 seqs) or BBA0005 (44 seqs). Use --limit
    to subsample large families for a quick smoke test before a full run.
  - Requires a GPU for reasonable runtime on the larger files; will fall
    back to CPU automatically but expect it to be slow for BBA0003/BBA0004.
"""

from __future__ import annotations

import argparse
import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform

from youshan_aligner_finetuned import YouShanAligner

RAW_BASE = "https://raw.githubusercontent.com/ishaqafridi/youshan/main"


@dataclass
class DatasetResult:
    dataset: str
    n_sequences: int
    fine_tune_seconds: float
    finetuned_embed_seconds: float
    baseline_embed_seconds: float
    mean_pairwise_similarity_finetuned: float
    mean_pairwise_similarity_baseline: float
    cophenetic_corr_finetuned: float
    cophenetic_corr_baseline: float
    guide_tree_order_changed: bool


def fetch_tfa(dataset_name: str, cache_dir: Path) -> Path:
    """Download BBA000N.tfa from the repo if not already cached locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / f"{dataset_name}.tfa"
    if local_path.exists():
        return local_path

    url = f"{RAW_BASE}/{dataset_name}.tfa"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    local_path.write_bytes(response.content)
    return local_path


def load_sequences(tfa_path: Path, limit: int | None = None) -> list[str]:
    records = list(SeqIO.parse(io.StringIO(tfa_path.read_text()), "fasta"))
    sequences = [str(r.seq).replace("-", "").upper() for r in records]  # strip gaps if any
    if limit:
        sequences = sequences[:limit]
    return sequences


def mean_offdiagonal(similarity_matrix: np.ndarray) -> float:
    n = similarity_matrix.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(similarity_matrix[mask].mean())


def cophenetic_correlation(similarity_matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """How well the guide tree's pairwise distances preserve the embedding distances."""
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    condensed = squareform(distance_matrix, checks=False)
    tree = linkage(condensed, method="average")
    coph_corr, _ = cophenet(tree, condensed)
    return float(coph_corr), tree


def run_one_dataset(
    dataset_name: str,
    cache_dir: Path,
    epochs: int,
    batch_size: int,
    limit: int | None,
) -> DatasetResult:
    print(f"\n=== {dataset_name} ===")
    tfa_path = fetch_tfa(dataset_name, cache_dir)
    sequences = load_sequences(tfa_path, limit=limit)
    print(f"Loaded {len(sequences)} sequences")

    # --- Baseline: frozen pretrained model, no fine-tuning ---
    baseline_aligner = YouShanAligner()
    t0 = time.time()
    baseline_sim = baseline_aligner.compute_similarity_matrix(sequences)
    baseline_embed_seconds = time.time() - t0
    baseline_mean_sim = mean_offdiagonal(baseline_sim)
    baseline_coph, baseline_tree = cophenetic_correlation(baseline_sim)

    # --- Fine-tuned: fresh encoder copy, MLM fine-tuning on this family's own sequences ---
    finetuned_aligner = YouShanAligner()
    t0 = time.time()
    finetuned_aligner.fine_tune(
        sequences, epochs=epochs, batch_size=batch_size, verbose=True
    )
    fine_tune_seconds = time.time() - t0

    t0 = time.time()
    finetuned_sim = finetuned_aligner.compute_similarity_matrix(sequences)
    finetuned_embed_seconds = time.time() - t0
    finetuned_mean_sim = mean_offdiagonal(finetuned_sim)
    finetuned_coph, finetuned_tree = cophenetic_correlation(finetuned_sim)

    order_changed = not np.array_equal(
        np.argsort(baseline_tree[:, 2]), np.argsort(finetuned_tree[:, 2])
    )

    return DatasetResult(
        dataset=dataset_name,
        n_sequences=len(sequences),
        fine_tune_seconds=fine_tune_seconds,
        finetuned_embed_seconds=finetuned_embed_seconds,
        baseline_embed_seconds=baseline_embed_seconds,
        mean_pairwise_similarity_finetuned=finetuned_mean_sim,
        mean_pairwise_similarity_baseline=baseline_mean_sim,
        cophenetic_corr_finetuned=finetuned_coph,
        cophenetic_corr_baseline=baseline_coph,
        guide_tree_order_changed=order_changed,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark YouShanAligner on BBA reference sets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["BBA0001", "BBA0002", "BBA0003", "BBA0004", "BBA0005", "BBA0006"],
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Subsample sequences per dataset (for a quick smoke test)")
    parser.add_argument("--cache-dir", type=Path, default=Path("bba_cache"))
    parser.add_argument("--out", type=Path, default=Path("results/bba_benchmark.csv"))
    args = parser.parse_args()

    results: list[DatasetResult] = []
    for name in args.datasets:
        result = run_one_dataset(
            name,
            cache_dir=args.cache_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        results.append(result)
        print(
            f"  fine-tune: {result.fine_tune_seconds:.1f}s | "
            f"mean sim (ft={result.mean_pairwise_similarity_finetuned:.3f}, "
            f"base={result.mean_pairwise_similarity_baseline:.3f}) | "
            f"cophenetic corr (ft={result.cophenetic_corr_finetuned:.3f}, "
            f"base={result.cophenetic_corr_baseline:.3f}) | "
            f"guide-tree order changed: {result.guide_tree_order_changed}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(args.out, index=False)
    print(f"\nSaved summary to {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
