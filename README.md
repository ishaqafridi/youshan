You-Shan: A Fine-Tuned Transformer for High-Fidelity Multiple Sequence Alignment

You-Shan couples a protein language model (default: Rostlab/prot_bert_bfd) with genuine masked-language-model (MLM) fine-tuning to produce sequence embeddings for guide-tree-based progressive multiple sequence alignment (MSA).

What's in this repo
File	Purpose
youshan_aligner_finetuned.py	Core YouShanAligner class: MLM fine-tuning loop, embedding extraction, similarity-matrix and guide-tree construction.
run_bba_benchmark.py	Downloads the six BBA000N.tfa reference datasets from this repo, fine-tunes and benchmarks the aligner on each, and compares against a frozen (non-fine-tuned) baseline.
BBA0001.tfa – BBA0006.tfa	BAliBASE-style reference protein family datasets used for benchmarking (23–248 sequences each).
Dataset_Extraction_and_Organization_Script.ipynb	Dataset preparation utilities.
Method
Fine-tuning. For each protein family, a fresh copy of the pretrained encoder is fine-tuned via masked language modeling directly on that family's own sequences (transductive, self-supervised fine-tuning — no external labels required). Fine-tuning follows the standard BERT masking recipe: 15% of residues are candidates for masking, of which 80% are replaced with [MASK], 10% with a random residue, and 10% left unchanged; loss is computed only at masked positions.
Embedding. After fine-tuning, the encoder is frozen and used to generate a fixed-length embedding per sequence (mean-pooled final hidden state).
Guide tree / progressive alignment. Pairwise cosine similarity between embeddings forms a similarity matrix, from which a guide tree is built via average-linkage hierarchical clustering, giving the sequence order for progressive alignment.

A per-family fine-tuning strategy (rather than one model fine-tuned across all families) was used because the benchmark datasets are unrelated protein superfamilies; pooling them risks blurring family-specific signal and catastrophic forgetting between unrelated families.

Installation
bash
pip install torch transformers biopython scipy scikit-learn pandas requests

A CUDA-capable GPU is strongly recommended — fine-tuning on the larger reference sets (BBA0003: 126 sequences, BBA0004: 248 sequences) is slow on CPU.

Usage
bash
# Quick smoke test (1 epoch, 5 sequences, fast)
python run_bba_benchmark.py --datasets BBA0001 --epochs 1 --limit 5

# Full benchmark across all six reference sets
python run_bba_benchmark.py --datasets BBA0001 BBA0002 BBA0003 BBA0004 BBA0005 BBA0006 \
    --epochs 3 --batch-size 4 --out results/bba_benchmark.csv

Each run reports, per dataset, the fine-tuning time, embedding-extraction time, mean pairwise embedding similarity, and cophenetic correlation of the guide tree — for both the fine-tuned model and a frozen-pretrained baseline, so the effect of fine-tuning can be inspected directly rather than assumed.

Known limitations
Metrics currently reported (mean pairwise similarity, cophenetic correlation) describe internal consistency of the embedding/guide-tree pipeline. They are not alignment-accuracy metrics. A proper SP-score / TC-score comparison against BAliBASE reference alignments (where available) is recommended before citing accuracy claims.
Full-parameter fine-tuning on small per-family datasets carries a risk of overfitting/catastrophic forgetting of the base model's pretrained protein-language knowledge; parameter-efficient fine-tuning (e.g., LoRA, last-N-layer unfreezing) is worth evaluating as an alternative, particularly for the smaller families (BBA0001, BBA0005).
Citation

If you use this code, please cite:

[Author names]. "You-Shan: A Fine-Tuned Transformer for High-Fidelity Multiple Sequence Alignment." [Journal], [Year].

License

GPL-3.0 (see LICENSE)
