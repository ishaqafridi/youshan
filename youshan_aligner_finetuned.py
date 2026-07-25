"""
You-Shan: A Fine-Tuned Transformer for High-Fidelity Multiple Sequence Alignment
----------------------------------------------------------------------------------
CORRECTED IMPLEMENTATION

What was wrong in the original notebooks
(Youshan_algorithms_implmtin_python.ipynb / revised_coding_file.ipynb):

    def fine_tune_embeddings(self, sequences, masking_probability=0.15):
        ...
        with torch.no_grad():
            embedding = self.model(**inputs).last_hidden_state.mean(dim=1)
        ...

`torch.no_grad()` disables autograd. No loss is computed, no `.backward()`
is called, and no optimizer ever updates a weight. The method masks the
*input* tokens but leaves the *model* frozen, so despite the name, nothing
about the transformer is fine-tuned -- it is used purely as a frozen
feature extractor (this is what the JCB editor's decision correctly flagged).

What this file does instead
----------------------------
1. Loads Rostlab/prot_bert_bfd as an AutoModelForMaskedLM (needed to compute
   an MLM loss, since AutoModel alone has no LM head and nothing to
   backpropagate against).
2. Implements an actual masked-language-modeling fine-tuning loop:
   - model.train()
   - real token masking with correct label construction (-100 for
     unmasked positions, per HuggingFace convention)
   - forward pass -> loss -> loss.backward() -> optimizer.step()
   - gradients flow into the transformer's weights and are updated
3. Saves/loads fine-tuned weights, so the "fine-tuned" model used at
   inference time is actually different from the pretrained checkpoint.
4. Uses the fine-tuned encoder (frozen only at *inference* time, which is
   correct and standard practice -- you fine-tune once, then infer without
   gradients) to generate embeddings for progressive alignment via
   hierarchical clustering, same downstream pipeline as the original code.

Install:
    pip install torch transformers biopython scipy scikit-learn --break-system-packages
"""

import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from scipy.cluster.hierarchy import linkage
from transformers import AutoModelForMaskedLM, AutoTokenizer


class YouShanAligner:
    """
    Wraps a protein language model (default: ProtBERT-BFD) and provides:
      - genuine gradient-based fine-tuning via masked language modeling (MLM)
      - embedding extraction from the fine-tuned encoder
      - similarity-matrix / guide-tree construction for progressive MSA
    """

    def __init__(
        self,
        model_name: str = "Rostlab/prot_bert_bfd",
        token: str | None = None,
        device: str | None = None,
    ):
        token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        # AutoModelForMaskedLM (not AutoModel) -- this is what makes a real
        # MLM fine-tuning loss possible. AutoModel has no LM head, so there
        # is nothing to compute a masked-token loss against.
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, token=token).to(self.device)
        self._is_fine_tuned = False

    # ------------------------------------------------------------------ #
    # Genuine fine-tuning
    # ------------------------------------------------------------------ #
    def fine_tune(
        self,
        sequences: list[str],
        epochs: int = 3,
        batch_size: int = 4,
        masking_probability: float = 0.15,
        lr: float = 2e-5,
        max_length: int = 512,
        verbose: bool = True,
    ):
        """
        Fine-tunes the underlying transformer on `sequences` using a masked
        language modeling objective. This performs real weight updates
        (unlike the original `torch.no_grad()`-wrapped method).
        """
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=lr)

        spaced_sequences = [self._space_residues(s) for s in sequences]

        for epoch in range(epochs):
            random.shuffle(spaced_sequences)
            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, len(spaced_sequences), batch_size):
                batch = spaced_sequences[batch_start : batch_start + batch_size]
                if not batch:
                    continue

                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                masked_input_ids, labels = self._mask_tokens(
                    input_ids, masking_probability
                )

                outputs = self.model(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()          # <-- gradients actually computed
                optimizer.step()         # <-- weights actually updated

                epoch_loss += loss.item()
                n_batches += 1

            if verbose and n_batches:
                print(f"[epoch {epoch + 1}/{epochs}] mean MLM loss = {epoch_loss / n_batches:.4f}")

        self.model.eval()
        self._is_fine_tuned = True

    def _mask_tokens(self, input_ids: torch.Tensor, masking_probability: float):
        """
        Standard HuggingFace-style MLM masking: 15% of non-special tokens
        are candidates; labels are -100 (ignored by loss) everywhere except
        the masked positions, where the label is the original token id.
        """
        labels = input_ids.clone()

        special_tokens_mask = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(seq.tolist(), already_has_special_tokens=True)
                for seq in input_ids
            ],
            dtype=torch.bool,
            device=input_ids.device,
        )

        probability_matrix = torch.full(labels.shape, masking_probability, device=input_ids.device)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100  # only compute loss on masked tokens

        # 80% -> [MASK], 10% -> random token, 10% -> unchanged (standard BERT recipe)
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool() & masked_indices
        input_ids = input_ids.clone()
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_tokens = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=input_ids.device)
        input_ids[indices_random] = random_tokens[indices_random]

        return input_ids, labels

    @staticmethod
    def _space_residues(sequence: str) -> str:
        """ProtBERT expects space-separated residues, e.g. 'M E T H I O ...'."""
        return " ".join(list(sequence.strip()))

    # ------------------------------------------------------------------ #
    # Inference (frozen, post-fine-tuning -- this no_grad usage is correct)
    # ------------------------------------------------------------------ #
    def get_embedding(self, sequence: str) -> np.ndarray:
        inputs = self.tokenizer(
            self._space_residues(sequence),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():  # correct here: we're doing inference, not training
            outputs = self.model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1)

        return embedding.squeeze().cpu().numpy()

    def compute_similarity_matrix(self, sequences: list[str]) -> np.ndarray:
        embeddings = np.array([self.get_embedding(seq) for seq in sequences])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.clip(norms, 1e-8, None)
        return normalized @ normalized.T

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        self.model = AutoModelForMaskedLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.eval()
        self._is_fine_tuned = True


def youshan_progressive_msa(sequences: list[str], aligner: YouShanAligner):
    """Guide-tree-based progressive alignment using the aligner's embeddings."""
    similarity_matrix = aligner.compute_similarity_matrix(sequences)
    guide_tree = linkage(1 - similarity_matrix, method="average")
    sorted_indices = np.argsort(guide_tree[:, 2])
    aligned_order = [sequences[i] for i in sorted_indices]
    return aligned_order, guide_tree


if __name__ == "__main__":
    # Minimal smoke test with toy sequences -- replace with your NCBI/BBA
    # training set for the real benchmarking run.
    demo_sequences = [
        "MKTFFVLLLCTFTVFA",
        "MKTAYVLLLCTFTVFA",
        "MKTFFILLLCTFAVFA",
        "GATTACAGATTACAGA",
    ]

    aligner = YouShanAligner()

    t0 = time.time()
    aligner.fine_tune(demo_sequences, epochs=2, batch_size=2)
    print(f"Fine-tuning took {time.time() - t0:.1f}s")

    order, tree = youshan_progressive_msa(demo_sequences, aligner)
    print("Guide-tree ordering:", order)
