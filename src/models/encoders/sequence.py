"""
src/models/encoders/sequence.py

The sequence view: ChemBERTa read as a language model over SMILES, adapted with LoRA.

    from src.models.encoders.sequence import SequenceEncoder
    enc = SequenceEncoder(lora_r=8)          # LoRA-adapted
    enc = SequenceEncoder(frozen=True)       # the inherited frozen reference

WHAT CHANGES FROM THE INHERITED VERSION
---------------------------------------
The inherited pipeline froze ChemBERTa completely and read only its `[CLS]` token. Two
things were left on the table:

1. *Nothing adapted to the task.* The encoder was pretrained on ZINC to model SMILES
   strings in general; it never learned anything about toxicity or solubility. Every
   dataset got the same vectors.
2. *One token summarised the molecule.* `[CLS]` is a single position. Mean-pooling over
   the real tokens uses the whole string, and the two summarise differently, so we keep
   both and concatenate: `h = [CLS ; masked mean]`, giving 2 x hidden.

WHY LoRA RATHER THAN FULL FINE-TUNING
-------------------------------------
Full fine-tuning updates all 44M parameters of the encoder. On datasets of 513 to 6,258
training molecules that is a very large model fitted to very little data, and it needs a
fresh 44M-parameter copy per dataset per seed -- 48 of them for the full protocol.

LoRA (Low-Rank Adaptation) freezes the pretrained weights and inserts a small trainable
correction beside each attention projection. Instead of learning a full d x d update it
learns two thin matrices, d x r and r x d, with r=8 here; their product has the same shape
as the weight it corrects but only 2dr parameters instead of d^2. The pretrained knowledge
is untouched, the adapter is a fraction of a percent of the model, and what gets saved per
run is the adapter rather than a whole encoder.

Note on cost: LoRA saves *parameters and memory*, not much compute. The backward pass
still runs through every layer to reach the adapters, so a training step costs about what
full fine-tuning costs. The saving that matters for us is statistical -- far fewer free
parameters against small datasets -- and in storage.

MASKED MEAN POOLING IS NOT OPTIONAL HERE
----------------------------------------
Batches are padded, and with length bucketing the amount of padding differs from batch to
batch. A plain `.mean(dim=1)` would average the padding in, so the same molecule would
encode differently depending on which batch it landed in. The mean below is taken over
real tokens only, using the attention mask.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

# Which projections receive an adapter. Query and value is the standard LoRA placement:
# it is where the original paper found most of the benefit, and adapting keys as well
# adds parameters for little gain.
LORA_TARGETS = ("query", "value")


class SequenceEncoder(nn.Module):
    """
    ChemBERTa over SMILES tokens -> a molecule embedding.

    `frozen=True` reproduces the inherited encoder (no adaptation) under this project's
    shared trainer, so the LoRA row in the results table has a matched control.
    """

    def __init__(self, model_name=MODEL_NAME, lora_r=8, lora_alpha=16, lora_dropout=0.1,
                 pooling="cls+mean", frozen=False):
        super().__init__()
        if pooling not in ("cls", "mean", "cls+mean"):
            raise ValueError(f"Unknown pooling: {pooling}")

        base = AutoModel.from_pretrained(model_name)
        hidden = base.config.hidden_size
        self.pooling = pooling
        self.frozen = frozen

        if frozen:
            for p in base.parameters():
                p.requires_grad = False
            self.encoder = base
        else:
            # Imported here so the frozen path does not require peft to be installed.
            from peft import LoraConfig, TaskType, get_peft_model

            cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(LORA_TARGETS),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            try:
                self.encoder = get_peft_model(base, cfg)
            except ImportError as e:
                # peft probes every optional backend while placing adapters, and some of
                # those probes raise instead of returning False when a package is present
                # but too old. Nothing here uses quantised LoRA, so the offending package
                # is not needed at all -- removing it is safer than upgrading it, which
                # can drag a different torch build along with it.
                if "torchao" in str(e):
                    raise ImportError(
                        f"peft could not place LoRA adapters: {e}" + 2 * chr(10)
                        + "This is an optional backend that this project does not "
                        + "use. Uninstall it and re-run:" + chr(10)
                        + "    pip uninstall -y torchao"
                    ) from e
                raise

        self.out_dim = hidden * (2 if pooling == "cls+mean" else 1)

    def train(self, mode=True):
        """
        Keep a frozen encoder in eval mode even inside a training loop.

        `model.train()` propagates to every submodule and re-enables dropout. Phase 0
        found exactly this bug in the inherited transformer: the encoder's weights were
        frozen, but its dropout was still active, so a "frozen" encoder emitted a
        different vector for the same molecule on every epoch. A LoRA encoder *should*
        train normally; a frozen one must not.
        """
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, batch):
        """`batch` is (input_ids, attention_mask), both (B, L) integer tensors."""
        input_ids, attention_mask = batch
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        tokens = out.last_hidden_state                      # (B, L, H)

        parts = []
        if self.pooling in ("cls", "cls+mean"):
            parts.append(tokens[:, 0])
        if self.pooling in ("mean", "cls+mean"):
            # Average over real tokens only; padding must not enter the mean.
            m = attention_mask.unsqueeze(-1).to(tokens.dtype)
            parts.append((tokens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9))
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
