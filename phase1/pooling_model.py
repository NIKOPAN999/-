"""Controlled alternatives to the contest audio/vision last-state pooling."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model import SelfMMContest, SequenceEncoder


class ObservedPoolingEncoder(SequenceEncoder):
    def __init__(self, input_dim, hidden, dropout, pooling):
        super().__init__(input_dim, hidden, dropout)
        if pooling not in ("last", "mean", "attention"):
            raise ValueError(pooling)
        self.pooling = pooling
        if pooling == "attention":
            # Do not alter the RNG sequence used by the shared network/dropout.
            rng = torch.random.get_rng_state()
            self.score = nn.Linear(hidden, 1)
            nn.init.zeros_(self.score.weight)
            nn.init.zeros_(self.score.bias)
            torch.random.set_rng_state(rng)

    def forward(self, features, observed, lengths):
        if self.pooling == "last":
            return super().forward(features, observed, lengths)
        mask = observed.bool()
        source = torch.cat((features * mask.unsqueeze(-1),
                            mask.unsqueeze(-1).to(features.dtype)), dim=-1)
        packed = pack_padded_sequence(source, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        output, _ = self.rnn(packed)
        states, _ = pad_packed_sequence(output, batch_first=True,
                                        total_length=features.shape[1])
        if self.pooling == "mean":
            weight = mask.to(states.dtype)
        else:
            logits = self.score(states).squeeze(-1).masked_fill(~mask, -1e4)
            weight = torch.softmax(logits, dim=1) * mask
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pooled = (states * weight.unsqueeze(-1)).sum(dim=1)
        return self.output(pooled) * mask.any(dim=1).unsqueeze(-1)


class Phase1Model(SelfMMContest):
    def __init__(self, config, bert_config=None):
        super().__init__(config["bert_name"], config["hidden"], config["dropout"],
                         config["freeze_bert_layers"], bert_config)
        for name, dim in (("audio", config["audio_dim"]), ("vision", config["vision_dim"])):
            old = getattr(self, name + "_encoder")
            # Last/mean have exactly the same parameter tensors; attention adds
            # only one scalar scoring layer per branch, initialized to mean pooling.
            rng = torch.random.get_rng_state()
            new = ObservedPoolingEncoder(dim, config["hidden"], config["dropout"],
                                         config["pooling"])
            if old.rnn.input_size == dim + 1:
                new.load_state_dict(old.state_dict(), strict=False)
                torch.random.set_rng_state(rng)
            setattr(self, name + "_encoder", new)
        self.text_only = config.get("text_only", False)

    def forward(self, batch):
        if self.text_only:
            batch = dict(batch)
            batch["audio"] = torch.zeros_like(batch["audio"])
            batch["vision"] = torch.zeros_like(batch["vision"])
            batch["observed"] = batch["observed"].clone()
            batch["observed"][:, :, 1:] = False
        return super().forward(batch)
