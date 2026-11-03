"""Self-MM late fusion adapted to the contest's aligned and locally missing inputs.

The paper's four scalar regression branches and modality-specific representations
are retained. The three-class head, missingness flags, and bounded intensity are
contest adaptations, not claims about the original Self-MM implementation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AutoModel, BertConfig


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.rnn = nn.LSTM(input_dim + 1, hidden, batch_first=True)
        self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                    nn.ReLU(), nn.LayerNorm(hidden))

    def forward(self, features, observed, lengths):
        source = torch.cat((features * observed.unsqueeze(-1),
                            torch.zeros_like(observed.unsqueeze(-1).to(features.dtype))), dim=-1)  # E2-13 no_obsflag
        packed = pack_padded_sequence(source, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (hidden, _) = self.rnn(packed)
        result = self.output(hidden[-1])
        return result * observed.any(dim=1).unsqueeze(-1)


class SelfMMContest(nn.Module):
    def __init__(self, bert_name: str, hidden: int = 128, dropout: float = 0.25,
                 freeze_bert_layers: int = 8, bert_config: dict | None = None):
        super().__init__()
        if bert_config is None:
            try:
                self.bert = AutoModel.from_pretrained(bert_name, local_files_only=True)
            except OSError as exc:
                raise FileNotFoundError(
                    "Pretrained bert-base-uncased is not cached. Set BERT_NAME in "
                    "code6/settings.py to its local directory.") from exc
        else:
            self.bert = AutoModel.from_config(BertConfig.from_dict(bert_config))
        if not 0 <= freeze_bert_layers <= len(self.bert.encoder.layer):
            raise ValueError("Invalid count of frozen BERT layers")
        self.freeze_bert_layers = freeze_bert_layers
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        for layer in self.bert.encoder.layer[:freeze_bert_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        if getattr(self.bert, "pooler", None) is not None:
            for p in self.bert.pooler.parameters():
                p.requires_grad = False

        self.text_project = nn.Sequential(nn.Linear(self.bert.config.hidden_size, hidden),
                                          nn.ReLU(), nn.LayerNorm(hidden))
        self.audio_encoder = SequenceEncoder(74, hidden, dropout)
        self.vision_encoder = SequenceEncoder(35, hidden, dropout)
        self.missing_text = nn.Parameter(torch.zeros(hidden))
        self.fusion = nn.Sequential(nn.Linear(3 * hidden + 3, hidden), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                    nn.ReLU(), nn.LayerNorm(hidden))
        self.multimodal_reg = nn.Linear(hidden, 1)
        self.multimodal_cls = nn.Linear(hidden, 3)
        self.unimodal_reg = nn.ModuleDict({
            name: nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                nn.ReLU(), nn.Linear(hidden, 1))
            for name in ("text", "audio", "vision")
        })

    def forward(self, batch: dict):
        bert = self.bert(input_ids=batch["ids"],
                         attention_mask=batch["attn"].long(),
                         token_type_ids=batch["segment"]).last_hidden_state[:, 0]
        observed = batch["observed"].bool()
        content_count = batch["content"].sum(dim=1).clamp_min(1)
        coverage = observed.sum(dim=1).to(bert.dtype) / content_count.unsqueeze(-1)
        text = self.text_project(bert)
        text = torch.where(coverage[:, :1] > 0, text,
                           self.missing_text.unsqueeze(0).expand_as(text))
        lengths = batch["attn"].sum(dim=1).clamp_min(1).long()
        audio = self.audio_encoder(batch["audio"], observed[:, :, 1], lengths)
        vision = self.vision_encoder(batch["vision"], observed[:, :, 2], lengths)
        fusion = self.fusion(torch.cat((text, audio, vision, coverage), dim=-1))
        return {
            "reg": 3.0 * torch.tanh(self.multimodal_reg(fusion).squeeze(-1) / 3.0),
            "logits": self.multimodal_cls(fusion),
            "aux": {name: 3.0 * torch.tanh(self.unimodal_reg[name](value).squeeze(-1) / 3.0)
                    for name, value in (("text", text), ("audio", audio), ("vision", vision))},
            "features": {"fusion": fusion, "text": text, "audio": audio,
                         "vision": vision},
            "availability": coverage,
        }
