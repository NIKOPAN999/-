"""CPU structural check; no competition data or pretrained weights required."""
import torch
import torch.nn.functional as F
from transformers import BertConfig

from data import block_mask
from model import SelfMMContest
from ulgm import UnimodalLabelBank


def main():
    torch.manual_seed(7)
    config = BertConfig(vocab_size=30522, hidden_size=64, num_hidden_layers=2,
                        num_attention_heads=4, intermediate_size=128,
                        max_position_embeddings=64)
    model = SelfMMContest("unused", hidden=32, dropout=0.1,
                          freeze_bert_layers=1, bert_config=config.to_dict())
    ids = torch.zeros(2, 50, dtype=torch.long)
    ids[:, 0], ids[:, 9] = 101, 102
    ids[:, 1:9] = torch.randint(1000, 3000, (2, 8))
    attn = torch.zeros(2, 50, dtype=torch.bool)
    attn[:, :10] = True
    content = torch.zeros_like(attn)
    content[:, 1:9] = True
    observed = torch.stack((content, content, content), dim=-1)
    audio = torch.randn(2, 50, 74) * observed[:, :, 1:2]
    vision = torch.randn(2, 50, 35) * observed[:, :, 2:3]
    batch = {"index": torch.arange(2), "ids": ids, "attn": attn,
             "segment": torch.zeros_like(ids), "content": content,
             "observed": observed, "audio": audio, "vision": vision,
             "cls": torch.tensor([0, 2]), "reg": torch.tensor([-1.0, 1.0])}
    damaged = block_mask(batch, (0, 1, 2), 0.5, "middle")
    result = model(damaged)
    assert result["logits"].shape == (2, 3)
    assert result["reg"].shape == (2,)
    loss = F.cross_entropy(result["logits"], batch["cls"]) + \
           F.l1_loss(result["reg"], batch["reg"])
    loss.backward()
    bank = UnimodalLabelBank(batch["reg"], 32)
    bank.begin_epoch()
    bank.record(batch["index"], result["features"])
    assert not bank.update(1)["updated"]
    bank.begin_epoch()
    bank.record(batch["index"], result["features"])
    assert bank.update(2)["updated"]
    print("Self-MM contest adaptation smoke check passed; no pretrained weights or competition training used")


if __name__ == "__main__":
    main()
