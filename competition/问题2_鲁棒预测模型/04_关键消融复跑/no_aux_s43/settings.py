"""Fixed defaults for an independent Self-MM contest experiment."""
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CODE = Path(__file__).resolve().parent
DATA_FILE = ROOT / "E题数据" / "附件2-数据集特征文件" / "aligned_50.pkl"
ATTACHMENT3 = ROOT / "E题数据" / "附件3-模态缺失特征样本" / "对齐版本"
OUTPUT_DIR = CODE / "outputs" / "selfmm_competition_run01"
BERT_NAME = "bert-base-uncased"  # Or an exact local bert-base-uncased directory.

CONFIG = {
    "seed": 43,
    "batch_size": 8,
    "epochs": 20,
    "patience": 4,
    "hidden": 128,
    "dropout": 0.25,
    "freeze_bert_layers": 8,
    "bert_lr": 1e-5,
    "lr": 2e-4,
    "weight_decay": 0.01,
    "classification_weight": 0.8,
    "regression_weight": 1.0,
    "auxiliary_weight": 0.0,
    "neutral_class_weight": 1.15,
    "mask_probability": 0.55,
    "text_bias": 0.6,
    "max_mask_ratio": 0.6,
    "triple_probability": 0.2,
}
