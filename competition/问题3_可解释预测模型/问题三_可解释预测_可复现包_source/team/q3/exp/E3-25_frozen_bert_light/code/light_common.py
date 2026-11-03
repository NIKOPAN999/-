"""E3-25 轻量版 Self-MM 的公共部分：路径、文本池化补丁、模型构建与紧凑权重读写。

轻量版与正式版 Self-MM 结构相同，只有两处改动：
1. BERT 参数全部冻结（use_finetune=false），权重与公开的 bert-base-uncased 完全一致，不随模型保存；
2. 可选把文本表示由 [CLS] 向量改为有效词元的平均（--pool mean），因为未微调 BERT 的 [CLS] 不是好的句子表示。
"""
import hashlib
import json
from pathlib import Path

import torch

TEAM = Path(__file__).resolve().parents[4]
BASE = TEAM / 'shared/baselines'
EXP = Path(__file__).resolve().parents[1]
BERT_DIR = BASE / 'pretrained/bert-base-uncased'
DATA_ATT2 = BASE / 'data/mosei_att2_aligned50.pkl'
DATA_ATT4 = BASE / 'data/mosei_att4_aligned50_infer.pkl'


def add_mmsa_path():
    import sys
    p = str(BASE / 'repos/MMSA/src')
    if p not in sys.path:
        sys.path.insert(0, p)


POOLS = ('cls', 'mean', 'mean4', 'mean4cat', 'attn4')


def bert_forward(bert, input_ids, attn, seg, pool, inputs_embeds=None):
    return bert(input_ids=None if inputs_embeds is not None else input_ids, inputs_embeds=inputs_embeds,
                attention_mask=attn, token_type_ids=seg, output_hidden_states=pool in ('mean4', 'mean4cat', 'attn4'))


def pool_hidden(out, attn, pool, scorer=None):
    """由 BERT 输出得到句子表示。cls：末层 [CLS]；mean：末层有效词元平均；
    mean4：末四层平均后再对有效词元平均；mean4cat：mean4 与末层 [CLS] 拼接（1536 维）；
    attn4：末四层平均的正文词元上做可学习注意力池化（打分器为 768→1 线性层，仅 769 个参数）。"""
    last = out[0]
    if pool == 'cls':
        return last[:, 0]
    m = attn.float().unsqueeze(-1)
    mp = lambda h: (h * m).sum(1) / m.sum(1).clamp(min=1.0)
    if pool == 'mean':
        return mp(last)
    h4 = torch.stack(out.hidden_states[-4:]).mean(0)
    if pool == 'attn4':
        body = attn.clone()
        body[:, 0] = 0
        body[torch.arange(len(body)), attn.sum(1).long() - 1] = 0      # 去掉 [CLS]、[SEP]
        e = scorer(h4).squeeze(-1).masked_fill(body == 0, -1e4)
        return (torch.softmax(e, -1).unsqueeze(-1) * h4).sum(1)
    return mp(h4) if pool == 'mean4' else torch.cat([mp(h4), last[:, 0]], -1)


def text_repr(bert, input_ids, attn, seg, pool, inputs_embeds=None, scorer=None, no_grad_bert=False):
    with torch.set_grad_enabled(torch.is_grad_enabled() and not no_grad_bert):
        out = bert_forward(bert, input_ids, attn, seg, pool, inputs_embeds)
    return pool_hidden(out, attn, pool, scorer)


def apply_pool_patch(pool):
    """让文本编码器的输出第 0 个位置等于所选句子表示；Self-MM 只取第 0 个位置。cls 保持原版行为。"""
    add_mmsa_path()
    from MMSA.models.subNets.BertTextEncoder import BertTextEncoder
    if pool == 'cls' or getattr(BertTextEncoder, '_e325_pool', None) == pool:
        return
    assert pool in POOLS

    def forward(self, text):
        ids, attn, seg = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if pool == 'attn4':
            v = text_repr(self.model, ids, attn, seg, pool, scorer=self.e325_scorer, no_grad_bert=True)
        else:
            with torch.no_grad():
                v = text_repr(self.model, ids, attn, seg, pool)
        return v.unsqueeze(1)

    if pool == 'attn4':
        orig_init = BertTextEncoder.__init__

        def __init__(self, *args, **kw):
            orig_init(self, *args, **kw)
            self.e325_scorer = torch.nn.Linear(768, 1)
            torch.nn.init.zeros_(self.e325_scorer.weight)      # 初始为均匀权重，即等价于 mean4
            torch.nn.init.zeros_(self.e325_scorer.bias)

        BertTextEncoder.__init__ = __init__
    BertTextEncoder.forward = forward
    BertTextEncoder._e325_pool = pool


def materialize_config():
    text = (BASE / 'configs/mmsa/config_regression_att2.json').read_text(encoding='utf-8').replace('{BASELINES}', BASE.as_posix())
    p = EXP / 'work/config_att2_e325.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')
    return p


OVERRIDE = {'need_data_aligned': True, 'use_bert': True, 'use_finetune': False}


def override_for(pool):
    o = dict(OVERRIDE)
    if pool == 'mean4cat':
        o['text_out'] = 1536
    if pool == 'attn4':
        o['learning_rate_bert'] = 1e-3       # BERT 本身无梯度，该学习率只作用于 769 个注意力参数
    return o


def build_args(custom_feature=DATA_ATT2, pool='cls'):
    add_mmsa_path()
    from MMSA.config import get_config_regression
    args = get_config_regression('self_mm', 'mosei', str(materialize_config()))
    args.update(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), train_mode='regression', cur_seed=1,
                custom_feature=str(custom_feature), feature_T=None, feature_A=None, feature_V=None,
                model_save_path=str(EXP / 'work/unused.pth'))
    args.update(override_for(pool))
    return args


def bert_fingerprint(state):
    """公开 BERT 参数的指纹：按名称排序后对全部张量字节求 SHA256。"""
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def split_state(full_state):
    """把 MMSA 保存的整模型参数拆成 BERT 部分和其余部分。"""
    bert = {k[len('Model.text_model.'):]: v for k, v in full_state.items() if k.startswith('Model.text_model.model.')}
    rest = {k: v for k, v in full_state.items() if not k.startswith('Model.text_model.model.')}
    return bert, rest


def load_light_model(compact_path, pool, device=None):
    """由紧凑权重（不含 BERT）+ 公开 BERT 重建完整 Self-MM，并核对 BERT 指纹。"""
    add_mmsa_path()
    apply_pool_patch(pool)
    from MMSA.models import AMIO
    args = build_args(pool=pool)
    if device is not None:
        args.update(device=device)
    pack = torch.load(compact_path, map_location='cpu', weights_only=False)
    model = AMIO(args)
    missing, unexpected = model.load_state_dict(pack['state'], strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith('Model.text_model.model.') for k in missing), [k for k in missing if not k.startswith('Model.text_model.model.')][:5]
    bert, _ = split_state(model.state_dict())
    fp = bert_fingerprint(bert)
    assert fp == pack['bert_fingerprint'], '公开 BERT 与训练时不一致'
    return model.to(args.device).eval(), args, pack


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
