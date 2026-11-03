"""Export times at exactly the supplied attachment-4 text_bert positions."""
from pathlib import Path
import argparse
import json
import pickle
import sys

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q1.aligners import Aligners
from q1.core import read_json, sha256, write_csv, write_json
from q1.media import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--videos', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(ROOT / 'configs/local.json')
    for name in ['bert_model_path', 'align_model_path', 'mfa_dictionary', 'mfa_acoustic']:
        config[name] = str((ROOT / config[name]).resolve())
    config['data_root'] = str(Path(args.videos).resolve())
    # Trusted competition data, also used by the shared baseline loader.
    with open(args.features, 'rb') as f:
        data = pickle.load(f)['test']
    tok = AutoTokenizer.from_pretrained(config['bert_model_path'], use_fast=True, local_files_only=True)
    aligners = Aligners(config)
    rows, checks, sources = [], [], []
    for sample_id, raw, text_bert in zip(data['id'], data['raw_text'], data['text_bert']):
        sample_id, raw = str(sample_id), str(raw)
        ids = np.asarray(text_bert[0], dtype=np.int64)
        mask = np.asarray(text_bert[1], dtype=np.int64)
        enc = tok(raw, max_length=len(ids), padding='max_length', truncation=True,
                  return_offsets_mapping=True, return_special_tokens_mask=True)
        if not np.array_equal(ids, enc['input_ids']) or not np.array_equal(mask, enc['attention_mask']):
            raise ValueError(f'{sample_id}: raw_text tokenization differs from supplied text_bert; refusing guessed positions')
        source = Path(args.videos) / f'{sample_id}.mp4'
        row = dict(sample_id=sample_id, raw_text=raw, relative_path=source.name, sha256=sha256(source))
        directory = output / 'outputs' / sample_id
        media = prepare(config, row, directory)
        alignment = aligners.run('whisperx', row, directory, media)
        words = alignment['words']
        valid, lexical = 0, 0
        for pos, (token_id, offsets, special, attention) in enumerate(zip(ids, enc['offset_mapping'], enc['special_tokens_mask'], mask)):
            a, b = offsets
            matches = [w for w in words if max(a, w['char_start']) < min(b, w['char_end'])] if attention and not special else []
            mapped = matches[0] if len(matches) == 1 else None
            good = mapped is not None and mapped['valid']
            lexical += int(mapped is not None)
            valid += int(good)
            rows.append(dict(sample_id=sample_id, token_pos=pos, wordpiece=tok.convert_ids_to_tokens(int(token_id)),
                             word_idx=mapped['index'] if mapped else -1, word=mapped['word'] if mapped else '',
                             start_sec=mapped['start'] + media['audio_offset_seconds'] if good else -1,
                             end_sec=mapped['end'] + media['audio_offset_seconds'] if good else -1,
                             align_conf=mapped['confidence'] if good else '',
                             status='aligned' if good else ('padding' if not attention else 'special' if special else 'nonlexical' if not mapped else mapped['reason'])))
        checks.append(dict(sample_id=sample_id, token_positions=len(ids), token_ids_exact=True,
                           lexical_positions=lexical, aligned_positions=valid,
                           word_coverage=alignment['valid_words']/alignment['total_words'],
                           lexical_token_coverage=valid/lexical if lexical else 0,
                           low_score_words=sum(w['valid'] and (w['confidence'] is None or w['confidence']<0.3) for w in words)))
        sources.append(row)
        print(json.dumps(checks[-1]), flush=True)
    write_csv(output / 'tables/att4_word_times.csv', rows)
    write_csv(output / 'tables/att4_word_time_checks.csv', checks)
    flagged = [r for r in rows if r['status']=='aligned' and (r['align_conf'] is None or r['align_conf']<0.3)]
    write_csv(output / 'tables/att4_low_confidence_tokens.csv', flagged, fields=list(rows[0]))
    write_json(output / 'outputs/provenance.json', dict(features_sha256=sha256(args.features), videos=sources,
               backend='whisperx', align_model_files={p.name:sha256(p) for p in Path(config['align_model_path']).glob('*') if p.is_file()},
               script_sha256=sha256(__file__), coordinate='source presentation time minus source origin; audio offset included',
               confidence='WhisperX acoustic score, not a calibrated probability of boundary correctness',
               invalid_times=-1, special_and_padding='Preserved at exact text_bert position; no fabricated time'))


if __name__ == '__main__':
    main()
