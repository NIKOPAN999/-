"""Independent greedy decoding diagnostic using the existing alignment model."""
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForCTC
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from q1.core import read_json,write_json
run=Path(read_json(ROOT/'work/latest_run.json')['path'])
d=run/'samples/94d7ec6188bed1e2'
x,sr=sf.read(d/'audio.wav')
processor=AutoProcessor.from_pretrained(ROOT/'models/align',local_files_only=True)
model=AutoModelForCTC.from_pretrained(ROOT/'models/align',local_files_only=True).to('cuda').eval()
inputs=processor(x,sampling_rate=sr,return_tensors='pt')
with torch.inference_mode():
    ids=model(inputs.input_values.to('cuda')).logits.argmax(-1).cpu()[0]
decoded=processor.tokenizer.decode(ids,output_word_offsets=True)
scale=model.config.inputs_to_logits_ratio/sr
result=dict(transcript=decoded.text,words=[dict(word=w['word'],start=w['start_offset']*scale,end=w['end_offset']*scale) for w in decoded.word_offsets],
            note='Greedy CTC recognition, not a human reference and not used to replace the supplied transcript')
write_json(d/'greedy_ctc_diagnostic.json',result)
print(result)
