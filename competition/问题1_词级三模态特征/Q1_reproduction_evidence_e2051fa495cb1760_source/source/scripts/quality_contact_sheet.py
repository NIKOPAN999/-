from pathlib import Path
import sys
import subprocess
import io
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from q1.core import read_json
run=Path(read_json(ROOT/'work/latest_run.json')['path'])
records=[]
for p in (run/'samples').glob('*/complete.json'):
    meta=read_json(p.parent/'metadata.json')
    if meta['quality']['vision_word_coverage']<.7:
        records.append(meta)
records.sort(key=lambda m:m['quality']['vision_word_coverage'])
out=Path('/root/autodl-tmp/team/reports/q1/figures')
out.mkdir(exist_ok=True,parents=True)
for page in range(0,len(records),6):
    group=records[page:page+6]
    canvas=Image.new('RGB',(960,210*len(group)), 'white')
    draw=ImageDraw.Draw(canvas)
    for i,m in enumerate(group):
        row=m['sample']; src=ROOT/'data/raw100'/row['relative_path']
        draw.text((8,i*210+5),f"{row['sample_id']} | visual word coverage {m['quality']['vision_word_coverage']:.1%}",fill='black')
        for j,fraction in enumerate([.15,.5,.85]):
            seconds=m['media']['clip_duration']*fraction
            data=subprocess.check_output(['/root/autodl-tmp/envs/q1-mfa/bin/ffmpeg','-threads','1','-v','error','-ss',str(seconds),'-i',str(src),'-frames:v','1','-vf','scale=310:170:force_original_aspect_ratio=decrease','-f','image2pipe','-vcodec','png','-threads','1','-'],timeout=30)
            image=Image.open(io.BytesIO(data)).convert('RGB')
            canvas.paste(image,(j*320,i*210+25))
            draw.text((j*320+8,i*210+195),f'{seconds:.2f}s',fill='black')
    target=out/f'Q1_low_vision_contact_{page//6+1}.jpg'
    canvas.save(target,quality=90)
    print(target)
