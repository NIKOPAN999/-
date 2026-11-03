"""Scientific diagnostic figures; no manual boundary annotations are invented."""
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from q1.core import read_json
T=Path('/root/autodl-tmp/team')
run=Path(read_json(ROOT/'work/latest_run.json')['path'])
d=run/'samples/94d7ec6188bed1e2'
x,sr=sf.read(d/'audio.wav')
fig,axes=plt.subplots(3,1,figsize=(13,6),sharex=True,gridspec_kw={'height_ratios':[2,1,1]})
axes[0].plot(np.arange(len(x))/sr,x,color='#34495e',linewidth=.4)
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Pilot word boundaries: model disagreement, not ground-truth error')
for ax,backend,color in zip(axes[1:],['whisperx','mfa'],['#187bcd','#d47925']):
    words=read_json(d/f'alignment_{backend}.json')['words']
    for j,w in enumerate(words):
        if not w['valid']: continue
        ax.add_patch(Rectangle((w['start'],0),w['end']-w['start'],1,alpha=.3,facecolor=color,edgecolor=color))
        ax.text((w['start']+w['end'])/2, .25 if j%2 else .7,w['word'],rotation=30,fontsize=8,ha='center')
    ax.set_ylabel(backend)
    ax.set_yticks([])
    ax.set_ylim(0,1.2)
axes[-1].set_xlim(0,len(x)/sr)
axes[-1].set_xlabel('Seconds from prepared audio start')
fig.tight_layout()
for target in [T/'q1/exp/Q1-2_alignment_validation/figures',T/'reports/q1/figures']:
    target.mkdir(exist_ok=True,parents=True)
    fig.savefig(target/'Q1_pilot_alignment.png',dpi=160)
    fig.savefig(target/'Q1_pilot_alignment.pdf')
plt.close(fig)
