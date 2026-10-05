import json
import matplotlib.pyplot as plt
import numpy as np
import os

DAPT_EVAL_DIR = 'C:/Users/user6/T2520814/DAPT_models/results/dapt_eval'
BASE_EVAL_DIR = 'C:/Users/user6/T2520814/DAPT_models/results/baselines'
OUT_PATH = 'C:/Users/user6/T2520814/DAPT_models/results/figures/baseline_vs_dapt_comparison.png'

models = ['BanglaBERT', 'sahajBERT', 'mBERT']
base_f1 = []
dapt_f1 = []

for m in models:
    try:
        with open(os.path.join(BASE_EVAL_DIR, f'{m}_metrics.json'), 'r') as f:
            base_f1.append(json.load(f)['f1'] * 100)
    except:
        base_f1.append(0)
        
    try:
        with open(os.path.join(DAPT_EVAL_DIR, f'{m}_dapt_metrics.json'), 'r') as f:
            dapt_f1.append(json.load(f)['f1'] * 100)
    except:
        dapt_f1.append(0)

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, base_f1, width, label='Baseline', color='#1f77b4')
rects2 = ax.bar(x + width/2, dapt_f1, width, label='DAPT+TAPT', color='#ff7f0e')

ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Baseline vs DAPT+TAPT Performance Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim([75, 90])

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
print('Generated baseline_vs_dapt_comparison.png')
