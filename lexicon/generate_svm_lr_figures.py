import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create results directory if it doesn't exist
os.makedirs('../results', exist_ok=True)

# Set global plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# ====================================================================
# FIGURE 4: Ablation Study Comparison (Macro F1) across LR and SVMs
# ====================================================================
models = ['Logistic Regression', 'Linear SVM', 'RBF SVM']
tfidf_only = [0.6845, 0.6800, 0.6700]
exact_lex = [0.6960, 0.6900, 0.6900]
semantic_lex = [0.7062, 0.6800, 0.6900]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
rects1 = ax.bar(x - width, tfidf_only, width, label='TF-IDF Only (Baseline)', color='#d3d3d3', edgecolor='black')
rects2 = ax.bar(x, exact_lex, width, label='+ Exact Lexicon', color='#4682b4', edgecolor='black')
rects3 = ax.bar(x + width, semantic_lex, width, label='+ Semantic Lexicon', color='#191970', edgecolor='black')

ax.set_ylabel('Macro F1-Score')
ax.set_title('Lexicon Ablation Impact Across Classical Models')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0.60, 0.75) # Zoom in to highlight the differences
ax.legend(loc='upper left')

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, rotation=0)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('../results/fig4_classical_models_ablation.png', dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================
# FIGURE 5: Lexicon Discriminative Power (Logistic Regression)
# ====================================================================
conditions = ['Random Chance', 'Lexicon Only\n(No Text)', 'TF-IDF Only\n(Full Vocab)']
f1_scores = [0.2500, 0.5268, 0.6845]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(conditions, f1_scores, color=['#ff9999', '#ffcc99', '#99ccff'], edgecolor='black', width=0.6)

ax.set_ylabel('Macro F1-Score')
ax.set_title('Finding 2: Discriminative Power of Lexicon Alone (LR)')
ax.set_ylim(0, 0.8)

# Add exact numbers
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

# Add arrow annotation
ax.annotate('Recovers 77% of\nfull TF-IDF baseline!', xy=(1, 0.53), xytext=(0.5, 0.65),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
            ha='center', fontsize=12, fontweight='bold', color='darkred')

plt.tight_layout()
plt.savefig('../results/fig5_lexicon_only_power.png', dpi=300, bbox_inches='tight')
plt.close()

print("SVM and LR charts generated.")
