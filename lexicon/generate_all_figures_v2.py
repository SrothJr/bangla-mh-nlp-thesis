import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create results directory if it doesn't exist
os.makedirs('../results', exist_ok=True)

# Set global plotting style for professional academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2
})

# Define a consistent, professional academic color palette
C_BASE = '#95A5A6'      # Grey for baselines
C_MID = '#5DADE2'       # Light blue for intermediate steps
C_FINAL = '#2874A6'     # Dark blue for final/best steps
C_ACCENT = '#E74C3C'    # Muted red for random chance/warnings
C_WARN = '#F39C12'      # Muted orange for intermediate/lexicon only

# ====================================================================
# FIGURE 1: Lexicon Mining Funnel
# ====================================================================
fig, ax = plt.subplots(figsize=(8, 6))
stages = ['Raw Extracted N-Grams', 'Frequency Filter (DF > 5)', 'Statistically Significant\n(q < 0.05, OR > 1)']
counts = [340652, 5051, 91]

bars = ax.bar(stages, counts, color=[C_BASE, C_MID, C_FINAL], edgecolor='black')
ax.set_yscale('log')
ax.set_ylabel('Number of Candidate Phrases (Log Scale)')
ax.set_title('Lexicon Mining Pipeline: Noise Reduction Funnel')

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval * 1.2, f'{int(yval):,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/fig1_mining_funnel.png', dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================
# FIGURE 2: Lexicon Category Distribution (Donut Chart)
# ====================================================================
try:
    df = pd.read_excel('../Bangla_Depression_Lexicon_latest.xlsx')
    cat_counts = df['category_name'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    # Use a professional sequential blue palette
    colors = sns.color_palette("Blues_r", len(cat_counts) + 2)[:len(cat_counts)]
    
    # Create donut chart with thicker wedges so numbers fit inside
    wedges, texts, autotexts = ax.pie(
        cat_counts, labels=cat_counts.index, autopct='%1.1f%%',
        startangle=140, colors=colors, pctdistance=0.75, 
        wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2)
    )
    
    # Ensure text is readable
    plt.setp(autotexts, size=11, weight="bold", color="white")
    # For lighter slices, make the text black for contrast
    for i, autotext in enumerate(autotexts):
        if i > (len(autotexts) // 2):
            autotext.set_color('black')

    ax.set_title('Distribution of Mined Lexicon Terms by Clinical Category (n=91)')
    
    plt.tight_layout()
    plt.savefig('../results/fig2_category_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Skipping Figure 2 due to file error: {e}")

# ====================================================================
# FIGURE 3: BanglaBERT Performance Shift (Grouped Bar Chart)
# ====================================================================
classes = ['Class 1 (Control)', 'Class 2 (Mild)', 'Class 3 (Moderate)', 'Class 4 (Severe)']
f1_base = [0.86, 0.81, 0.54, 0.75]
f1_lexicon = [0.85, 0.79, 0.56, 0.79]

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, f1_base, width, label='Base BanglaBERT', color=C_BASE, edgecolor='black')
rects2 = ax.bar(x + width/2, f1_lexicon, width, label='BanglaBERT + Lexicon', color=C_FINAL, edgecolor='black')

ax.set_ylabel('F1-Score')
ax.set_title('Impact of Lexicon Integration on Transformer Class Prediction')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

# Removed arrow annotation per user request

plt.tight_layout()
plt.savefig('../results/fig3_transformer_shift.png', dpi=300, bbox_inches='tight')
plt.close()

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

rects1 = ax.bar(x - width, tfidf_only, width, label='TF-IDF Only (Baseline)', color=C_BASE, edgecolor='black')
rects2 = ax.bar(x, exact_lex, width, label='+ Exact Lexicon', color=C_MID, edgecolor='black')
rects3 = ax.bar(x + width, semantic_lex, width, label='+ Semantic Lexicon', color=C_FINAL, edgecolor='black')

ax.set_ylabel('Macro F1-Score')
ax.set_title('Lexicon Ablation Impact Across Classical Models')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0.60, 0.75)
ax.legend(loc='upper left')

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
bars = ax.bar(conditions, f1_scores, color=[C_ACCENT, C_WARN, C_BASE], edgecolor='black', width=0.6)

ax.set_ylabel('Macro F1-Score')
ax.set_title('Discriminative Power of Lexicon Alone (LR)')
ax.set_ylim(0, 0.8)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

# Removed arrow annotation per user request

plt.tight_layout()
plt.savefig('../results/fig5_lexicon_only_power.png', dpi=300, bbox_inches='tight')
plt.close()

print("All updated, professional figures generated successfully.")
