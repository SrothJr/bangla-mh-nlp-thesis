import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create results directory if it doesn't exist
os.makedirs('../results', exist_ok=True)

# Set global plotting style for academic papers
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
# FIGURE 1: Lexicon Mining Funnel (Log Scale to show massive reduction)
# ====================================================================
fig, ax = plt.subplots(figsize=(8, 6))
stages = ['Raw Extracted N-Grams', 'Frequency Filter (DF > 5)', 'Statistically Significant\n(q < 0.05, OR > 1)']
counts = [340652, 5051, 91]

# Use a logarithmic scale because 340,652 is so much larger than 91
bars = ax.bar(stages, counts, color=['#b0c4de', '#4682b4', '#191970'], edgecolor='black')
ax.set_yscale('log')
ax.set_ylabel('Number of Candidate Phrases (Log Scale)')
ax.set_title('Lexicon Mining Pipeline: Noise Reduction Funnel')

# Add exact numbers on top of bars
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
    colors = sns.color_palette('pastel')[0:len(cat_counts)]
    
    # Create donut chart
    wedges, texts, autotexts = ax.pie(
        cat_counts, labels=cat_counts.index, autopct='%1.1f%%',
        startangle=140, colors=colors, wedgeprops=dict(width=0.4, edgecolor='white')
    )
    
    plt.setp(autotexts, size=10, weight="bold")
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
rects1 = ax.bar(x - width/2, f1_base, width, label='Base BanglaBERT', color='#d3d3d3', edgecolor='black')
rects2 = ax.bar(x + width/2, f1_lexicon, width, label='BanglaBERT + Lexicon', color='#2e8b57', edgecolor='black')

ax.set_ylabel('F1-Score')
ax.set_title('Impact of Lexicon Integration on Transformer Class Prediction')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right')

# Add F1 scores on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

# Highlight the Class 4 improvement with a subtle arrow or text
ax.annotate('+0.04 Shift', xy=(3.17, 0.80), xytext=(3.17, 0.90),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
            ha='center', fontsize=11, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('../results/fig3_transformer_shift.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figures successfully generated and saved to d:/Sameer/thesis/results/")
