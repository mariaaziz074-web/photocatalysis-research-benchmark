"""
Generate publication-ready multi-panel figures for Scientific Data manuscript.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import matplotlib.pyplot as plt

# Set publication style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# Load data
df_bench = pd.read_csv(PROJECT_ROOT / "results/tables/benchmark_results.csv")

# Color scheme
VIBRANT_CATEGORICAL = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

# ========================================
# FIGURE 3: Performance Comparison (3 panels)
# ========================================

fig = plt.figure(figsize=(18, 5))

# Panel A: RMSE Comparison
ax1 = plt.subplot(1, 3, 1)
pivot_rmse = df_bench.pivot(index='split_strategy', columns='model', values='RMSE')
pivot_rmse.plot(kind='bar', ax=ax1, width=0.75, color=VIBRANT_CATEGORICAL, 
                edgecolor='white', linewidth=1.5)
ax1.set_title('(A) Root Mean Squared Error', fontweight='bold', fontsize=13)
ax1.set_xlabel('Split Strategy', fontsize=11, fontweight='bold')
ax1.set_ylabel('RMSE (%)', fontsize=11, fontweight='bold')
ax1.legend(title='Model', frameon=False, loc='upper left', fontsize=9)
ax1.set_xticklabels(['Catalyst\nHoldout', 'Dye\nHoldout', 'Paper\nHoldout', 
                     'Random', 'Time-Based'], rotation=0, ha='center', fontsize=9)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_ylim([0, 16])

# Panel B: R² Comparison
ax2 = plt.subplot(1, 3, 2)
pivot_r2 = df_bench.pivot(index='split_strategy', columns='model', values='R2')
pivot_r2.plot(kind='bar', ax=ax2, width=0.75, color=VIBRANT_CATEGORICAL,
              edgecolor='white', linewidth=1.5)
ax2.set_title('(B) Coefficient of Determination', fontweight='bold', fontsize=13)
ax2.set_xlabel('Split Strategy', fontsize=11, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax2.legend(title='Model', frameon=False, loc='lower left', fontsize=9)
ax2.set_xticklabels(['Catalyst\nHoldout', 'Dye\nHoldout', 'Paper\nHoldout', 
                     'Random', 'Time-Based'], rotation=0, ha='center', fontsize=9)
ax2.set_ylim([0.70, 1.0])
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

# Panel C: Conformal Coverage
ax3 = plt.subplot(1, 3, 3)
pivot_cov = df_bench.pivot(index='split_strategy', columns='model', values='conformal_coverage')
pivot_cov.plot(kind='bar', ax=ax3, width=0.75, color=VIBRANT_CATEGORICAL,
               edgecolor='white', linewidth=1.5)
ax3.axhline(y=0.90, color='#E74C3C', linestyle='--', linewidth=2.5, 
            label='Target 90%', zorder=10)
ax3.set_title('(C) Conformal Prediction Coverage', fontweight='bold', fontsize=13)
ax3.set_xlabel('Split Strategy', fontsize=11, fontweight='bold')
ax3.set_ylabel('Empirical Coverage', fontsize=11, fontweight='bold')
ax3.legend(title='Model', frameon=False, loc='lower left', fontsize=9)
ax3.set_xticklabels(['Catalyst\nHoldout', 'Dye\nHoldout', 'Paper\nHoldout', 
                     'Random', 'Time-Based'], rotation=0, ha='center', fontsize=9)
ax3.set_ylim([0.4, 1.0])
ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "results/figures/fig3_performance_comparison.png", 
            dpi=300, bbox_inches='tight')
plt.savefig(PROJECT_ROOT / "results/figures/fig3_performance_comparison.pdf", 
            bbox_inches='tight')
plt.close()

print("✓ Figure 3 saved: fig3_performance_comparison.png/pdf")

# ========================================
# SUPPLEMENTARY TABLE S2: Split Characteristics
# ========================================

split_info = {
    'random': {'train': 415, 'test': 104, 'description': '80/20 random split'},
    'catalyst_holdout': {'train': 440, 'test': 79, 'description': 'Held out: Au/TiO2, F-doped'},
    'dye_holdout': {'train': 415, 'test': 104, 'description': 'Held out: Direct Blue 15, Rhodamine B'},
    'paper_holdout': {'train': 419, 'test': 100, 'description': '10 papers held out'},
    'time_based': {'train': 415, 'test': 104, 'description': 'Chronological split'}
}

split_stats = []
for split in ['random', 'catalyst_holdout', 'dye_holdout', 'paper_holdout', 'time_based']:
    split_data = df_bench[df_bench['split_strategy'] == split]
    
    split_stats.append({
        'Split Strategy': split,
        'Train Size': split_info[split]['train'],
        'Test Size': split_info[split]['test'],
        'Description': split_info[split]['description'],
        'Avg RMSE': f"{split_data['RMSE'].mean():.2f}",
        'Avg R2': f"{split_data['R2'].mean():.3f}",
        'Best Model': split_data.loc[split_data['R2'].idxmax(), 'model']
    })

df_split_stats = pd.DataFrame(split_stats)
df_split_stats.to_csv(PROJECT_ROOT / "results/tables/table_s2_split_characteristics.csv", index=False)
print("✓ Table S2 saved: table_s2_split_characteristics.csv")

print("\n" + "="*70)
print("PUBLICATION FIGURES COMPLETE")
print("="*70)
print("\nManuscript-ready figures:")
print("  - fig1_dataset_overview.png (existing)")
print("  - fig2_chemical_space.png (existing)")
print("  - fig3_performance_comparison.png (NEW)")
print("  - shap_summary_XGBoost.png (existing)")
print("  - shap_bar_XGBoost.png (existing)")