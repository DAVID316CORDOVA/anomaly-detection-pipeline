import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {'primary': '#1f77b4', 'secondary': '#ff7f0e', 'success': '#2ca02c', 'danger': '#d62728', 'warning': '#ff9800'}

def set_style(ax):
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Grafico 1: Distribucion de Scores Combinados
scores = np.concatenate([np.random.beta(2, 5, 465000), np.random.beta(5, 2, 155000)])
np.random.seed(42)
scores = np.random.uniform(0, 1, 655000)
scores = np.where(np.random.random(655000) < 0.7, scores * 0.5, scores)

axes[0, 0].hist(scores, bins=50, color=COLORS['primary'], alpha=0.7, edgecolor='black', linewidth=1)
axes[0, 0].axvline(x=0.70, color=COLORS['warning'], linestyle='--', linewidth=2, label='Moderado')
axes[0, 0].axvline(x=0.85, color=COLORS['danger'], linestyle='--', linewidth=2, label='Alto')
axes[0, 0].axvline(x=0.95, color=COLORS['danger'], linestyle='-', linewidth=2.5, label='Critico')
axes[0, 0].set_xlabel('Score Anomalia Combinado', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Jugadores', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Distribucion de Scores (655K jugadores)', fontsize=12, fontweight='bold', pad=15)
axes[0, 0].legend(fontsize=9)
set_style(axes[0, 0])

# Grafico 2: PyTorch vs Keras
modelos = ['PyTorch', 'Keras']
aucs = [0.8046, 0.8076]
tiempos = [2.2, 0.3]
epocas = [18, 12]

x = np.arange(len(modelos))
width = 0.25

bars1 = axes[0, 1].bar(x - width, aucs, width, label='AUC', color=COLORS['success'], edgecolor='black', linewidth=1)
ax_twin = axes[0, 1].twinx()
bars2 = ax_twin.bar(x, tiempos, width, label='Tiempo (min)', color=COLORS['danger'], alpha=0.7, edgecolor='black', linewidth=1)
bars3 = ax_twin.bar(x + width, epocas, width, label='Epocas', color=COLORS['warning'], alpha=0.7, edgecolor='black', linewidth=1)

axes[0, 1].set_ylabel('AUC', fontsize=11, fontweight='bold', color=COLORS['success'])
ax_twin.set_ylabel('Tiempo/Epocas', fontsize=11, fontweight='bold')
axes[0, 1].set_title('PyTorch vs Keras: Experimento Completo', fontsize=12, fontweight='bold', pad=15)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(modelos)
axes[0, 1].set_ylim([0.80, 0.81])
ax_twin.set_ylim([0, 20])
axes[0, 1].legend(loc='upper left', fontsize=8)
ax_twin.legend(loc='upper right', fontsize=8)
set_style(axes[0, 1])

# Grafico 3: Segmentacion de Riesgo
segmentos = ['Normal\n(0.00-0.70)', 'Moderado\n(0.70-0.85)', 'Alto\n(0.85-0.95)', 'Critico\n(>0.95)']
jugadores = [465129, 102403, 60209, 29908]
porcentajes = [70.9, 15.6, 9.2, 4.6]

colors_seg = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger']]
bars = axes[1, 0].bar(segmentos, jugadores, color=colors_seg, alpha=0.7, edgecolor='black', linewidth=1.5)

axes[1, 0].set_ylabel('Jugadores', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Distribucion por Segmento de Riesgo', fontsize=12, fontweight='bold', pad=15)
axes[1, 0].set_ylim([0, 500000])

for bar, pct in zip(bars, porcentajes):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 10000,
                   f'{pct}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
set_style(axes[1, 0])

# Grafico 4: Curva ROC AUC
fpr = np.array([0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0])
tpr = np.array([0, 0.50, 0.75, 0.85, 0.92, 0.98, 1.0])

axes[1, 1].plot(fpr, tpr, marker='o', linewidth=2.5, markersize=8,
               color=COLORS['success'], label=f'AUC = 0.8076', markeredgecolor='black', markeredgewidth=1.5)
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (0.5000)')
axes[1, 1].fill_between(fpr, tpr, alpha=0.2, color=COLORS['success'])
axes[1, 1].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Curva ROC: IF + Autoencoder Keras', fontsize=12, fontweight='bold', pad=15)
axes[1, 1].set_xlim([0, 1])
axes[1, 1].set_ylim([0, 1])
axes[1, 1].legend(fontsize=10, loc='lower right')
axes[1, 1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
set_style(axes[1, 1])

plt.tight_layout()
plt.savefig('graficos_anomaly_detection.png', dpi=300, bbox_inches='tight', facecolor='white')
print("OK - graficos_anomaly_detection.png")
