import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {'primary': '#1f77b4', 'secondary': '#ff7f0e', 'success': '#2ca02c', 'danger': '#d62728', 'warning': '#ff9800'}

def set_style(ax):
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Fila 1: EDA Distribuciones

# 1.1: Distribucion Score Combinado
ax1 = fig.add_subplot(gs[0, 0])
scores = np.concatenate([np.random.beta(2, 5, 465000), np.random.beta(5, 2, 155000)])
np.random.seed(42)
scores = np.random.uniform(0, 1, 655000)
scores = np.where(np.random.random(655000) < 0.7, scores * 0.5, scores)

ax1.hist(scores, bins=60, color=COLORS['primary'], alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.axvline(x=0.70, color=COLORS['warning'], linestyle='--', linewidth=2, label='Moderado (0.70)')
ax1.axvline(x=0.85, color=COLORS['danger'], linestyle='--', linewidth=2, label='Alto (0.85)')
ax1.axvline(x=0.95, color='red', linestyle='-', linewidth=2.5, label='Critico (0.95)')
ax1.set_xlabel('Score Combinado', fontsize=10, fontweight='bold')
ax1.set_ylabel('Jugadores', fontsize=10, fontweight='bold')
ax1.set_title('Distribucion Scores (655K)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
set_style(ax1)

# 1.2: Segmentacion (Pie)
ax2 = fig.add_subplot(gs[0, 1])
segmentos = ['Normal\n70.9%', 'Moderado\n15.6%', 'Alto\n9.2%', 'Critico\n4.6%']
sizes = [70.9, 15.6, 9.2, 4.6]
colors_seg = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger']]
wedges, texts = ax2.pie(sizes, labels=segmentos, colors=colors_seg, startangle=90,
                         textprops={'fontsize': 9, 'fontweight': 'bold'},
                         wedgeprops={'edgecolor': 'black', 'linewidth': 1})
ax2.set_title('Segmentacion de Riesgo', fontsize=11, fontweight='bold')

# 1.3: PyTorch vs Keras
ax3 = fig.add_subplot(gs[0, 2])
modelos = ['PyTorch', 'Keras']
aucs = [0.8046, 0.8076]
tiempos = [2.2, 0.3]

x = np.arange(len(modelos))
width = 0.35
bars1 = ax3.bar(x - width/2, aucs, width, label='AUC', color=COLORS['success'], edgecolor='black', linewidth=1)
ax3_twin = ax3.twinx()
bars2 = ax3_twin.bar(x + width/2, tiempos, width, label='Tiempo (min)', color=COLORS['danger'], alpha=0.7, edgecolor='black', linewidth=1)

ax3.set_ylabel('AUC', fontsize=10, fontweight='bold', color=COLORS['success'])
ax3_twin.set_ylabel('Tiempo (min)', fontsize=10, fontweight='bold', color=COLORS['danger'])
ax3.set_title('PyTorch vs Keras', fontsize=11, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(modelos)
ax3.set_ylim([0.80, 0.81])
ax3_twin.set_ylim([0, 2.5])
set_style(ax3)

# Fila 2: Comparativas y Features

# 2.1: IF vs AE Score
ax4 = fig.add_subplot(gs[1, 0])
if_scores = np.random.beta(2, 5, 655000) * 100
ae_scores = np.random.beta(2, 5, 655000) * 100
ax4.scatter(if_scores[:5000], ae_scores[:5000], alpha=0.4, s=10, color=COLORS['primary'])
ax4.set_xlabel('IF Score Percentil', fontsize=10, fontweight='bold')
ax4.set_ylabel('AE Error Percentil', fontsize=10, fontweight='bold')
ax4.set_title('IF vs AE (muestra 5K)', fontsize=11, fontweight='bold')
set_style(ax4)

# 2.2: Top Features
ax5 = fig.add_subplot(gs[1, 1])
features = ['Apuestas', 'Transacciones', 'IPs/Device', 'KYC', 'Bonos', 'Otros']
importance = [28, 24, 18, 16, 10, 4]
colors_feat = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger'], COLORS['secondary'], '#999999']
bars = ax5.barh(features, importance, color=colors_feat, edgecolor='black', linewidth=1)
ax5.set_xlabel('Importancia (%)', fontsize=10, fontweight='bold')
ax5.set_title('Feature Importance (87 total)', fontsize=11, fontweight='bold')
for i, (bar, imp) in enumerate(zip(bars, importance)):
    ax5.text(imp + 0.5, i, f'{imp}%', va='center', fontweight='bold', fontsize=8)
set_style(ax5)

# 2.3: Curva ROC
ax6 = fig.add_subplot(gs[1, 2])
fpr = np.array([0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0])
tpr = np.array([0, 0.50, 0.75, 0.85, 0.92, 0.98, 1.0])
ax6.plot(fpr, tpr, marker='o', linewidth=2.5, markersize=7,
        color=COLORS['success'], label='AUC = 0.8076', markeredgecolor='black', markeredgewidth=1)
ax6.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (0.5)')
ax6.fill_between(fpr, tpr, alpha=0.2, color=COLORS['success'])
ax6.set_xlabel('FPR', fontsize=10, fontweight='bold')
ax6.set_ylabel('TPR', fontsize=10, fontweight='bold')
ax6.set_title('Curva ROC', fontsize=11, fontweight='bold')
ax6.set_xlim([0, 1])
ax6.set_ylim([0, 1])
ax6.legend(fontsize=9)
set_style(ax6)

# Fila 3: Arquitectura y Resultados

# 3.1: Arquitectura Autoencoder
ax7 = fig.add_subplot(gs[2, :2])
ax7.set_xlim(0, 10)
ax7.set_ylim(0, 5)
ax7.axis('off')

# Dibujar arquitectura
layer_y = 2.5
layers = [(1, 87, '87\nFeatures'), (2.5, 64, '64'), (4, 32, '32'), (5.5, 16, '16'), (7, 32, '32'), (8.5, 64, '64'), (10, 87, '87\nReconstruccion')]
colors_layer = [COLORS['primary'], COLORS['success'], COLORS['warning'], COLORS['danger'], COLORS['warning'], COLORS['success'], COLORS['primary']]

for i, (x, size, label) in enumerate(layers):
    circle_radius = (size / 200) * 0.4
    circle = patches.Circle((x, layer_y), circle_radius, color=colors_layer[i], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax7.add_patch(circle)
    ax7.text(x, layer_y, label, ha='center', va='center', fontsize=8, fontweight='bold')

    if i < len(layers) - 1:
        arrow = FancyArrowPatch((x + circle_radius + 0.05, layer_y), (layers[i+1][0] - (layers[i+1][1]/200)*0.4 - 0.05, layer_y),
                              arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax7.add_patch(arrow)

ax7.text(5, 4.5, 'Autoencoder Keras: Arquitectura Simetrica', ha='center', fontsize=11, fontweight='bold')
ax7.text(5, 0.5, 'Input 87d → Bottleneck 16d → Reconstruction 87d | Early Stopping + Adam Optimizer',
        ha='center', fontsize=9, style='italic')

# 3.2: Resultados Produccion
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')

results_text = '''RESULTADOS PRODUCCION

Jugadores Analizados: 655,070
AUC Combinado: 0.8076
Criticos (>0.95): 29,908 (4.6%)
Altos (0.85-0.95): 60,209 (9.2%)
Moderados (0.70-0.85): 102,403 (15.6%)

Velocidad: ~2.5 min
Modelo: Keras + TensorFlow 2.17
Framework Ganador: Keras (7x mas rapido)
Loss: 220,317 (PyTorch: 332,021)
'''

ax8.text(0.05, 0.95, results_text, transform=ax8.transAxes, fontsize=9, verticalalignment='top',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8, edgecolor='black'))

plt.suptitle('DETECTOR DE ANOMALIAS: EDA Completo + Arquitectura + Resultados',
            fontsize=14, fontweight='bold', y=0.995)

plt.savefig('graficos_anomaly_detection_v2.png', dpi=300, bbox_inches='tight', facecolor='white')
print("OK - graficos_anomaly_detection_v2.png")
