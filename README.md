# Detector de Anomalías — Detección No Supervisada de Comportamiento Anómalo

Modelo no supervisado que identifica jugadores con patrones de comportamiento estadísticamente inusuales mediante Isolation Forest y Autoencoder Keras. Complementa la detección supervisada del Score de Riesgo capturando comportamientos nuevos no etiquetados previamente.

**Estado:** En producción | **Schedule:** 13:00 UTC diariamente (08:00 Lima) | **AUC:** 0.8076

![Gráficos Detector Anomalías](./graficos_anomaly_detection.png)

## Resumen Ejecutivo

El Detector de Anomalías analiza **655,070 jugadores activos** mediante dos algoritmos complementarios:

- **Isolation Forest:** Detecta outliers globales (jugadores estadísticamente lejanos)
- **Autoencoder Keras:** Detecta patrones difíciles de reconstruir (comportamiento inusual)

**Resultados:** AUC combinado **0.8076** | **29,908 jugadores críticos** (4.6%) | **Keras 7x más rápido que PyTorch en CPU**

## Problema y Contexto

El Score de Riesgo supervisado solo detecta patrones etiquetados previamente (4,732 de 655K jugadores). Esto deja déficit crítico:

1. **Patrones nuevos:** Comportamientos anómalos nunca vistos
2. **Falsos negativos:** Jugadores riesgosos sin etiquetar
3. **Evolución dinámica:** Riesgo que emerge recientemente

Solución: Detector no supervisado que identifique desviaciones estadísticas sin depender de etiquetas históricas.

## Análisis Exploratorio de Datos (EDA)

### Distribución de Jugadores por Segmento

| Segmento | Rango Score | Jugadores | Porcentaje | Interpretación |
|----------|-------------|-----------|-----------|-----------------|
| Normal | 0.00-0.70 | 465,129 | 70.9% | Comportamiento típico, sin señales de alerta |
| Moderado | 0.70-0.85 | 102,403 | 15.6% | Comportamiento ligeramente desviado, monitorear |
| Alto | 0.85-0.95 | 60,209 | 9.2% | Comportamiento sospechoso, revisión manual recomendada |
| Crítico | > 0.95 | 29,908 | 4.6% | Comportamiento muy anómalo, acción inmediata |

**Insight:** Concentración en Normal (70.9%) indica que IF+AE son selectivos, reduciendo falsos positivos.

### Análisis de Features

**87 Features por jugador:**
- 82 del mart (`mart_player_features`): apuestas, transacciones, KYC, IPs, bonos
- 3 de velocidad depósito→retiro (timestamp)
- 2 de score supervisado del día

**Período:** Desde 2025-01-01 (datos recientes)

## Experimento: PyTorch vs Keras

### Resultados Comparativos

| Métrica | PyTorch | Keras | Ganador | Razón |
|---------|---------|-------|---------|-------|
| **AUC Combinado** | 0.8046 | **0.8076** | Keras | +0.003 mejor discriminación |
| **Loss Final (AE)** | 332,021 | **220,317** | Keras | -33.6% menor error |
| **Épocas Entrenadas** | 18 | **12** | Keras | Early stopping más rápido |
| **Tiempo AE** | 2.2 min | **0.3 min** | Keras | **7x más rápido en CPU** |
| **Críticos Detectados** | 28,736 | 29,908 | Similar | Cobertura equivalente |

### Por Qué Keras Ganó

1. **Mejor AUC:** 0.8076 vs 0.8046 → discriminación superior
2. **Menor Loss:** 220K vs 332K → mejor reconstrucción
3. **Velocidad:** 0.3 min vs 2.2 min → **7x más rápido**
   - TensorFlow usa BLAS vectorizado para CPU
   - PyTorch overhead mayor sin GPU
4. **Estabilidad:** Converge más rápido

**Decisión:** Keras seleccionado para producción.

## Solución Implementada

### Algoritmos

**Isolation Forest:**
- 300 árboles de aislamiento
- Contamination = 5% (asume top 5% como anómalos)

**Autoencoder Keras:**
- Arquitectura: 87 → 64 → 32 → 16 → 32 → 64 → 87
- Error de reconstrucción alto = patrón inusual
- Early stopping (patience=5)
- Batch size: 2048

**Score Combinado:**
```
combined_score = 0.5 × IF_percentil + 0.5 × AE_percentil
```

## Resultados de Producción

| Métrica | Valor | Interpretación |
|---------|-------|-----------------|
| Modelo | Keras TF 2.17 | Elegido por AUC + velocidad |
| AUC Combinado | 0.7247 | Producción (0.81 en experimento) |
| Jugadores Analizados | 655,070 | 100% base activa |
| **Críticos (> 0.95)** | **28,753** | Requieren revisión inmediata |
| Altos (0.85-0.95) | 65,762 | Revisión manual |
| Moderados (0.70-0.85) | 95,426 | Monitoreo |
| Duración Total | ~2.5 min | Para 655K jugadores |

## Cómo Ejecutarlo

### En Airflow

```bash
airflow dags trigger orse_anomaly_detection
```

### Manualmente

```bash
ssh olimpo-ml
source ~/ml-env/bin/activate
python3 ~/olimpo/anomaly_detection_v3.py
```

### Verificar Resultados

```sql
SELECT COUNT(*) as total,
       SUM(CASE WHEN combined_score > 0.95 THEN 1 ELSE 0 END) as criticos,
       ROUND(AVG(combined_score), 3) as score_promedio
FROM ml_score_risk.anomaly_scores_daily
WHERE run_date = CURRENT_DATE;
```

## Limitaciones

1. Validación manual requerida
2. Contamination fijo (5%)
3. Black-box (sin interpretabilidad)
4. Sensible a cambios abruptos
5. Dependencia del mart

## Próximos Pasos

- Feature importance para explicabilidad
- API real-time (<100ms)
- Validación manual de falsos positivos
- Ensemble con LOF/Elliptic Envelope
- Monitoreo de concept drift
