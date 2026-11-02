# Detección de Anomalías en Telemetría Satelital mediante Inteligencia Computacional

**Trabajo Final - Inteligencia Computacional (4.° Año)**  
**Integrantes:** Agustín Barbero, Conrado Ciorciari  

---

## 🛰️ 1. Definición del Problema y Motivación

La supervisión de satélites en órbita depende de flujos continuos de telemetría (temperaturas, voltajes, corrientes, presiones). Tradicionalmente, los centros de control utilizan umbrales estáticos prefijados (*Out-of-Limits* u OOL) para alertar desvíos.

Sin embargo, este enfoque presenta limitaciones operativas críticas:
1. **Anomalías dentro de rango:** La mayoría de las fallas incipientes no superan los límites absolutos (e.g., congelamiento de señal, ruido de alta frecuencia o derivas lentas).
2. **Dependencias multivariadas y contexto:** Valores normales bajo ciertas condiciones (e.g., baja corriente de paneles solares durante un eclipse) son anómalos bajo otras (luz solar directa).
3. **Escala temporal masiva:** Años de misión continua vuelven inviable la inspección manual continua.

### Objetivo
Desarrollar un sistema de detección temprana de anomalías en telemetría satelital mediante **aprendizaje semi-supervisado**. El modelo aprende la dinámica normal de vuelo y alerta ante desvíos imprevistos, combinando predicción temporal con proyección bidimensional para diagnóstico visual.

---

## 📊 2. Dataset: ESA Anomaly Benchmark (Misión 1)

Se utiliza el **ESA Anomaly Dataset** (Agencia Espacial Europea, Airbus Defence and Space y KP Labs), que contiene telemetría real de vuelo (2000–2013) con muestreos cada ~90 segundos y fallas certificadas por ingenieros de vuelo en `labels.csv`.

### Subconjunto Seleccionado
Para garantizar viabilidad computacional y rigor experimental, el estudio se enfoca en el **Subsistema 5 (Grupo funcional 8)** en el intervalo **2004–2005**:
* **Canal Objetivo Principal (Target):** `channel_41` (cuenta con más de 200 anomalías reales documentadas).
* **Canales de Soporte Multivariado:** `channel_42` y `channel_43` (mismo subsistema físico y dinámica correlacionada).
* **Canal de Contexto Global:** `channel_1` (monitoreo del estado general de la nave / bus principal).

---

## 🛠️ 3. Metodología Paso a Paso

El pipeline se estructura en 6 etapas secuenciales:

```
┌──────────────────────────────────────────────────────────────┐
│ PASO 1: Preprocesamiento y Ventanas Deslizantes              │
│ - Normalización y partición cronológica (Train/Val/Test)     │
│ - Creación de secuencias [t-L ... t-1] -> [t] (L = 50)       │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ PASO 2: Aprendizaje de la Normalidad (LSTM)                  │
│ - Entrenamiento en régimen sano minimizando error MSE        │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ PASO 3: Predicción y Cálculo del Error                       │
│ - Predicción paso a paso sobre datos de prueba               │
│ - Señal de error residual instantáneo: e(t) = |y(t) - ŷ(t)|  │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ PASO 4: Filtrado del Error y Criterio de Detección           │
│ - Suavizado de ruido de alta frecuencia mediante EWMA        │
│ - Umbral estadístico de decisión: Umbral = μ₀ + k·σ₀         │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ PASO 5: Visualización de Estados con SOM (2D)                │
│ - Agrupamiento no supervisado mediante red de Kohonen        │
│ - Diagnóstico visual de trayectorias (régimen sano vs falla) │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ PASO 6: Validación y Métricas de Desempeño                   │
│ - Contraste contra eventos etiquetados (labels.csv)          │
│ - Matriz de confusión, Precisión, Sensibilidad y F1-Score    │
└──────────────────────────────────────────────────────────────┘
```

1. **Preprocesamiento:** Escalado $[0, 1]$, partición temporal estricta (entrenamiento sobre período sano, evaluación sobre período con anomalías) y generación de ventanas temporales ($L=50$).
2. **Modelado Recurrente (LSTM):** Red dinámica que captura dependencias temporales y la física subyacente del satélite en régimen nominal.
3. **Error Residual:** Cálculo de la discrepancia paso a paso $e_t = |y_t - \hat{y}_t|$. En condiciones normales $e_t \approx 0$; ante una falla la red no puede predecir el comportamiento y el residuo se eleva.
4. **Filtrado EWMA y Umbral:** Atenuación de picos de ruido instrumental mediante media móvil exponencial ($s_t = \lambda e_t + (1-\lambda)s_{t-1}$) y umbral calibrado $\mu_0 + k\sigma_0$ sobre el período normal.
5. **Mapeo Topológico (SOM):** Proyección bidimensional de patrones multivariados y residuos para diagnóstico visual del estado operativo de la nave.
6. **Evaluación Experimental:** Comparación con las marcas temporales reales (`labels.csv`), generando métricas (Precisión, Recall, $F_1$) y gráficos sincronizados.

---

## 📚 4. Bibliografía de Referencia

* **Dataset Oficial:** Kotowski et al. (2024). *European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry (ESA-ADB)*. NeurIPS / [arXiv:2406.17826](https://arxiv.org/abs/2406.17826).
* **Arquitectura LSTM de Referencia:** Hundman et al. (NASA JPL, 2018). *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding*. ACM KDD / [arXiv:1802.04431](https://arxiv.org/abs/1802.04431).
* **Filtrado EWMA y Detección en Streaming:** Carter & Streilein (MIT Lincoln Lab, 2012). *Probabilistic reasoning for streaming anomaly detection*. IEEE SSP / [arXiv:1905.02559](https://arxiv.org/abs/1905.02559).
* **Mapas Auto-organizativos (SOM):** Kohonen, T. (1990). *The Self-Organizing Map*. Proceedings of the IEEE / [Stanford University](https://web.stanford.edu/class/ee373a/selforganizingmap.pdf).
* **Métricas en Series Temporales:** Tatbul et al. (Intel Labs / MIT, 2018). *Precision and Recall for Time Series*. [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2018/file/8f468c873a32bb0619eaeb2050ba45d1-Paper.pdf).
