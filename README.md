# Detección de Anomalías en Telemetría Satelital mediante Inteligencia Computacional

**Trabajo Final - Inteligencia Computacional**  
**Integrantes:** Agustín Barbero, Conrado Ciorciari  

---

## Idea del Proyecto

La supervisión de satélites en órbita depende de flujos continuos de datos provenientes de múltiples sensores (temperaturas, corrientes, voltajes). El método convencional de control utiliza límites fijos (*out-of-limits*), pero resulta insuficiente ante fallas sutiles que ocurren dentro del rango normal de operación, o ante situaciones donde un valor solo es anómalo según el contexto temporal y multivariado de la nave.

La propuesta consiste en desarrollar un sistema de detección temprana de anomalías utilizando un enfoque semi-supervisado. El modelo aprende la dinámica normal de vuelo del satélite durante períodos sin fallas. Cuando ocurre una degradación o rotura de componentes, el sistema detecta que la señal real se aparta del comportamiento predicho y emite una alerta, complementando la detección con una proyección bidimensional para inspeccionar visualmente el estado del subsistema.

Para validar el sistema se utiliza el **ESA Anomaly Dataset** (telemetría real de misiones de la Agencia Espacial Europea), focalizando el análisis en un subsistema crítico (Subsistema 5, `channel_41` como variable objetivo, junto a canales correlacionados de soporte y contexto global).

---

## Flujo Metodológico

1. **Preprocesamiento y ventanas temporales:** Limpieza, normalización y generación de secuencias temporales para alimentar modelos secuenciales.
2. **Aprendizaje de la normalidad (Red Dinámica / LSTM):** Entrenamiento de una red recurrente únicamente sobre períodos nominales para capturar la dinámica temporal habitual.
3. **Predicción y cálculo del error residual:** Generación de la predicción un paso hacia adelante y cálculo de la discrepancia instantánea frente a la medición real.
4. **Filtrado EWMA y umbral estadístico:** Suavizado del error residual mediante una media móvil ponderada exponencialmente para atenuar ruido de alta frecuencia, aplicando un umbral de decisión basado en la dispersión del régimen sano.
5. **Visualización de estados con SOM (2D):** Mapeo topológico mediante una red de Kohonen para agrupar estados normales y visualizar la dispersión hacia regiones periféricas ante condiciones de falla.
6. **Evaluación de resultados:** Contraste de las alarmas detectadas contra los eventos catalogados por ingenieros de vuelo en `labels.csv` (matriz de confusión, precisión, sensibilidad y F1-score).

