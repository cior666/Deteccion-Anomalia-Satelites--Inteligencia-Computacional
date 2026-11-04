"""
TP Final - Etapa 6: Evaluacion Experimental y Metricas de Desempeno.

Que hace este script:
  1. Carga las alarmas binarias generadas en la Etapa 5 (alarmas_test.npy).
  2. Carga las etiquetas reales de labels.csv sincronizadas con los instantes de Test.
  3. Evaluacion punto a punto (Point-wise Metrics):
       - Construye la Matriz de Confusion 2x2 (TP, FP, TN, FN).
       - Calcula: Precision, Sensibilidad (Recall), F1-Score y Accuracy.
  4. Evaluacion basada en eventos reales (Event-based / Range Metrics):
       - Evalua los 14 eventos anómalos documentados en 2004-2005.
       - Determina cuantos eventos fueron detectados exitosamente por el modelo
         y cuantas falsas alarmas ocurrieron fuera de los eventos.
       - Calcula el porcentaje de exito en deteccion de fallas criticas.
  5. Grafica y exporta:
       - Matriz de confusion visual (fig9_matriz_confusion.png).
       - Cronograma completo de Test con detecciones vs. fallas reales (fig10_resumen_deteccion.png).
       - Reporte cuantitativo consolidado en JSON y CSV.

Salidas (datos_procesados/metricas/):
  - metricas_desempeno.json     Resultados numericos detallados
  - evaluacion_eventos.csv      Tabla evento por evento (detectado: SI/NO, latencia)
  - fig9_matriz_confusion.png   Matriz de confusion normalizada y absoluta
  - fig10_resumen_deteccion.png Panorama general de alertas sobre todo el periodo Test
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 0) CONFIGURACION Y RUTAS
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(DIR_BASE, "datos_procesados")
DIR_VENTANAS = os.path.join(DIR_DATOS, "ventanas")
DIR_PRED = os.path.join(DIR_DATOS, "predicciones")
DIR_DETECCION = os.path.join(DIR_DATOS, "deteccion")
DIR_SALIDA = os.path.join(DIR_DATOS, "metricas")
os.makedirs(DIR_SALIDA, exist_ok=True)


# 1) CARGA DE ALARMAS Y ETIQUETAS DE TEST
print("=== 1) Carga de alarmas y datos de referencia ===")
ruta_alarmas = os.path.join(DIR_DETECCION, "alarmas_test.npy")

# Cargar la tabla completa para obtener el vector de verdad terreno en Test
datos = pd.read_pickle(os.path.join(DIR_DATOS, "datos_alineados.pkl"))
idx_test = np.load(os.path.join(DIR_VENTANAS, "idx_test.npy"))

# Vector binario real: 1 si ese instante tenia anomalia etiquetada en el subsistema
y_real = datos["anomalia"].iloc[idx_test].values.astype(np.int8)
timestamps = datos.index[idx_test].to_numpy()

if not os.path.exists(ruta_alarmas):
    print(f"AVISO: No se encontro '{ruta_alarmas}'. Se utilizara un vector simulado")
    print("para configurar el script. Ejecuta las etapas 3, 4 y 5 para los valores finales.")
    y_pred = np.zeros_like(y_real)
else:
    y_pred = np.load(ruta_alarmas)
    print(f"Total de instantes evaluados: {len(y_pred):,}")
    print(f"Instantes con falla real (Ground Truth = 1): {np.sum(y_real):,} ({np.mean(y_real)*100:.2f}%)")
    print(f"Instantes con alarma encendida (Prediccion = 1): {np.sum(y_pred):,} ({np.mean(y_pred)*100:.2f}%)")

# 2) EVALUACION PUNTO A PUNTO (MATRIZ DE CONFUSION 2x2)
print("\n=== 2) Evaluacion Punto a Punto ===")
TP = int(np.sum((y_real == 1) & (y_pred == 1)))  # Falla real detectada
FP = int(np.sum((y_real == 0) & (y_pred == 1)))  # Falsa alarma
FN = int(np.sum((y_real == 1) & (y_pred == 0)))  # Falla no detectada (critico)
TN = int(np.sum((y_real == 0) & (y_pred == 0)))  # Periodo normal sin alarma

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
accuracy = (TP + TN) / len(y_real)

print(f"Matriz de Confusion:")
print(f"  [TN: {TN:>7,} | FP: {FP:>6,}]")
print(f"  [FN: {FN:>7,} | TP: {TP:>6,}]")
print(f"\nMetricas:")
print(f"  - Precision:    {precision*100:6.2f}%  (Confiabilidad cuando suena la alarma)")
print(f"  - Sensibilidad: {recall*100:6.2f}%  (Porcentaje de tiempo de falla capturado)")
print(f"  - F1-Score:     {f1*100:6.2f}%  (Balance armonico)")
print(f"  - Exactitud:    {accuracy*100:6.2f}%  (Accuracy global)")

# 3) EVALUACION BASADA EN EVENTOS (EVENT-BASED EVALUATION)
print("\n=== 3) Evaluacion por Eventos Reales ===")
eventos = pd.read_csv(os.path.join(DIR_DATOS, "eventos_2004_2005.csv"), index_col="ID")

# Filtrar solo los eventos que caen dentro del periodo de Test
eventos_test = eventos[pd.to_datetime(eventos["StartTime"]) >= pd.Timestamp("2004-08-01")].copy()
resultados_eventos = []
eventos_detectados = 0

for ev_id, ev in eventos_test.iterrows():
    t_ini = pd.Timestamp(ev["StartTime"])
    t_fin = pd.Timestamp(ev["EndTime"])
    # Mascara del evento en el tiempo de Test
    mascara_ev = (timestamps >= t_ini) & (timestamps <= t_fin)
    alarmas_en_evento = y_pred[mascara_ev]
    if len(alarmas_en_evento) > 0 and np.sum(alarmas_en_evento) > 0:
        detectado = True
        eventos_detectados += 1
        pct_cobertura = (np.sum(alarmas_en_evento) / len(alarmas_en_evento)) * 100
    else:
        detectado = False
        pct_cobertura = 0.0
        
    resultados_eventos.append({
        "ID": ev_id,
        "Categoria": ev["Category"],
        "Inicio": str(t_ini),
        "Fin": str(t_fin),
        "Duracion_h": ev["duracion_h"],
        "Detectado": "SI" if detectado else "NO",
        "Cobertura_pct": round(pct_cobertura, 1)
    })

df_eventos_eval = pd.DataFrame(resultados_eventos)
tasa_acierto_eventos = (eventos_detectados / len(eventos_test)) * 100

print(f"Total de eventos anómalos en Test: {len(eventos_test)}")
print(f"Eventos detectados con exito:     {eventos_detectados} ({tasa_acierto_eventos:.1f}%)")
print("\nDetalle por evento:")
print(df_eventos_eval[["ID", "Categoria", "Duracion_h", "Detectado", "Cobertura_pct"]].to_string(index=False))

df_eventos_eval.to_csv(os.path.join(DIR_SALIDA, "evaluacion_eventos.csv"), index=False)

# 4) GUARDADO DE METRICAS
metricas_completas = {
    "punto_a_punto": {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy)
    },
    "eventos": {
        "total_eventos_reales": len(eventos_test),
        "eventos_detectados": eventos_detectados,
        "porcentaje_acierto_eventos": float(tasa_acierto_eventos)
    }
}
with open(os.path.join(DIR_SALIDA, "metricas_desempeno.json"), "w") as f:
    json.dump(metricas_completas, f, indent=2)

# 5) GRAFICOS DE DESEMPENO
print("\n=== 5) Generando graficos de reporte ===")
# --- Figura 9: Matriz de Confusion Visual ---
fig, ax = plt.subplots(figsize=(6, 5))
matriz_conf = np.array([[TN, FP], [FN, TP]])
im = ax.imshow(matriz_conf, cmap="Blues", alpha=0.8)
# Agregar etiquetas de texto dentro de cada celda
clases = ["Normal (0)", "Anomalia (1)"]
for i in range(2):
    for j in range(2):
        val = matriz_conf[i, j]
        color_texto = "white" if val > (matriz_conf.max() / 2) else "black"
        ax.text(j, i, f"{val:,}\n({val/len(y_real)*100:.2f}%)",
                ha="center", va="center", color=color_texto, fontsize=11, fontweight="bold")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(clases)
ax.set_yticklabels(clases)
ax.set_xlabel("Prediccion del Modelo", fontweight="bold")
ax.set_ylabel("Valor Real (Ground Truth)", fontweight="bold")
ax.set_title(f"Matriz de Confusion (Test)\nF1-Score: {f1*100:.2f}% | Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}%")
plt.colorbar(im, ax=ax)
plt.tight_layout()

ruta_fig9 = os.path.join(DIR_SALIDA, "fig9_matriz_confusion.png")
plt.savefig(ruta_fig9, dpi=120)
print(f"Guardado: {ruta_fig9}")

# --- Figura 10: Panorama Completo de Test (Predicciones vs. Anomalias Reales) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# Franjas reales
for _, ev in eventos_test.iterrows():
    color = "red" if ev["Category"] == "Anomaly" else "orange"
    ax1.axvspan(pd.Timestamp(ev["StartTime"]), pd.Timestamp(ev["EndTime"]), color=color, alpha=0.4)
    ax2.axvspan(pd.Timestamp(ev["StartTime"]), pd.Timestamp(ev["EndTime"]), color=color, alpha=0.4)

# Curva de alarmas predichas (muestreada cada hora para que sea visible)
serie_alarmas = pd.Series(y_pred, index=timestamps).resample("1h").max()
ax1.plot(serie_alarmas.index, serie_alarmas.values, color="crimson", lw=1.0, label="Alarma Activa (Prediccion)")
ax1.set_ylabel("Alarma")
ax1.set_yticks([0, 1])
ax1.set_title("Resumen General de Test (2004-2005): Alarmas del Modelo frente a Fallas Reales")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)

# Curva de verdad terreno
serie_real = pd.Series(y_real, index=timestamps).resample("1h").max()
ax2.plot(serie_real.index, serie_real.values, color="black", lw=1.0, label="Ground Truth (Falla Real)")
ax2.set_ylabel("Etiqueta Real")
ax2.set_yticks([0, 1])
ax2.set_xlabel("Fecha")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)

plt.tight_layout()
ruta_fig10 = os.path.join(DIR_SALIDA, "fig10_resumen_deteccion.png")
plt.savefig(ruta_fig10, dpi=120)
print(f"Guardado: {ruta_fig10}")
plt.show()
print("\nEtapa 6 configurada exitosamente.")
