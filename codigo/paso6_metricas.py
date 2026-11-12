"""
TP Final - Etapa 6: Evaluacion experimental contra Ground Truth y metricas.

Que hace este script (programado a mano en NumPy):
  1. Carga las alarmas y errores suavizados de la Etapa 5 (ewma_test.pkl).
  2. Calcula a mano la Matriz de Confusion (TP, FP, TN, FN) y las metricas:
     Precision, Sensibilidad (Recall) y F1-Score.
  3. Compara el desempeno SIN filtro EWMA (error crudo) vs. CON filtro EWMA.
  4. Evalua la deteccion evento por evento contra labels.csv (Anomalies y Rare Events).
  5. Analiza como varian Precision, Sensibilidad y F1 al mover el factor k del umbral.
  6. Genera los graficos finales de resultados para el informe y la defensa.

Salidas (TP Final/datos_procesados/metricas/):
  - metricas_resumen.json          TP, FP, TN, FN, Precision, Recall, F1
  - deteccion_por_evento.csv       tabla de cobertura de los 14 eventos de Test
  - fig11_matriz_confusion_y_metricas.png
  - fig12_curva_umbral_k.png
  - fig13_panel_sincronizado_resultados.png
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0) CONFIGURACION
# ---------------------------------------------------------------------------
DIR_CODIGO = os.path.dirname(os.path.abspath(__file__))
DIR_TP = os.path.dirname(DIR_CODIGO)
DIR_DATOS = os.path.join(DIR_TP, "datos_procesados")
DIR_EWMA = os.path.join(DIR_DATOS, "ewma")
DIR_MLP = os.path.join(DIR_DATOS, "mlp")
DIR_SALIDA = os.path.join(DIR_DATOS, "metricas")
os.makedirs(DIR_SALIDA, exist_ok=True)


# ---------------------------------------------------------------------------
# 1) CARGA DE DATOS Y PARAMETROS DE LA ETAPA 5
# ---------------------------------------------------------------------------
test = pd.read_pickle(os.path.join(DIR_EWMA, "ewma_test.pkl"))
with open(os.path.join(DIR_EWMA, "config_ewma.json")) as f:
    cfg_ewma = json.load(f)

eventos = pd.read_csv(os.path.join(DIR_DATOS, "eventos_2004_2005.csv"),
                      index_col="ID", parse_dates=["StartTime", "EndTime"])

mu0 = cfg_ewma["mu0"]
sigma0 = cfg_ewma["sigma0"]
k_opt = cfg_ewma["k_umbral"]
umbral = cfg_ewma["umbral"]
alpha = cfg_ewma["alpha"]

print(f"Evaluando Test: {len(test):,} instantes | Umbral = {umbral:.4f} (k = {k_opt})")


# ---------------------------------------------------------------------------
# 2) MATRIZ DE CONFUSION Y METRICAS A MANO EN NUMPY
# ---------------------------------------------------------------------------
def calcular_metricas(y_real, y_pred):
    """Calcula TP, FP, TN, FN, Precision, Sensibilidad (Recall) y F1-Score.

    Formulas (Paso 6 de la Propuesta):
      - Precision = TP / (TP + FP)          -> confiabilidad de las alarmas
      - Sensibilidad (Recall) = TP / (TP + FN) -> cobertura de fallas reales
      - F1-Score = 2 * Precision * Recall / (Precision + Recall)
    """
    y_real = np.asarray(y_real, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)

    tp = int(np.sum((y_real == 1) & (y_pred == 1)))   # Verdadero Positivo: falla detectada
    fp = int(np.sum((y_real == 0) & (y_pred == 1)))   # Falso Positivo: falsa alarma
    tn = int(np.sum((y_real == 0) & (y_pred == 0)))   # Verdadero Negativo: normal bien clasificado
    fn = int(np.sum((y_real == 1) & (y_pred == 0)))   # Falso Negativo: falla no detectada

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * sensibilidad / (precision + sensibilidad)
          if (precision + sensibilidad) > 0 else 0.0)

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision,
        "sensibilidad": sensibilidad,
        "f1": f1,
    }


# Verdad terreno (Ground Truth de labels.csv): 1 si hay anomalia, 0 si es normal
y_real = test["anomalia"].values.astype(np.int8)

# Alarmas SIN filtro EWMA (aplicando el mismo umbral al error crudo) vs CON EWMA
alarma_cruda = (test["err_medio"].values > umbral).astype(np.int8)
alarma_ewma = test["alarma"].values.astype(np.int8)

met_crudo = calcular_metricas(y_real, alarma_cruda)
met_ewma = calcular_metricas(y_real, alarma_ewma)

# Cargamos las metricas del modelo comparativo MLP (generadas por paso_mlp.py)
with open(os.path.join(DIR_MLP, "metricas_mlp.json")) as f:
    resumen_mlp = json.load(f)
met_mlp_crudo = resumen_mlp["mlp_sin_ewma"]
met_mlp_ewma = resumen_mlp["mlp_con_ewma"]

print("\n=== 2) Matriz de Confusion y Metricas: MLP vs. LSTM (Instante por Instante) ===")
print(f"{'Modelo / Metodo':<20} | {'TP':>6} | {'FP':>6} | {'TN':>7} | {'FN':>5} | "
      f"{'Precision':>9} | {'Sensibil.':>9} | {'F1-Score':>8}")
print("-" * 88)
comparativa = [
    ("MLP Sin EWMA", met_mlp_crudo),
    ("MLP Con EWMA", met_mlp_ewma),
    ("LSTM Sin EWMA", met_crudo),
    ("LSTM Con EWMA", met_ewma),
]
for nombre, m in comparativa:
    print(f"{nombre:<20} | {m['TP']:>6,} | {m['FP']:>6,} | {m['TN']:>7,} | {m['FN']:>5,} | "
          f"{m['precision']*100:>8.2f}% | {m['sensibilidad']*100:>8.2f}% | {m['f1']*100:>7.2f}%")


# ---------------------------------------------------------------------------
# 3) EVALUACION EVENTO POR EVENTO (COBERTURA DE ANOMALIAS)
# ---------------------------------------------------------------------------
# En operaciones satelitales importa saber si cada evento documentado por la ESA
# disparo la alarma mientras ocurria.
filas_ev = []
for ev_id, ev in eventos.iterrows():
    tramo = test.loc[ev["StartTime"]:ev["EndTime"]]
    if len(tramo) == 0:          # id_93 cae en Train
        continue
    n_inst = len(tramo)
    n_alarmas = int(tramo["alarma"].sum())
    detectado = n_alarmas > 0
    filas_ev.append({
        "ID": ev_id,
        "Category": ev["Category"],
        "inicio": str(ev["StartTime"].date()),
        "duracion_h": ev["duracion_h"],
        "instantes": n_inst,
        "inst_en_alarma": n_alarmas,
        "cobertura_%": round(n_alarmas / n_inst * 100, 1),
        "detectado": "SI" if detectado else "NO",
    })

tabla_ev = pd.DataFrame(filas_ev).set_index("ID")
tabla_ev.to_csv(os.path.join(DIR_SALIDA, "deteccion_por_evento.csv"))

anom_df = tabla_ev[tabla_ev["Category"] == "Anomaly"]
rare_df = tabla_ev[tabla_ev["Category"] == "Rare Event"]
n_anom_det = (anom_df["detectado"] == "SI").sum()
n_rare_det = (rare_df["detectado"] == "SI").sum()

print("\n=== 3) Cobertura por Evento (Ground Truth ESA) ===")
print(tabla_ev.to_string())
print(f"\nResumen de eventos detectados:")
print(f"  - Anomalias reales (Category = Anomaly): {n_anom_det}/{len(anom_df)} "
      f"({n_anom_det/len(anom_df)*100:.0f}% detectadas)")
print(f"  - Eventos raros    (Category = Rare Event): {n_rare_det}/{len(rare_df)} "
      f"({n_rare_det/len(rare_df)*100:.1f}% detectados)")


# ---------------------------------------------------------------------------
# 4) ANALISIS DE SENSIBILIDAD DEL COEFICIENTE k DEL UMBRAL
# ---------------------------------------------------------------------------
valores_k = np.linspace(1.5, 5.5, 17)
curva_k = []
for k in valores_k:
    u_k = mu0 + k * sigma0
    pred_k = (test["ewma_medio"].values > u_k).astype(np.int8)
    m_k = calcular_metricas(y_real, pred_k)
    m_k["k"] = float(k)
    m_k["umbral"] = float(u_k)
    curva_k.append(m_k)
df_k = pd.DataFrame(curva_k)


# ---------------------------------------------------------------------------
# 5) GRAFICOS DE RESULTADOS
# ---------------------------------------------------------------------------

# --- Figura 11: Matrices de Confusion (MLP vs LSTM) + Comparacion de Metricas
def dibujar_matriz_confusion(ax, met, titulo, cmap="Blues"):
    """Dibuja una matriz de confusion 2x2 con conteos y porcentajes."""
    matriz_c = np.array([
        [met["TN"], met["FP"]],
        [met["FN"], met["TP"]],
    ])
    ax.imshow(np.log10(matriz_c + 1), cmap=cmap, vmin=2, vmax=6.2)
    etiquetas_celda = [
        [f"TN (Verdadero Neg.)\n{met['TN']:,}\n({met['TN']/len(y_real)*100:.2f}%)",
         f"FP (Falsa Alarma)\n{met['FP']:,}\n({met['FP']/len(y_real)*100:.2f}%)"],
        [f"FN (Falla Omitida)\n{met['FN']:,}\n({met['FN']/len(y_real)*100:.2f}%)",
         f"TP (Falla Detectada)\n{met['TP']:,}\n({met['TP']/len(y_real)*100:.2f}%)"],
    ]
    for i in range(2):
        for j in range(2):
            color_txt = "white" if (i == 0 and j == 0) else "black"
            ax.text(j, i, etiquetas_celda[i][j], ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=color_txt)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicho: 0 (Normal)", "Predicho: 1 (Anomalia)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Real: 0 (Normal)", "Real: 1 (Anomalia)"])
    ax.set_title(titulo, fontweight="bold")


fig, (ax_mlp, ax_lstm, ax_bar) = plt.subplots(
    1, 3, figsize=(18.5, 5.4), gridspec_kw={"width_ratios": [1, 1, 1.35]}
)

dibujar_matriz_confusion(ax_mlp, met_mlp_ewma, "Matriz de Confusion: MLP (con EWMA)", cmap="Oranges")
dibujar_matriz_confusion(ax_lstm, met_ewma, "Matriz de Confusion: LSTM (con EWMA)", cmap="Blues")

nombres_m = ["Precision", "Sensibilidad (Recall)", "F1-Score"]
vals_mlp_crudo = [met_mlp_crudo["precision"] * 100, met_mlp_crudo["sensibilidad"] * 100, met_mlp_crudo["f1"] * 100]
vals_mlp_ewma = [met_mlp_ewma["precision"] * 100, met_mlp_ewma["sensibilidad"] * 100, met_mlp_ewma["f1"] * 100]
vals_lstm_crudo = [met_crudo["precision"] * 100, met_crudo["sensibilidad"] * 100, met_crudo["f1"] * 100]
vals_lstm_ewma = [met_ewma["precision"] * 100, met_ewma["sensibilidad"] * 100, met_ewma["f1"] * 100]

x = np.arange(len(nombres_m))
ancho = 0.19
b1 = ax_bar.bar(x - 1.5 * ancho, vals_mlp_crudo, ancho, label="MLP Sin EWMA", color="navajowhite", edgecolor="peru")
b2 = ax_bar.bar(x - 0.5 * ancho, vals_mlp_ewma, ancho, label="MLP Con EWMA", color="darkorange", edgecolor="saddlebrown")
b3 = ax_bar.bar(x + 0.5 * ancho, vals_lstm_crudo, ancho, label="LSTM Sin EWMA", color="lightsteelblue", edgecolor="steelblue")
b4 = ax_bar.bar(x + 1.5 * ancho, vals_lstm_ewma, ancho, label="LSTM Con EWMA", color="royalblue", edgecolor="navy")

for barras in [b1, b2, b3, b4]:
    for b in barras:
        h = b.get_height()
        ax_bar.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(nombres_m, fontweight="bold")
ax_bar.set_ylabel("Porcentaje (%)")
ax_bar.set_ylim(0, 112)
ax_bar.set_title("Comparacion de Metricas: MLP vs. LSTM (con y sin EWMA)", fontweight="bold")
ax_bar.legend(loc="lower right", ncol=2, fontsize=8.5)
ax_bar.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DIR_SALIDA, "fig11_matriz_confusion_y_metricas.png"), dpi=120)


# --- Figura 12: Curvas de Precision, Sensibilidad y F1 en funcion de k
plt.figure(figsize=(8.5, 4.5))
plt.plot(df_k["k"], df_k["precision"] * 100, "o-", label="Precision", color="tab:blue")
plt.plot(df_k["k"], df_k["sensibilidad"] * 100, "s-", label="Sensibilidad (Recall)", color="tab:orange")
plt.plot(df_k["k"], df_k["f1"] * 100, "D-", lw=2, label="F1-Score", color="tab:green")
plt.axvline(k_opt, color="crimson", ls="--", label=f"k elegido = {k_opt} (F1 = {met_ewma['f1']*100:.1f}%)")
plt.xlabel("Coeficiente de sensibilidad k (Umbral = mu_0 + k * sigma_0)")
plt.ylabel("Metrica (%)")
plt.title("Compromiso Precision vs. Sensibilidad al variar el umbral k")
plt.ylim(40, 103)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIR_SALIDA, "fig12_curva_umbral_k.png"), dpi=120)


# --- Figura 13: Panel sincronizado de resultados (Propuesta Paso 6, punto 4)
# Mostramos lado a lado dos anomalias reales de distinta naturaleza:
#   Columna izquierda: id_96 (anomalia sutil dentro del rango normal)
#   Columna derecha:   id_101 (anomalia severa por caida abrupta de senal)
fig, axes = plt.subplots(3, 2, figsize=(15, 8.5), sharex="col",
                         gridspec_kw={"height_ratios": [2, 2, 1]})

casos = [
    ("id_96", "Anomalia sutil dentro de rango (id_96)"),
    ("id_101", "Anomalia severa por caida abrupta (id_101)"),
]
for col, (ev_id, titulo) in enumerate(casos):
    ev = eventos.loc[ev_id]
    desde = ev["StartTime"] - pd.Timedelta("10h")
    hasta = ev["EndTime"] + pd.Timedelta("10h")
    tramo = test.loc[desde:hasta]

    # 1) Serie temporal real vs prediccion LSTM (canal 41)
    axes[0, col].plot(tramo.index, tramo["real_41"], lw=0.8, color="tab:blue", label="Real (ch41)")
    axes[0, col].plot(tramo.index, tramo["pred_41"], lw=0.8, color="tab:orange", label="Prediccion LSTM")
    axes[0, col].axvspan(ev["StartTime"], ev["EndTime"], color="red", alpha=0.22, label="Ground Truth ESA")
    axes[0, col].set_title(titulo)
    axes[0, col].set_ylabel("ch41 (escalado)")
    axes[0, col].legend(loc="upper left", fontsize=8)
    axes[0, col].grid(alpha=0.3)

    # 2) Curva de error residual suavizado (EWMA) cruzando el umbral
    axes[1, col].plot(tramo.index, tramo["ewma_medio"], lw=1.4, color="black", label="Error EWMA $s_t$")
    axes[1, col].axhline(umbral, color="green", ls="--", lw=1.3, label=f"Umbral = {umbral:.3f}")
    axes[1, col].axvspan(ev["StartTime"], ev["EndTime"], color="red", alpha=0.22)
    axes[1, col].set_ylabel("Error EWMA $s_t$")
    axes[1, col].legend(loc="upper left", fontsize=8)
    axes[1, col].grid(alpha=0.3)

    # 3) Bandera de alarma del modelo vs Ground Truth
    axes[2, col].fill_between(tramo.index, 0, tramo["anomalia"].astype(int),
                              step="post", color="red", alpha=0.3, label="Anomalia Real (ESA)")
    axes[2, col].step(tramo.index, tramo["alarma"], where="post",
                      color="crimson", lw=1.4, label="Alarma Modelo ($\\hat{C}_t$)")
    axes[2, col].set_yticks([0, 1])
    axes[2, col].set_yticklabels(["Normal", "Alarma"])
    axes[2, col].set_ylim(-0.1, 1.15)
    axes[2, col].legend(loc="upper left", fontsize=8)
    axes[2, col].grid(alpha=0.3)
    axes[2, col].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(DIR_SALIDA, "fig13_panel_sincronizado_resultados.png"), dpi=120)


# ---------------------------------------------------------------------------
# 6) GUARDADO DEL RESUMEN DE METRICAS
# ---------------------------------------------------------------------------
resumen_json = {
    "mlp_sin_ewma": met_mlp_crudo,
    "mlp_con_ewma": met_mlp_ewma,
    "lstm_sin_ewma": met_crudo,
    "lstm_con_ewma": met_ewma,
    "eventos_anomaly_detectados": f"{n_anom_det}/{len(anom_df)}",
    "eventos_rare_detectados": f"{n_rare_det}/{len(rare_df)}",
}
with open(os.path.join(DIR_SALIDA, "metricas_resumen.json"), "w") as f:
    json.dump(resumen_json, f, indent=2)

print(f"\nGuardado en {DIR_SALIDA}")
plt.show()
