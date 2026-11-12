
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DIR_CODIGO = os.path.dirname(os.path.abspath(__file__))
DIR_TP = os.path.dirname(DIR_CODIGO)
DIR_DATOS = os.path.join(DIR_TP, "datos_procesados")
DIR_EWMA = os.path.join(DIR_DATOS, "ewma")
DIR_MLP = os.path.join(DIR_DATOS, "mlp")
DIR_SALIDA = os.path.join(DIR_DATOS, "metricas")
os.makedirs(DIR_SALIDA, exist_ok=True)


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



def calcular_metricas(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)

    tp = int(np.sum((y_real == 1) & (y_pred == 1)))
    fp = int(np.sum((y_real == 0) & (y_pred == 1)))
    tn = int(np.sum((y_real == 0) & (y_pred == 0)))
    fn = int(np.sum((y_real == 1) & (y_pred == 0)))

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


# verdad terreno : 1 si hay anomalia, 0 si es normal
y_real = test["anomalia"].values.astype(np.int8)

# alarmas sin filtro ewma vs con ewma
alarma_cruda = (test["err_medio"].values > umbral).astype(np.int8)
alarma_ewma = test["alarma"].values.astype(np.int8)

met_crudo = calcular_metricas(y_real, alarma_cruda)
met_ewma = calcular_metricas(y_real, alarma_ewma)

# metricas del modelo comparativo MLP 
test_mlp = pd.read_pickle(os.path.join(DIR_MLP, "mlp_test.pkl"))
with open(os.path.join(DIR_MLP, "metricas_mlp.json")) as f:
    resumen_mlp = json.load(f)
met_mlp_crudo = resumen_mlp["mlp_sin_ewma"]
met_mlp_ewma = resumen_mlp["mlp_con_ewma"]

print("\n   matriz de confusion y metricas: MLP vs. LSTM (Instante por Instante) ")
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



# en operaciones satelitales importa saber si cada evento documentado por la ESA
# disparo la alarma mientras ocurria (comparamos MLP vs LSTM)
filas_ev = []
for ev_id, ev in eventos.iterrows():
    tramo_lstm = test.loc[ev["StartTime"]:ev["EndTime"]]
    tramo_mlp = test_mlp.loc[ev["StartTime"]:ev["EndTime"]]
    if len(tramo_lstm) == 0:          # id_93 cae en Train
        continue
    n_inst = len(tramo_lstm)
    n_al_mlp = int(tramo_mlp["alarma"].sum())
    n_al_lstm = int(tramo_lstm["alarma"].sum())
    filas_ev.append({
        "ID": ev_id,
        "Category": ev["Category"],
        "inicio": str(ev["StartTime"].date()),
        "duracion_h": ev["duracion_h"],
        "instantes": n_inst,
        "inst_mlp": n_al_mlp,
        "cob_mlp_%": round(n_al_mlp / n_inst * 100, 1),
        "det_mlp": "SI" if n_al_mlp > 0 else "NO",
        "inst_lstm": n_al_lstm,
        "cob_lstm_%": round(n_al_lstm / n_inst * 100, 1),
        "det_lstm": "SI" if n_al_lstm > 0 else "NO",
    })

tabla_ev = pd.DataFrame(filas_ev).set_index("ID")
tabla_ev.to_csv(os.path.join(DIR_SALIDA, "deteccion_por_evento.csv"))

anom_df = tabla_ev[tabla_ev["Category"] == "Anomaly"]
rare_df = tabla_ev[tabla_ev["Category"] == "Rare Event"]
n_anom_det = (anom_df["det_lstm"] == "SI").sum()
n_rare_det = (rare_df["det_lstm"] == "SI").sum()
n_anom_mlp = (anom_df["det_mlp"] == "SI").sum()
n_rare_mlp = (rare_df["det_mlp"] == "SI").sum()

print("\n    cobertura por evento (MLP vs LSTM)    ")
print(tabla_ev.to_string())
print(f"\n resumen de eventos detectados:")
print(f"  - anomalias reales (Anomaly):    MLP = {n_anom_mlp}/{len(anom_df)} | LSTM = {n_anom_det}/{len(anom_df)}")
print(f"  - eventos raros    (Rare Event): MLP = {n_rare_mlp}/{len(rare_df)} | LSTM = {n_rare_det}/{len(rare_df)}")



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




def dibujar_matriz_confusion(ax, met, titulo, cmap="Blues"):
   
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


# --- Figura EWMA segun variacion de alpha (impacto en senal y en metricas)
def filtrar_ewma_rapido(e, a):
    e = np.asarray(e, dtype=np.float64)
    s = np.empty_like(e)
    s[0] = e[0]
    inercia = 1.0 - a
    for t in range(1, len(e)):
        s[t] = a * e[t] + inercia * s[t - 1]
    return s.astype(np.float32)


valores_alpha = [0.01, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
err_crudo_lstm = test["err_medio"].values
err_crudo_mlp = test_mlp["err_medio"].values

curva_alpha_lstm = []
curva_alpha_mlp = []
for a in valores_alpha:
    s_l = filtrar_ewma_rapido(err_crudo_lstm, a) if a < 1.0 else err_crudo_lstm
    s_m = filtrar_ewma_rapido(err_crudo_mlp, a) if a < 1.0 else err_crudo_mlp
    m_l = calcular_metricas(y_real, (s_l > umbral).astype(np.int8))
    m_m = calcular_metricas(y_real, (s_m > umbral).astype(np.int8))
    m_l["alpha"] = a
    m_m["alpha"] = a
    curva_alpha_lstm.append(m_l)
    curva_alpha_mlp.append(m_m)

df_a_lstm = pd.DataFrame(curva_alpha_lstm)
df_a_mlp = pd.DataFrame(curva_alpha_mlp)

fig, (ax_sig, ax_met) = plt.subplots(1, 2, figsize=(15.5, 5.0))

# Panel A: efecto de distintos valores de alpha sobre el tramo del evento id_96
ev96 = eventos.loc["id_96"]
desde96 = ev96["StartTime"] - pd.Timedelta("12h")
hasta96 = ev96["EndTime"] + pd.Timedelta("12h")
tramo96 = test.loc[desde96:hasta96]
err_tramo = tramo96["err_medio"].values

ax_sig.plot(tramo96.index, err_tramo, lw=0.6, color="lightsteelblue", alpha=0.8,
            label="Sin filtro ($\\alpha = 1.00$, error crudo)")
ax_sig.plot(tramo96.index, filtrar_ewma_rapido(err_tramo, 0.25), lw=1.1,
            color="darkorange", alpha=0.85, label="Suavizado leve ($\\alpha = 0.25$)")
ax_sig.plot(tramo96.index, filtrar_ewma_rapido(err_tramo, 0.05), lw=1.8,
            color="navy", label="Suavizado elegido ($\\alpha = 0.05$)")
ax_sig.axhline(umbral, color="green", ls="--", lw=1.4, label=f"Umbral = {umbral:.3f}")
ax_sig.axvspan(ev96["StartTime"], ev96["EndTime"], color="red", alpha=0.20, label="Anomalia real (id_96)")
ax_sig.set_title("A) Efecto de $\\alpha$ sobre el error residual (evento id_96)", fontweight="bold")
ax_sig.set_ylabel("Error residual / EWMA $s_t$")
ax_sig.set_xlabel("Tiempo")
ax_sig.legend(loc="upper left", fontsize=8.5)
ax_sig.grid(alpha=0.3)
ax_sig.tick_params(axis="x", rotation=15)

# Panel B: evolucion de Precision, Sensibilidad y F1-Score al variar alpha
ax_met.plot(df_a_lstm["alpha"], df_a_lstm["precision"] * 100, "o-", color="royalblue", label="Precision (LSTM)")
ax_met.plot(df_a_lstm["alpha"], df_a_lstm["sensibilidad"] * 100, "s--", color="steelblue", alpha=0.75, label="Sensibilidad (LSTM)")
ax_met.plot(df_a_lstm["alpha"], df_a_lstm["f1"] * 100, "D-", lw=2.2, color="navy", label="F1-Score (LSTM)")

ax_met.plot(df_a_mlp["alpha"], df_a_mlp["precision"] * 100, "o-", color="sandybrown", label="Precision (MLP)")
ax_met.plot(df_a_mlp["alpha"], df_a_mlp["f1"] * 100, "D-", lw=2.0, color="darkorange", label="F1-Score (MLP)")

ax_met.axvline(alpha, color="crimson", ls="--", lw=1.5, label=f"$\\alpha$ elegido = {alpha}")
ax_met.set_title("B) Desempeno en Test al variar el factor de suavizado $\\alpha$", fontweight="bold")
ax_met.set_xlabel("Factor de suavizado EWMA $\\alpha$ ($1.0$ = sin filtro)")
ax_met.set_ylabel("Porcentaje (%)")
ax_met.set_ylim(10, 103)
ax_met.grid(alpha=0.3)
ax_met.legend(loc="center right", fontsize=8.5, ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(DIR_SALIDA, "fig_ewma_variacion_alpha.png"), dpi=120)



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

    # serie temporal real vs prediccion LSTM (canal 41)
    axes[0, col].plot(tramo.index, tramo["real_41"], lw=0.8, color="tab:blue", label="Real (ch41)")
    axes[0, col].plot(tramo.index, tramo["pred_41"], lw=0.8, color="tab:orange", label="Prediccion LSTM")
    axes[0, col].axvspan(ev["StartTime"], ev["EndTime"], color="red", alpha=0.22, label="Ground Truth ESA")
    axes[0, col].set_title(titulo)
    axes[0, col].set_ylabel("ch41 (escalado)")
    axes[0, col].legend(loc="upper left", fontsize=8)
    axes[0, col].grid(alpha=0.3)

    # curva de error residual suavizado (ewma) cruzando el umbral
    axes[1, col].plot(tramo.index, tramo["ewma_medio"], lw=1.4, color="black", label="Error EWMA $s_t$")
    axes[1, col].axhline(umbral, color="green", ls="--", lw=1.3, label=f"Umbral = {umbral:.3f}")
    axes[1, col].axvspan(ev["StartTime"], ev["EndTime"], color="red", alpha=0.22)
    axes[1, col].set_ylabel("Error EWMA $s_t$")
    axes[1, col].legend(loc="upper left", fontsize=8)
    axes[1, col].grid(alpha=0.3)

    # bandera de alarma del modelo vs ground truth
    axes[2, col].fill_between(tramo.index, 0, tramo["anomalia"].astype(int),
                              step="post", color="red", alpha=0.3, label="Anomalia Real (ESA)")
    axes[2, col].step(tramo.index, tramo["alarma"], where="post",
                      color="crimson", lw=1.4, label="Alarma Modelo")
    axes[2, col].set_yticks([0, 1])
    axes[2, col].set_yticklabels(["Normal", "Alarma"])
    axes[2, col].set_ylim(-0.1, 1.15)
    axes[2, col].legend(loc="upper left", fontsize=8)
    axes[2, col].grid(alpha=0.3)
    axes[2, col].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(DIR_SALIDA, "fig13_panel_sincronizado_resultados.png"), dpi=120)



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
