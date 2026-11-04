import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 0) configuracion y rutas principales
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(DIR_BASE, "datos_procesados")
DIR_PRED = os.path.join(DIR_DATOS, "predicciones")
DIR_SALIDA = os.path.join(DIR_DATOS, "ewma")
DIR_DETECCION = os.path.join(DIR_DATOS, "deteccion")
os.makedirs(DIR_SALIDA, exist_ok=True)
os.makedirs(DIR_DETECCION, exist_ok=True)

num = [str(i) for i in range(41, 47)]  # canales objetivo: "41" a "46"

# parametros del filtro ewma y calibracion del umbral
alpha = 0.05       #  menor alpha otorga mayor inercia y filtra picos de ruido
k_umbral = 3.0     # factor multiplicador del desvio tipico p el umbral de decision (3 sigmas)

# 1) carga de predicciones y residuos instantaneos del paso 4
val = pd.read_pickle(os.path.join(DIR_PRED, "pred_val.pkl"))
test = pd.read_pickle(os.path.join(DIR_PRED, "pred_test.pkl"))
eventos = pd.read_csv(os.path.join(DIR_DATOS, "eventos_2004_2005.csv"),index_col="ID", parse_dates=["StartTime", "EndTime"])

print(f"Cargados -> Val: {len(val):,} instantes | Test: {len(test):,} instantes")

# 2) filtro ewma 
def filtrar_ewma(serie_error, factor_alpha):
    # ecuacion  s_t = alpha * e_t + (1 - alpha) * s_{t-1}
    # otorga inercia temporal p atenuar picos aislados de ruido y resaltar anomalias sostenidas
    e = np.asarray(serie_error, dtype=np.float64)
    s = np.empty_like(e)
    s[0] = e[0]  # primer error registrado
    inercia = 1.0 - factor_alpha
    for t in range(1, len(e)):
        s[t] = factor_alpha * e[t] + inercia * s[t - 1]
    return s.astype(np.float32)

print(f"\n=== Aplicacion de filtro EWMA (alpha = {alpha}) ===")
for tabla in [val, test]:
    # suavizado individual canal por canal (util p diagnostico y p el som)
    for n in num:
        tabla[f"ewma_{n}"] = filtrar_ewma(tabla[f"err_{n}"].values, alpha)
    # suavizado del error medio del subsistema
    tabla["ewma_medio"] = filtrar_ewma(tabla["err_medio"].values, alpha)

# 3) umbral dinamico de decision sobre regimen sano (validacion)
# calculamos media y desvio estandar de los errores residuales en validacion
mu0 = float(val["err_medio"].mean())
sigma0 = float(val["err_medio"].std())
umbral = mu0 + k_umbral * sigma0

print("\n=== Umbral estadistico sobre regimen normal (Val) ===")
print(f"  Media normal (mu_0)          = {mu0:.4f}")
print(f"  Desvio estandar (sigma_0)    = {sigma0:.4f}")
print(f"  Coeficiente k                = {k_umbral:.1f}")
print(f"  Umbral = mu_0 + k * sigma_0  = {umbral:.4f}")

# calculamos y guardamos el umbral especifico de cada canal p diagnostico individual
umbrales_canal = {}
for n in num:
    m_c = float(val[f"err_{n}"].mean())
    s_c = float(val[f"err_{n}"].std())
    u_c = m_c + k_umbral * s_c
    umbrales_canal[n] = {"mu0": m_c, "sigma0": s_c, "umbral": u_c}
    val[f"alarma_{n}"] = (val[f"ewma_{n}"] > u_c).astype(np.int8)
    test[f"alarma_{n}"] = (test[f"ewma_{n}"] > u_c).astype(np.int8)

# 4) clasificacion binaria: alarma = 1 si s_t > umbral, 0 si normal
val["alarma"] = (val["ewma_medio"] > umbral).astype(np.int8)
test["alarma"] = (test["ewma_medio"] > umbral).astype(np.int8)

# comparamos la tasa de falsas alarmas sin ewma vs con ewma en regimen sano
fp_val_crudo = (val["err_medio"] > umbral).mean() * 100
fp_val_ewma = (val["alarma"] == 1).mean() * 100

sano_test = ~test["anomalia"]
fp_test_crudo = (test.loc[sano_test, "err_medio"] > umbral).sum()
fp_test_ewma = (test.loc[sano_test, "alarma"] == 1).sum()

print("\n=== Efecto del filtrado EWMA sobre los picos de ruido ===")
print(f"  En Val (100% sano):  error crudo > Umbral = {fp_val_crudo:.2f}% | "
      f"con EWMA = {fp_val_ewma:.2f}% ({val['alarma'].sum()} instantes)")
print(f"  En Test (tramo sano): error crudo > Umbral = {fp_test_crudo:,} instantes | "
      f"con EWMA = {fp_test_ewma:,} instantes")

# 5) graficos de verificacion visual
color_cat = {"Anomaly": "red", "Rare Event": "orange"}

def sombrear(ax_grafico, desde=None, hasta=None):
    # sombrea las regiones temporales donde ocurrieron fallas documentadas
    for _, ev in eventos.iterrows():
        if desde is not None and (ev["EndTime"] < desde or ev["StartTime"] > hasta):
            continue
        ax_grafico.axvspan(ev["StartTime"], ev["EndTime"],color=color_cat.get(ev["Category"], "gray"), alpha=0.3, lw=0)

# figura 9: efecto antes vs despues del ewma sobre la anomalia id_96
ev = eventos.loc["id_96"]
desde = ev["StartTime"] - pd.Timedelta("12h")
hasta = ev["EndTime"] + pd.Timedelta("12h")
tramo = test.loc[desde:hasta]

fig, axes = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True,gridspec_kw={"height_ratios": [2, 2, 1]})

# panel 1: error crudo vs error suavizado ewma y umbral
axes[0].plot(tramo.index, tramo["err_medio"], lw=0.6, color="steelblue", alpha=0.55, label="Error crudo e_t (con ruido)")
axes[0].plot(tramo.index, tramo["ewma_medio"], lw=1.6, color="black",label=f"Error suavizado EWMA s_t (alpha={alpha})")
axes[0].axhline(umbral, color="green", ls="--", lw=1.4,label=f"Umbral mu_0 + {k_umbral:.0f}*sigma_0 = {umbral:.3f}")
sombrear(axes[0], desde, hasta)
axes[0].set_ylabel("Error medio")
axes[0].set_title("Efecto del filtro EWMA sobre la anomalia sutil id_96")
axes[0].legend(loc="upper left")
axes[0].grid(alpha=0.3)

# panel 2: los 6 errores individuales suavizados
for n in num:
    axes[1].plot(tramo.index, tramo[f"ewma_{n}"], lw=0.9, label=f"ewma ch{n}")
axes[1].axhline(umbral, color="green", ls="--", lw=1.2)
sombrear(axes[1], desde, hasta)
axes[1].set_ylabel("EWMA por canal")
axes[1].legend(loc="upper left", ncol=6, fontsize=8)
axes[1].grid(alpha=0.3)

# panel 3: clasificacion binaria de alarma
axes[2].fill_between(tramo.index, 0, tramo["alarma"], step="post",color="crimson", alpha=0.7, label="Alarma activada (1)")
sombrear(axes[2], desde, hasta)
axes[2].set_yticks([0, 1])
axes[2].set_yticklabels(["0 (Normal)", "1 (Alarma)"])
axes[2].set_ylim(-0.1, 1.15)
axes[2].legend(loc="upper left")
axes[2].grid(alpha=0.3)
plt.tight_layout()
ruta_fig9 = os.path.join(DIR_SALIDA, "fig9_ewma_antes_despues_id_96.png")
plt.savefig(ruta_fig9, dpi=120)
print(f"Grafico guardado: {ruta_fig9}")
plt.show()

# 6) guardado de tablas procesadas y configuracion
val.to_pickle(os.path.join(DIR_SALIDA, "ewma_val.pkl"))
test.to_pickle(os.path.join(DIR_SALIDA, "ewma_test.pkl"))

config_ewma = {
    "alpha": alpha,
    "k_umbral": k_umbral,
    "mu0": mu0,
    "sigma0": sigma0,
    "umbral": umbral,
    "umbrales_canal": umbrales_canal,
}
with open(os.path.join(DIR_SALIDA, "config_ewma.json"), "w") as f:
    json.dump(config_ewma, f, indent=2)

# guardado de compatibilidad en arrays npy p asegurar integracion fluida con paso 6 y som
cols_ewma = [f"ewma_{n}" for n in num]
error_suavizado_test = test[cols_ewma].values.astype(np.float32)
alarmas_test = test["alarma"].values.astype(np.int8)

np.save(os.path.join(DIR_DETECCION, "alarmas_test.npy"), alarmas_test)
np.save(os.path.join(DIR_DETECCION, "error_suavizado_test.npy"), error_suavizado_test)
with open(os.path.join(DIR_DETECCION, "config_umbral.json"), "w") as f:
    json.dump(config_ewma, f, indent=2)

