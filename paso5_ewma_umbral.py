"""
TP Final - Etapa 5: Filtrado EWMA y Deteccion por Umbral Estadistico.

Que hace este script:
  1. Carga los errores residuales instantaneos de Test generados en la Etapa 4
     (e(t) = |y(t) - y_hat(t)|).
  2. Aplica un suavizado EWMA (Exponentially Weighted Moving Average) para
     atenuar picos transitorios de ruido y resaltar desvios fisicos sostenidos:
       s(t) = alpha * e(t) + (1 - alpha) * s(t-1)
  3. Calibra el umbral de decision sobre el regimen normal de referencia:
       Umbral = mu_0 + k * sigma_0  (con k tipico entre 3 y 4)
  4. Genera la senal de alarma binaria:
       Alarma(t) = 1 si s(t) > Umbral, 0 en caso contrario.
  5. Sincroniza con labels.csv y grafica el comportamiento de la alarma frente
     a las anomalias reales (verificacion visual del detector antes de metricas).

Salidas (datos_procesados/deteccion/):
  - error_suavizado_test.npy    Matriz de errores suavizados s(t)
  - alarmas_test.npy            Vector de alarmas binarias (0 o 1)
  - config_umbral.json          Parametros mu_0, sigma_0, k y umbral elegidos
  - fig7_ewma_vs_crudo.png      Comparacion Error Crudo vs. Error Filtrado EWMA
  - fig8_deteccion_umbral.png   Senal de error cruzando el umbral frente a fallas
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0) CONFIGURACION Y RUTAS
# ---------------------------------------------------------------------------
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(DIR_BASE, "datos_procesados")
DIR_PRED = os.path.join(DIR_DATOS, "predicciones")
DIR_SALIDA = os.path.join(DIR_DATOS, "deteccion")
os.makedirs(DIR_SALIDA, exist_ok=True)

# Hiperparametros del filtrado y decision
ALPHA = 0.15     # Factor de suavizado EWMA (menor alpha = mas inercia/filtro)
K_SIGMA = 3.5    # Multiplicador del umbral: mu + k * sigma


# ---------------------------------------------------------------------------
# 1) CARGA DE RESIDUOS DE LA ETAPA 4
# ---------------------------------------------------------------------------
print("=== 1) Carga de residuos de Test ===")
ruta_error = os.path.join(DIR_PRED, "error_test.npy")

if not os.path.exists(ruta_error):
    print(f"AVISO: No se encontro '{ruta_error}'. Este script procesara los datos")
    print("apenas se ejecute 'paso4_prediccion.py'.")
    # Dejamos una matriz dummy solo por si se ejecuta antes para no crashear
    error_test = np.zeros((100, 6), dtype=np.float32)
    timestamps_test = pd.date_range("2004-08-01", periods=100, freq="90s").to_numpy()
else:
    error_test = np.load(ruta_error)
    timestamps_test = np.load(os.path.join(DIR_PRED, "timestamps_test.npy"), allow_pickle=True)
    print(f"Residuos cargados: {len(error_test):,} instantes x {error_test.shape[1]} canales")


# ---------------------------------------------------------------------------
# 2) FILTRADO EWMA (MEDIA MOVIL EXPONENCIAL)
# ---------------------------------------------------------------------------
print(f"\n=== 2) Aplicando filtrado EWMA (alpha = {ALPHA}) ===")

def calcular_ewma(serie_error, alpha):
    """Aplica la ecuacion recursiva EWMA de forma vectorizada en NumPy:
    s[t] = alpha * e[t] + (1 - alpha) * s[t-1]"""
    s = np.empty_like(serie_error)
    s[0] = serie_error[0]
    for t in range(1, len(serie_error)):
        s[t] = alpha * serie_error[t] + (1.0 - alpha) * s[t-1]
    return s

# Aplicamos EWMA columna por columna (para cada canal)
error_suavizado = np.empty_like(error_test)
for j in range(error_test.shape[1]):
    error_suavizado[:, j] = calcular_ewma(error_test[:, j], ALPHA)

# Error compuesto del subsistema: promedio ponderado o maximo de los canales target
# Usamos el error del canal objetivo principal (channel_41) como referencia central
idx_target = 0   # channel_41
error_principal = error_suavizado[:, idx_target]


# ---------------------------------------------------------------------------
# 3) CALIBRACION DEL UMBRAL ESTADISTICO
# ---------------------------------------------------------------------------
print(f"\n=== 3) Calibracion del umbral estadistico (regla de {K_SIGMA}-sigmas) ===")

# Para no contaminar el umbral con las anomalias futuras,
# calculamos mu_0 y sigma_0 sobre el primer tramo sano de Test (o usamos la referencia de Train)
# Tomamos los primeros 10.000 instantes de Test (~10 dias) que son de operacion sana comprobada
tramo_sano_ref = error_principal[:10000]
mu_0 = float(np.mean(tramo_sano_ref))
sigma_0 = float(np.std(tramo_sano_ref))
umbral = mu_0 + K_SIGMA * sigma_0

print(f"Referencia sana: Media (mu_0) = {mu_0:.4f} | Desvio (sigma_0) = {sigma_0:.4f}")
print(f"Umbral de deteccion resultante = {umbral:.4f} (mu_0 + {K_SIGMA} * sigma_0)")


# ---------------------------------------------------------------------------
# 4) GENERACION DE ALARMAS BINARIAS (0 o 1)
# ---------------------------------------------------------------------------
print("\n=== 4) Generacion de alarmas ===")
alarmas = (error_principal > umbral).astype(np.int8)

total_alarmas = np.sum(alarmas)
pct_alarmas = (total_alarmas / len(alarmas)) * 100
print(f"Instantes en estado de alarma: {total_alarmas:,} de {len(alarmas):,} ({pct_alarmas:.2f}%)")


# ---------------------------------------------------------------------------
# 5) GUARDADO DE RESULTADOS
# ---------------------------------------------------------------------------
np.save(os.path.join(DIR_SALIDA, "error_suavizado_test.npy"), error_suavizado)
np.save(os.path.join(DIR_SALIDA, "alarmas_test.npy"), alarmas)

config_umbral = {
    "alpha_ewma": ALPHA,
    "k_sigma": K_SIGMA,
    "mu_0": mu_0,
    "sigma_0": sigma_0,
    "umbral_deteccion": umbral,
    "total_alarmas": int(total_alarmas),
    "pct_tiempo_alarma": float(pct_alarmas)
}
with open(os.path.join(DIR_SALIDA, "config_umbral.json"), "w") as f:
    json.dump(config_umbral, f, indent=2)
print(f"Archivos de deteccion guardados en: {DIR_SALIDA}")


# ---------------------------------------------------------------------------
# 6) GRAFICOS DE VERIFICACION DEL FILTRADO Y UMBRAL
# ---------------------------------------------------------------------------
print("\n=== 6) Generando graficos de verificacion visual ===")

# Cargamos el catalogo de eventos para sombrear las anomalias reales
ruta_ev = os.path.join(DIR_DATOS, "eventos_2004_2005.csv")
if os.path.exists(ruta_ev):
    eventos = pd.read_csv(ruta_ev, index_col="ID")
    ev = eventos.loc["id_96"]
    t_ini_ev = pd.Timestamp(ev["StartTime"])
    t_fin_ev = pd.Timestamp(ev["EndTime"])

    margen = pd.Timedelta("3D")
    t_inicio_zoom = t_ini_ev - margen
    t_fin_zoom = t_fin_ev + margen

    mascara_zoom = (timestamps_test >= t_inicio_zoom) & (timestamps_test <= t_fin_zoom)
    t_zoom = timestamps_test[mascara_zoom]
    err_crudo_zoom = error_test[mascara_zoom, idx_target]
    err_ewma_zoom = error_suavizado[mascara_zoom, idx_target]
    alarmas_zoom = alarmas[mascara_zoom]

    # --- Figura 7: Efecto del filtrado EWMA (Crudo vs. Suavizado) ---
    plt.figure(figsize=(14, 5))
    plt.plot(t_zoom, err_crudo_zoom, color="gray", alpha=0.6, lw=0.8, label="Error crudo e(t) [con ruido]")
    plt.plot(t_zoom, err_ewma_zoom, color="tab:blue", lw=1.5, label=f"Error filtrado EWMA s(t) [alpha={ALPHA}]")
    plt.axhline(umbral, color="red", ls="--", lw=1.5, label=f"Umbral ({umbral:.3f})")
    plt.axvspan(t_ini_ev, t_fin_ev, color="red", alpha=0.25, label=f"Falla real certificada ({ev['Category']})")
    plt.ylabel("Magnitud del Error")
    plt.title("Efecto del Suavizado EWMA: Supresion de Ruido y Deteccion de Tendencia Sostenida")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    ruta_fig7 = os.path.join(DIR_SALIDA, "fig7_ewma_vs_crudo.png")
    plt.savefig(ruta_fig7, dpi=120)
    plt.close()
    print(f"Guardado: {ruta_fig7}")

    # --- Figura 8: Disparo de la Alarma Binaria ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    
    ax1.plot(t_zoom, err_ewma_zoom, color="tab:blue", lw=1.2, label="Error residual suavizado s(t)")
    ax1.axhline(umbral, color="red", ls="--", lw=1.2, label="Umbral de decision")
    ax1.axvspan(t_ini_ev, t_fin_ev, color="red", alpha=0.25, label="Anomalia real")
    ax1.set_ylabel("Error suavizado")
    ax1.set_title("Deteccion de Fallas: Error Filtrado vs. Umbral de Alarma")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    ax2.step(t_zoom, alarmas_zoom, color="crimson", lw=1.5, where="mid", label="Bandera de Alarma (1=Anomalia)")
    ax2.axvspan(t_ini_ev, t_fin_ev, color="red", alpha=0.25)
    ax2.set_ylabel("Alarma")
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_yticks([0, 1])
    ax2.set_xlabel("Fecha y Hora")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    ruta_fig8 = os.path.join(DIR_SALIDA, "fig8_deteccion_umbral.png")
    plt.savefig(ruta_fig8, dpi=120)
    plt.close()
    print(f"Guardado: {ruta_fig8}")

print("Etapa 5 lista y configurada con exito.")
