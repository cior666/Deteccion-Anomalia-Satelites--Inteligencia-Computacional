
import os
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SEMILLA = 42
np.random.seed(SEMILLA)

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(DIR_BASE, "datos_procesados")
DIR_VENT = os.path.join(DIR_DATOS, "ventanas")
DIR_EWMA = os.path.join(DIR_DATOS, "ewma")
DIR_SALIDA = os.path.join(DIR_DATOS, "mlp")
os.makedirs(DIR_SALIDA, exist_ok=True)

NUM = [str(i) for i in range(41, 47)]     # canales target: "41" .. "46"

# el mlp no tiene memoria recurrente interna, toma como vector de entrada
# 9 instantes equiespaciados dentro de la ventana historica de L = 80 pasos
# (t-80, t-70, t-60, t-50, t-40, t-30, t-20, t-10, t-1) x 7 canales = 63 entradas.
RETARDOS = [80, 70, 60, 50, 40, 30, 20, 10, 1]
N_ENTRADAS = len(RETARDOS) * 7            
N_OCULTAS = 16                            
N_SALIDAS = 6                             
CAPAS = [N_ENTRADAS, N_OCULTAS, N_SALIDAS]

# Factor de escala para llevar los z-scores normales [-2.9, +2.5] al rango
# activo (-0.85, +0.85) de la sigmoide bipolar [-1, +1] 
ESCALA_SIGMOIDE = 3.5


VEL_APRENDIZAJE = 0.01
MAX_EPOCAS = 8
PASO_SUBMUESTREO = 10   # submuestreo de ventanas de Train 

# parametros de deteccion 
ALPHA_EWMA = 0.05
K_UMBRAL = 3.0



def sigmoide(v, b=1):
    v = np.clip(v, -50.0, 50.0)
    return 2.0 / (1.0 + np.exp(-b * v)) - 1.0


def derivada_sigmoide(y):
    return 0.5 * (1.0 + y) * (1.0 - y)


def inicializar_pesos(capas):
    w = []
    for p in range(len(capas) - 1):
        filas = capas[p + 1]
        columnas = capas[p] + 1           # +1 por el sesgo x0 = -1
        matriz = np.random.uniform(-0.5, 0.5, size=(filas, columnas))
        w.append(matriz)
    return w


def PropagacionAdelante(x, w, activacion):
    y = [np.array(x)]
    for p in range(len(w)):
        entrada_con_bias = np.concatenate(([-1.0], y[p]))
        v = np.dot(w[p], entrada_con_bias)
        y_capa = activacion(v)
        y.append(y_capa)
    return y


def DeltaSalida(e, y_salida):
    return 0.5 * e * (1.0 + y_salida) * (1.0 - y_salida)


def DeltaOculta(delta_siguiente, w_siguiente, y_capa):
    w_sin_bias = w_siguiente[:, 1:]       # se excluye la columna 0 (bias)
    suma_deltas = np.dot(w_sin_bias.T, delta_siguiente)
    return 0.5 * (1.0 + y_capa) * (1.0 - y_capa) * suma_deltas


def Retropropagacion(e, y, w):
    L = len(w)
    deltas = [None] * L
    deltas[L - 1] = DeltaSalida(e, y[L])
    for p in reversed(range(L - 1)):
        deltas[p] = DeltaOculta(deltas[p + 1], w[p + 1], y[p + 1])
    return deltas


def ActualizarPesos(w_capa, delta_capa, entrada_con_bias, vel_aprendizaje):
    return w_capa + vel_aprendizaje * np.outer(delta_capa, entrada_con_bias)



matriz = np.load(os.path.join(DIR_VENT, "matriz_escalada.npy"))
idx_train = np.load(os.path.join(DIR_VENT, "idx_train.npy"))
idx_val = np.load(os.path.join(DIR_VENT, "idx_val.npy"))
idx_test = np.load(os.path.join(DIR_VENT, "idx_test.npy"))


def armar_entradas(indices):
    return np.hstack([matriz[indices - d] for d in RETARDOS]) / ESCALA_SIGMOIDE


# submuestreamos train cada 10 pasos para entrenar patron por patron sin redundancia
idx_train_sub = idx_train[::PASO_SUBMUESTREO].copy()
X_train = armar_entradas(idx_train_sub)
Yd_train = matriz[idx_train_sub, :6] / ESCALA_SIGMOIDE

X_val = armar_entradas(idx_val)
Y_val_real = matriz[idx_val, :6]

X_test = armar_entradas(idx_test)
Y_test_real = matriz[idx_test, :6]

print(f"Arquitectura MLP (Guia 2): {CAPAS} | Sesgo x0 = -1")
print(f"Patrones de entrenamiento: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")


w = inicializar_pesos(CAPAS)
historial_mse_train = []
historial_mse_val = []

t0 = time.time()
orden = np.arange(len(X_train))

for epoca in range(1, MAX_EPOCAS + 1):
    np.random.shuffle(orden)
    suma_error_cuad = 0.0

    # aprendizaje patron por patron 
    for n in orden:
        x = X_train[n]
        yd = Yd_train[n]

        y = PropagacionAdelante(x, w, sigmoide)

        e = yd - y[-1]
        suma_error_cuad += np.mean((e * ESCALA_SIGMOIDE) ** 2)

        deltas = Retropropagacion(e, y, w)

        for p in range(len(w)):
            entrada_con_bias = np.concatenate(([-1.0], y[p]))
            w[p] = ActualizarPesos(w[p], deltas[p], entrada_con_bias, VEL_APRENDIZAJE)

    mse_train = suma_error_cuad / len(X_train)

    # evaluacion al final de la epoca sobre validacion
    v1 = np.dot(np.column_stack([-np.ones(len(X_val)), X_val]), w[0].T)
    y1 = sigmoide(v1)
    v2 = np.dot(np.column_stack([-np.ones(len(y1)), y1]), w[1].T)
    pred_val_ep = sigmoide(v2) * ESCALA_SIGMOIDE
    mse_val = float(np.mean((Y_val_real - pred_val_ep) ** 2))

    historial_mse_train.append(float(mse_train))
    historial_mse_val.append(mse_val)
    print(f"  Epoca {epoca:02d}/{MAX_EPOCAS} | MSE Train: {mse_train:.4f} | MSE Val: {mse_val:.4f}")

print(f"Entrenamiento MLP finalizado en {time.time() - t0:.2f} s")
np.savez(os.path.join(DIR_SALIDA, "pesos_mlp.npz"), w0=w[0], w1=w[1], capas=np.array(CAPAS))



def predecir_lote(X, w):
    """Aplica la misma propagacion hacia adelante con x0 = -1 sobre una matriz X."""
    a = X
    for p in range(len(w)):
        entrada_con_bias = np.column_stack([-np.ones(len(a)), a])
        v = np.dot(entrada_con_bias, w[p].T)
        a = sigmoide(v)
    return a * ESCALA_SIGMOIDE


def filtrar_ewma(e, alpha):
    """Filtro recursivo EWMA identico al de la Etapa 5."""
    e = np.asarray(e, dtype=np.float64)
    s = np.empty_like(e)
    s[0] = e[0]
    inercia = 1.0 - alpha
    for t in range(1, len(e)):
        s[t] = alpha * e[t] + inercia * s[t - 1]
    return s.astype(np.float32)


pred_val = predecir_lote(X_val, w)
err_val = np.mean(np.abs(Y_val_real - pred_val), axis=1)

pred_test = predecir_lote(X_test, w)
err_test = np.mean(np.abs(Y_test_real - pred_test), axis=1)
ewma_test = filtrar_ewma(err_test, ALPHA_EWMA)

# usamos el mismo umbral de tolerancia de error que la LSTM
# para evaluar ambos modelos bajo la misma exigencia fisica.
with open(os.path.join(DIR_EWMA, "config_ewma.json")) as f:
    cfg_ewma = json.load(f)

mu0_mlp = float(err_val.mean())
sigma0_mlp = float(err_val.std())
umbral_mlp = float(cfg_ewma["umbral"])

print("\n=== Estadisticos del MLP en Validacion y Umbral Comun ===")
print(f"  Media normal MLP (mu_0)      = {mu0_mlp:.4f} (vs {cfg_ewma['mu0']:.4f} en LSTM)")
print(f"  Desvio normal MLP (sigma_0)  = {sigma0_mlp:.4f} (vs {cfg_ewma['sigma0']:.4f} en LSTM)")
print(f"  Umbral comun de comparacion  = {umbral_mlp:.4f}")



def calcular_metricas(y_real, y_pred):
    """Calcula TP, FP, TN, FN, Precision, Sensibilidad (Recall) y F1-Score."""
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


test_lstm = pd.read_pickle(os.path.join(DIR_EWMA, "ewma_test.pkl"))
y_real = test_lstm["anomalia"].values.astype(np.int8)

alarma_mlp_cruda = (err_test > umbral_mlp).astype(np.int8)
alarma_mlp_ewma = (ewma_test > umbral_mlp).astype(np.int8)

met_mlp_crudo = calcular_metricas(y_real, alarma_mlp_cruda)
met_mlp_ewma = calcular_metricas(y_real, alarma_mlp_ewma)

print("\n=== Metricas del MLP en Test ===")
print(f"{'Metodo':<20} | {'TP':>6} | {'FP':>6} | {'TN':>7} | {'FN':>5} | "
      f"{'Precision':>9} | {'Sensibil.':>9} | {'F1-Score':>8}")
print("-" * 88)
for nombre, m in [("MLP Sin EWMA", met_mlp_crudo), ("MLP Con EWMA", met_mlp_ewma)]:
    print(f"{nombre:<20} | {m['TP']:>6,} | {m['FP']:>6,} | {m['TN']:>7,} | {m['FN']:>5,} | "
          f"{m['precision']*100:>8.2f}% | {m['sensibilidad']*100:>8.2f}% | {m['f1']*100:>7.2f}%")


df_mlp_test = pd.DataFrame({
    "err_medio": err_test,
    "ewma_medio": ewma_test,
    "alarma_cruda": alarma_mlp_cruda,
    "alarma": alarma_mlp_ewma,
    "anomalia": test_lstm["anomalia"].values,
}, index=test_lstm.index)
df_mlp_test.to_pickle(os.path.join(DIR_SALIDA, "mlp_test.pkl"))

resumen_mlp = {
    "arquitectura": CAPAS,
    "retardos": RETARDOS,
    "vel_aprendizaje": VEL_APRENDIZAJE,
    "epocas": MAX_EPOCAS,
    "mse_val": historial_mse_val[-1],
    "mu0": mu0_mlp,
    "sigma0": sigma0_mlp,
    "umbral": umbral_mlp,
    "mlp_sin_ewma": met_mlp_crudo,
    "mlp_con_ewma": met_mlp_ewma,
}
with open(os.path.join(DIR_SALIDA, "metricas_mlp.json"), "w") as f:
    json.dump(resumen_mlp, f, indent=2)


# grafico de curva de aprendizaje del MLP
plt.figure(figsize=(7.5, 4.2))
epocas_eje = np.arange(1, MAX_EPOCAS + 1)
plt.plot(epocas_eje, historial_mse_train, "o-", label="MSE Train (MLP)")
plt.plot(epocas_eje, historial_mse_val, "s-", label="MSE Validacion (MLP)")
plt.axhline(0.0472, color="green", ls="--", label="Mejor MSE Validacion LSTM (0.0472)")
plt.xlabel("Epoca")
plt.ylabel("Error Cuadratico Medio (MSE)")
plt.title("Curva de aprendizaje del MLP (programado a mano) vs. referencia LSTM")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIR_SALIDA, "fig_mlp_curva_aprendizaje.png"), dpi=120)

# grafico de matriz de confusion y metricas en test (crudo vs ewma)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

matriz_conf = np.array([
    [met_mlp_ewma["TN"], met_mlp_ewma["FP"]],
    [met_mlp_ewma["FN"], met_mlp_ewma["TP"]],
])
im = ax1.imshow(np.log10(matriz_conf + 1), cmap="Oranges", vmin=2, vmax=6.2)
etiquetas_celda = [
    [f"TN (verdadero neg.)\n{met_mlp_ewma['TN']:,}\n({met_mlp_ewma['TN']/len(y_real)*100:.2f}%)",
     f"FP (falsa alarma)\n{met_mlp_ewma['FP']:,}\n({met_mlp_ewma['FP']/len(y_real)*100:.2f}%)"],
    [f"FN (falla omitida)\n{met_mlp_ewma['FN']:,}\n({met_mlp_ewma['FN']/len(y_real)*100:.2f}%)",
     f"TP (falla detectada)\n{met_mlp_ewma['TP']:,}\n({met_mlp_ewma['TP']/len(y_real)*100:.2f}%)"],
]
for i in range(2):
    for j in range(2):
        color_txt = "white" if (i == 0 and j == 0) else "black"
        ax1.text(j, i, etiquetas_celda[i][j], ha="center", va="center",
                 fontsize=10, fontweight="bold", color=color_txt)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["predicho: 0 (normal)", "predicho: 1 (anomalia)"])
ax1.set_yticks([0, 1])
ax1.set_yticklabels(["real: 0 (normal)", "real: 1 (anomalia)"])
ax1.set_title("matriz de confusion en test - MLP (con EWMA)")

nombres_m = ["Precision", "Sensibilidad (Recall)", "F1-Score"]
vals_crudo = [met_mlp_crudo["precision"] * 100, met_mlp_crudo["sensibilidad"] * 100, met_mlp_crudo["f1"] * 100]
vals_ewma = [met_mlp_ewma["precision"] * 100, met_mlp_ewma["sensibilidad"] * 100, met_mlp_ewma["f1"] * 100]

x = np.arange(len(nombres_m))
ancho = 0.35
b1 = ax2.bar(x - ancho / 2, vals_crudo, ancho, label="sin EWMA (error crudo)", color="navajowhite", edgecolor="peru")
b2 = ax2.bar(x + ancho / 2, vals_ewma, ancho, label=f"con EWMA (alpha={ALPHA_EWMA}, k={K_UMBRAL})", color="darkorange", edgecolor="saddlebrown")

for barras in [b1, b2]:
    for b in barras:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(nombres_m)
ax2.set_ylabel("porcentaje (%)")
ax2.set_ylim(0, 110)
ax2.set_title("impacto del filtrado EWMA en metricas (MLP)")
ax2.legend(loc="lower right")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
ruta_fig_matriz = os.path.join(DIR_SALIDA, "fig_mlp_matriz_confusion.png")
plt.savefig(ruta_fig_matriz, dpi=120)
print(f"Matriz de confusion guardada en {ruta_fig_matriz}")

print(f"\nResultados del MLP guardados en {DIR_SALIDA}")
plt.show()
