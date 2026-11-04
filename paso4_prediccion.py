"""
TP Final - Etapa 4: Prediccion en Test y calculo del error residual.

Que hace este script:
  1. Carga el modelo LSTM entrenado en la Etapa 3 (modelo_lstm.pt).
  2. Carga la matriz escalada completa y los indices validos de Test (idx_test.npy).
  3. Procesa el conjunto de Test por lotes (batches) de forma eficiente:
       - Arma las ventanas de 80 pasos al vuelo (sin saturar la memoria RAM).
       - Genera la prediccion un paso adelante (y_hat) para los 6 canales target.
  4. Calcula la matriz de error residual instantaneo:
       e(t) = |y(t) - y_hat(t)|  para cada canal.
  5. Sincroniza los errores con las marcas de anomalias reales de labels.csv.
  6. Grafica la verificacion visual del detector:
       - Comparacion Senal Real vs. Prediccion LSTM.
       - Evolucion del error residual e(t) frente a las franjas de anomalia.

Salidas (datos_procesados/predicciones/):
  - error_test.npy              (N_test, 6) matriz de errores residuales e(t)
  - y_test_real.npy             (N_test, 6) valores reales observados en t
  - y_test_pred.npy             (N_test, 6) valores predichos por la LSTM
  - timestamps_test.npy         Fechas/horas correspondientes a cada prediccion
  - fig6_prediccion_vs_real.png Grafico detallado sobre un evento anomalo real
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from ventanas import armar_ventanas

# ---------------------------------------------------------------------------
# 0) CONFIGURACION Y RUTAS
# ---------------------------------------------------------------------------
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(DIR_BASE, "datos_procesados")
DIR_VENTANAS = os.path.join(DIR_DATOS, "ventanas")
DIR_MODELO = os.path.join(DIR_DATOS, "modelo")
DIR_SALIDA = os.path.join(DIR_DATOS, "predicciones")
os.makedirs(DIR_SALIDA, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo para inferencia: {DEVICE}")

# Cargar configuracion de ventanas generada en la Etapa 2
with open(os.path.join(DIR_VENTANAS, "config_ventanas.json"), "r") as f:
    config = json.load(f)

L = config["L"]
CANALES = config["canales"]
CANALES_TARGET = config["canales_target"]
N_TARGET = len(CANALES_TARGET)
BATCH_SIZE = 256   # Lote mas grande para que la inferencia sea ultrarrapida

# 1) DEFINICION Y CARGA DEL MODELO ENTRENADO
class PredictorLSTM(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, output_dim=6, dropout=0.2):
        super(PredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        pred = self.fc(out[:, -1, :])
        return pred

ruta_pesos = os.path.join(DIR_MODELO, "modelo_lstm.pt")
model = PredictorLSTM().to(DEVICE)

# 2) CARGA DE DATOS Y PARTICION DE TEST
print("\n=== 2) Carga de datos de Test ===")
matriz = np.load(os.path.join(DIR_VENTANAS, "matriz_escalada.npy"))
idx_test = np.load(os.path.join(DIR_VENTANAS, "idx_test.npy"))
datos_completos = pd.read_pickle(os.path.join(DIR_DATOS, "datos_alineados.pkl"))

N_test = len(idx_test)
print(f"Total de ventanas a evaluar en Test: {N_test:,} instantes")



# 3) INFERENCIA POR LOTES (PREDICCION PASO A PASO)
print(f"\n=== 3) Generando predicciones por lotes (batch_size={BATCH_SIZE}) ===")
y_pred_list = []
y_real_list = []

t_inicio = time.time()

with torch.no_grad():
    for start in range(0, N_test, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N_test)
        indices_batch = idx_test[start:end]
        # Armar las ventanas 3D de este lote
        X_batch, y_batch = armar_ventanas(matriz, indices_batch, L, N_TARGET)
        tensor_x = torch.from_numpy(X_batch).to(DEVICE)
        pred = model(tensor_x)
        y_pred_list.append(pred.cpu().numpy())
        y_real_list.append(y_batch)
        
        if (start // BATCH_SIZE) % 200 == 0 and start > 0:
            progreso = (end / N_test) * 100
            print(f"  Progreso: {progreso:5.1f}% ({end:,}/{N_test:,} ventanas)")

y_test_pred = np.vstack(y_pred_list)
y_test_real = np.vstack(y_real_list)
duracion = time.time() - t_inicio
print(f"Inferencia completada en {duracion:.1f} segundos ({len(y_test_pred)/(duracion):.0f} ventanas/seg).")


# 4) CALCULO DE LA SENAL DE ERROR RESIDUAL e(t)
# Error instantaneo absoluto e(t) = |y(t) - y_hat(t)|
error_test = np.abs(y_test_real - y_test_pred)

print("\n=== 4) Estadisticas del error residual e(t) en Test ===")
for j, c in enumerate(CANALES_TARGET):
    print(f"{c:>11}: Error medio = {error_test[:, j].mean():.4f} | "
          f"Desvio = {error_test[:, j].std():.4f} | Maximo = {error_test[:, j].max():.4f}")

# Timestamps correspondientes a las predicciones
timestamps_test = datos_completos.index[idx_test].to_numpy()


# 5) GUARDADO DE RESIDUOS Y PREDICCIONES

np.save(os.path.join(DIR_SALIDA, "error_test.npy"), error_test)
np.save(os.path.join(DIR_SALIDA, "y_test_real.npy"), y_test_real)
np.save(os.path.join(DIR_SALIDA, "y_test_pred.npy"), y_test_pred)
np.save(os.path.join(DIR_SALIDA, "timestamps_test.npy"), timestamps_test)
print(f"\nArchivos guardados en: {DIR_SALIDA}")

# 6) VERIFICACION VISUAL: COMPARACION REAL VS. PREDICHO EN ANOMALIA

print("\n=== 6) Generando grafico de verificacion sobre evento anomalo ===")

# Vamos a graficar un zoom alrededor del evento real id_96
eventos = pd.read_csv(os.path.join(DIR_DATOS, "eventos_2004_2005.csv"), index_col="ID")
ev = eventos.loc["id_96"]
t_ini_ev = pd.Timestamp(ev["StartTime"])
t_fin_ev = pd.Timestamp(ev["EndTime"])

# Tomamos 2 dias antes y 2 dias despues para ver el cambio de regimen
margen = pd.Timedelta("2D")
t_inicio_zoom = t_ini_ev - margen
t_fin_zoom = t_fin_ev + margen

# Mascara sobre las predicciones de Test
mascara_zoom = (timestamps_test >= t_inicio_zoom) & (timestamps_test <= t_fin_zoom)
tiempo_zoom = timestamps_test[mascara_zoom]

# Graficamos channel_41 (el canal target principal)
idx_ch41 = 0
real_zoom = y_test_real[mascara_zoom, idx_ch41]
pred_zoom = y_test_pred[mascara_zoom, idx_ch41]
err_zoom  = error_test[mascara_zoom, idx_ch41]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

# Panel Superior: Telemetria Real vs. Prediccion LSTM
ax1.plot(tiempo_zoom, real_zoom, label="Real (y_t)", color="tab:blue", lw=1.0)
ax1.plot(tiempo_zoom, pred_zoom, label="Prediccion LSTM (y_hat_t)", color="tab:orange", lw=1.0, ls="--")
ax1.axvspan(t_ini_ev, t_fin_ev, color="red", alpha=0.3, label=f"Anomalia real ({ev['Category']})")
ax1.set_ylabel("Valor escalado (Z-Score)")
ax1.set_title(f"Canal 41 - Seguimiento del Modelo vs. Telemetria Real (Evento id_96)")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)

# Panel Inferior: Error residual e(t)
ax2.plot(tiempo_zoom, err_zoom, label="Error instantaneo |y - y_hat|", color="black", lw=1.0)
ax2.axvspan(t_ini_ev, t_fin_ev, color="red", alpha=0.3, label="Periodo etiquetado")
ax2.set_xlabel("Fecha y Hora")
ax2.set_ylabel("Error residual e(t)")
ax2.set_title("Senal de Discrepancia Instantanea")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)

plt.tight_layout()
ruta_fig6 = os.path.join(DIR_SALIDA, "fig6_prediccion_vs_real.png")
plt.savefig(ruta_fig6, dpi=120)
print(f"Grafico guardado: {ruta_fig6}")
print("Etapa 4 lista para ser ejecutada una vez entrenado el modelo.")
plt.show()
