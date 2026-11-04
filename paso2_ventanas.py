import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ventanas import indices_validos, armar_ventanas

# 0) CONFIGURACION
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(DIR_BASE, "datos_procesados")
DIR_SALIDA = os.path.join(DIR_DATOS, "ventanas")
os.makedirs(DIR_SALIDA, exist_ok=True)
canales_objetivo = [f"channel_{i}" for i in range(41, 47)]
canal_contexto = "channel_1"
canales = canales_objetivo + [canal_contexto]   # orden de columnas: 6 target + contexto
numeros_target = len(canales_objetivo)

L = 80   # largo de la historia (se justifica con la autocorrelacion del punto 2:
# el ciclo dominante dura 33 pasos, asi que 80 pasos ven ~2.4 ciclos completos)

# 1) CARGA
datos = pd.read_pickle(os.path.join(DIR_DATOS, "datos_alineados.pkl"))
print(f"Tabla cargada: {datos.shape[0]:,} instantes")

# Filas "sanas de Train": las unicas que pueden usarse para aprender
# cualquier cosa (escalado incluido). Se excluye el instante de id_93.
sanas_train = (datos["conjunto"] == "train") & ~datos["hueco"] & ~datos["anomalia"]

# 2) ELECCION DE L: ¿CADA CUANTO SE REPITE LA SENAL?
# La autocorrelacion mide cuanto se parece la senal a si misma desplazada 'lag'pasos.Si tiene un pico en lag = P, la senal se repite cada P pasos.
x = datos.loc[sanas_train, "channel_41"].values[:20000] #se toma una porción de 200k de instantes
x = x - x.mean() #centro la señal en 0
ac = np.correlate(x, x, mode="full")[len(x) - 1:]
ac = ac / ac[0]
cantidad_pasos = 300 # este 300 representa el desplazamiento temp max p buscar la periodicidad
# El primer pico despues de que la autocorrelacion se vuelve negativa es el periodo
primer_negativo = int(np.argmax(ac < 0))
periodo = primer_negativo + int(np.argmax(ac[primer_negativo:cantidad_pasos]))
print(f"\n=== 2) Ciclo dominante: {periodo} pasos = {periodo*90/60:.0f} min "
      f"-> L = {L} pasos = {L*90/60:.0f} min ({L/periodo:.1f} ciclos)")

plt.figure(figsize=(10, 3.5))
plt.plot(ac[:cantidad_pasos])
plt.axvline(periodo, color="red", ls="--", label=f"periodo = {periodo} pasos")
plt.axvline(L, color="green", ls="--", label=f"L elegido = {L}")
plt.xlabel("lag (pasos de 90 s)")
plt.ylabel("autocorrelacion")
plt.title("Autocorrelacion de channel_41 (Train)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
ruta_fig3 = os.path.join(DIR_SALIDA, "fig3_autocorrelacion.png")
plt.savefig(ruta_fig3, dpi=120)
plt.show()
print(f"Guardado: {ruta_fig3}")

# 3) ESCALADO (ajustado SOLO con filas sanas de Train)
print("\n=== 3) Escalado ===")
escalado = {} #diccionario dde guardamos los parametros de cada canal
matriz = np.empty((len(datos), len(canales)), dtype=np.float32)
for j, c in enumerate(canales): #recorremos los canales
    ref = datos.loc[sanas_train, c] #usamos solo las filas de entrenamiento sano del canal
    if c == canal_contexto:
        a, b = float(ref.min()), float(ref.max() - ref.min())
        metodo = "minmax" #para el canal 1 usamos min max, porq es un canal q esta fijo la mayor parte del tiempo en un valor cte
    else:
        a, b = float(ref.mean()), float(ref.std())
        metodo = "zscore" #para los target usamos el zscore
    escalado[c] = {"metodo": metodo, "centro": a, "escala": b}
    matriz[:, j] = (datos[c].values - a) / b
    print(f"{c:>11}: {metodo:6} centro={a:.5f} escala={b:.6f} | "
          f"Train -> [{matriz[sanas_train.values, j].min():6.2f}, "
          f"{matriz[sanas_train.values, j].max():6.2f}] | "
          f"Test -> [{np.nanmin(matriz[(datos['conjunto'] == 'test').values, j]):7.2f}, "
          f"{np.nanmax(matriz[(datos['conjunto'] == 'test').values, j]):7.2f}]")

# 4) INSTANTES VALIDOS POR CONJUNTO
print(f"\n=== 4) Ventanas validas (L = {L}) ===")
conj = datos["conjunto"].values
hueco = datos["hueco"].values
anom = datos["anomalia"].values

idx_limpio = indices_validos(hueco | anom, L)     # para Train/Val
idx_sin_hueco = indices_validos(hueco, L)         # para Test

idx = {"train": idx_limpio[conj[idx_limpio] == "train"],"val": idx_limpio[conj[idx_limpio] == "val"],
"test": idx_sin_hueco[conj[idx_sin_hueco] == "test"],}
for nombre, ind in idx.items():
    total = (conj == nombre).sum()
    print(f"{nombre:>5}: {len(ind):>7,} ventanas de {total:>7,} instantes "
          f"({total - len(ind):,} descartados)")
    np.save(os.path.join(DIR_SALIDA, f"idx_{nombre}.npy"), ind)


# 5) ARMADO DE TENSORES 3D Y VERIFICACION
print("\n=== 5) Tensores ===")
X_train, y_train = armar_ventanas(matriz, idx["train"], L, numeros_target)
X_val, y_val = armar_ventanas(matriz, idx["val"], L, numeros_target)

for nombre, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
    print(f"X_{nombre}.shape = {X.shape}  (muestras, L, canales) | "
          f"y_{nombre}.shape = {y.shape}  (muestras, canales target) | "
          f"{X.nbytes/1e6:.0f} MB")

# Chequeos automaticos: si algo esta mal armado, el script se detiene aca
assert not np.isnan(X_train).any() and not np.isnan(X_val).any(), "Hay NaN en X"
i, t = 1234, idx["train"][1234]
assert np.allclose(X_train[i, -1], matriz[t - 1]), "La ultima fila de X no es t-1"
assert np.allclose(X_train[i, 0], matriz[t - L]), "La primera fila de X no es t-L"
assert np.allclose(y_train[i], matriz[t, :numeros_target]), "y no es la fila t (target)"
print("Chequeos OK: sin NaN, X = filas [t-L, t-1], y = fila t de los 6 target")

# Guardado de tensores de entrenamiento y validacion
np.save(os.path.join(DIR_SALIDA, "X_train.npy"), X_train)
np.save(os.path.join(DIR_SALIDA, "y_train.npy"), y_train)
np.save(os.path.join(DIR_SALIDA, "X_val.npy"), X_val)
np.save(os.path.join(DIR_SALIDA, "y_val.npy"), y_val)
print(f"(Test: {len(idx['test']):,} ventanas, se armaran por lotes en la Etapa 4)")

# 6) GRAFICO: ¿QUE VE LA RED EN UNA MUESTRA?
fig, axes = plt.subplots(len(canales), 1, figsize=(12, 11), sharex=True)
pasos = np.arange(-L, 0)
for j, (ax, c) in enumerate(zip(axes, canales)):
    ax.plot(pasos, X_train[i, :, j], ".-", ms=3, label="entrada X (historia)")
    if j < numeros_target:
        ax.plot(0, y_train[i, j], "r*", ms=12, label="salida y (a predecir)")
    ax.set_ylabel(c.replace("channel_", "ch"), rotation=0, labelpad=20)
    ax.grid(alpha=0.3)
axes[0].legend(loc="upper left", fontsize=8)
axes[0].set_title(f"Muestra {i} de Train (t = {datos.index[t]}): "
                  f"{L} pasos x 7 canales -> 6 valores en t")
axes[-1].set_xlabel("pasos relativos a t (cada paso = 90 s)")
plt.tight_layout()
ruta_fig4 = os.path.join(DIR_SALIDA, "fig4_ejemplo_ventana.png")
plt.savefig(ruta_fig4, dpi=120)
plt.close()
print(f"Guardado: {ruta_fig4}")

# 7) GUARDADO DE CONFIGURACION
np.save(os.path.join(DIR_SALIDA, "matriz_escalada.npy"), matriz)
config = {"L": L, "canales": canales, "canales_target": canales_objetivo,
          "periodo_ciclo_pasos": periodo, "escalado": escalado}
with open(os.path.join(DIR_SALIDA, "config_ventanas.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\n=== 7) Guardado final exitoso en {DIR_SALIDA} ===")
