import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ventanas import indices_validos, armar_ventanas

# 0) configuracion y rutas principales
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(DIR_BASE, "datos_procesados")
DIR_SALIDA = os.path.join(DIR_DATOS, "ventanas")
os.makedirs(DIR_SALIDA, exist_ok=True)

canales_objetivo = [f"channel_{i}" for i in range(41, 47)]
canal_contexto = "channel_1"
canales = canales_objetivo + [canal_contexto]  # orden de columnas: 6 target + contexto
numeros_target = len(canales_objetivo)

L = 80  # largo de la historia temporal (justificado por autocorrelacion: ciclo dominante de 33 pasos, 80 cubre ~2.4 ciclos)

# 1) carga de datos alineados y seleccion de filas sanas
datos = pd.read_pickle(os.path.join(DIR_DATOS, "datos_alineados.pkl"))
print(f"Tabla cargada: {datos.shape[0]:,} instantes")

# filas sanas de train: las unicas autorizadas p ajustar parametros y escaladores
sanas_train = (datos["conjunto"] == "train") & ~datos["hueco"] & ~datos["anomalia"]

# 2) eleccion de L mediante autocorrelacion y periodicidad ciclica
# la autocorrelacion evalua la similitud de la serie consigo misma desplazada 'lag' pasos
x = datos.loc[sanas_train, "channel_41"].values[:20000]  # recorte de 20.000 instantes sanos
x = x - x.mean()  # centrado en cero
ac = np.correlate(x, x, mode="full")[len(x) - 1:]
ac = ac / ac[0]
cantidad_pasos = 300  # desplazamiento temporal maximo analizado

# el primer pico positivo posterior al primer cruce por cero define el periodo del ciclo
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
plt.title("Autocorrelacion de channel_41 (Train sano)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
ruta_fig3 = os.path.join(DIR_SALIDA, "fig3_autocorrelacion.png")
plt.savefig(ruta_fig3, dpi=120)
print(f"Guardado: {ruta_fig3}")
plt.show()

# 3) escalado estadistico ajustado exclusivamente con datos sanos de train
print("\n=== 3) Escalado estadistico ===")
escalado = {}  # parametros aprendidos por canal
matriz = np.empty((len(datos), len(canales)), dtype=np.float32)
for j, c in enumerate(canales):
    ref = datos.loc[sanas_train, c]  # referencia sana de entrenamiento
    if c == canal_contexto:
        a, b = float(ref.min()), float(ref.max() - ref.min())
        metodo = "minmax"  # canal 1 casi constante gran parte del tiempo, conviene minmax
    else:
        a, b = float(ref.mean()), float(ref.std())
        metodo = "zscore"  # canales 41 a 46 siguen distribucion normal, zscore
    escalado[c] = {"metodo": metodo, "centro": a, "escala": b}
    matriz[:, j] = (datos[c].values - a) / b
    print(f"{c:>11}: "
          f"Train [{matriz[sanas_train.values, j].min():6.2f}, "
          f"{matriz[sanas_train.values, j].max():6.2f}] | "
          f"Test [{np.nanmin(matriz[(datos['conjunto'] == 'test').values, j]):7.2f}, "
          f"{np.nanmax(matriz[(datos['conjunto'] == 'test').values, j]):7.2f}]")

# 4) seleccion de instantes y ventanas validas por conjunto
print(f"\n=== 4) Ventanas validas (L = {L}) ===")
conj = datos["conjunto"].values
hueco = datos["hueco"].values
anom = datos["anomalia"].values

idx_limpio = indices_validos(hueco | anom, L)  # ventanas sin huecos ni anomalias p train y val
idx_sin_hueco = indices_validos(hueco, L)      # ventanas sin huecos p test (incluye anomalias)

idx = {
    "train": idx_limpio[conj[idx_limpio] == "train"],
    "val": idx_limpio[conj[idx_limpio] == "val"],
    "test": idx_sin_hueco[conj[idx_sin_hueco] == "test"],
}

for nombre, ind in idx.items():
    total = (conj == nombre).sum()
    print(f"{nombre:>5}: {len(ind):>7,} ventanas de {total:>7,} instantes "
          f"({total - len(ind):,} descartados)")
    np.save(os.path.join(DIR_SALIDA, f"idx_{nombre}.npy"), ind)

# 5) armado y verificacion de tensores supervisados 3D
print("\n=== 5) Armado de tensores supervisados ===")
x_train, y_train = armar_ventanas(matriz, idx["train"], L, numeros_target)
x_val, y_val = armar_ventanas(matriz, idx["val"], L, numeros_target)

# verificacion de coherencia y ausencia de valores nulos
assert not np.isnan(x_train).any() and not np.isnan(x_val).any(), "hay nan en x"
i, t = 1234, idx["train"][1234]
assert np.allclose(x_train[i, -1], matriz[t - 1]), "la ultima fila de x no es t-1"
assert np.allclose(x_train[i, 0], matriz[t - L]), "la primera fila de x no es t-L"
assert np.allclose(y_train[i], matriz[t, :numeros_target]), "y no coincide con el target en t"
print("Verificaciones OK: sin NaN, x = [t-L, t-1], y = target en instante t")

# 6) guardado de tensores, indices y configuracion
# guardamos con nombre estandar p maxima compatibilidad con scripts de mlp y otros modelos
np.save(os.path.join(DIR_SALIDA, "X_train.npy"), x_train)
np.save(os.path.join(DIR_SALIDA, "y_train.npy"), y_train)
np.save(os.path.join(DIR_SALIDA, "X_val.npy"), x_val)
np.save(os.path.join(DIR_SALIDA, "y_val.npy"), y_val)
np.save(os.path.join(DIR_SALIDA, "matriz_escalada.npy"), matriz)

config = {
    "L": L,
    "canales": canales,
    "canales_target": canales_objetivo,
    "periodo_ciclo_pasos": periodo,
    "escalado": escalado,
}
with open(os.path.join(DIR_SALIDA, "config_ventanas.json"), "w") as f:
    json.dump(config, f, indent=2)

print("\n=== 6) Guardado de tensores y configuracion exitoso ===")
print(f"Archivos guardados en: {DIR_SALIDA}")
