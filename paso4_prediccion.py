import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from modelo import cargar_modelo

# 0) configuracion y rutas principales
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(DIR_BASE, "datos_procesados")
DIR_VENT = os.path.join(DIR_DATOS, "ventanas")
DIR_MODELO = os.path.join(DIR_DATOS, "modelo")
DIR_SALIDA = os.path.join(DIR_DATOS, "predicciones")
os.makedirs(DIR_SALIDA, exist_ok=True)

tam_parte = 4096  # cantidad de ventanas evaluadas por bloque en la gpu
eventos_zoom = ["id_96", "id_10", "id_39"]  # sutil, caida brusca y evento raro corto

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("dispositivo p inferencia:", dispositivo,
      f"({torch.cuda.get_device_name(0)})" if dispositivo.type == "cuda" else "")

# 1) carga de configuracion, datos alineados y modelo entrenado
with open(os.path.join(DIR_VENT, "config_ventanas.json")) as f:
    cfg = json.load(f)

L = cfg["L"]
canales_target = cfg["canales_target"]
cant_target = len(canales_target)
num = [c.split("_")[1] for c in canales_target]  # "41", "42", ...

matriz = np.load(os.path.join(DIR_VENT, "matriz_escalada.npy"))
datos = pd.read_pickle(os.path.join(DIR_DATOS, "datos_alineados.pkl"))
eventos = pd.read_csv(os.path.join(DIR_DATOS, "eventos_2004_2005.csv"),index_col="ID", parse_dates=["StartTime", "EndTime"])

# cargamos el modelo 
ruta_modelo = os.path.join(DIR_MODELO, "modelo_lstm.pt")
modelo, ckpt = cargar_modelo(ruta_modelo, dispositivo)

# 2) prediccion optimizada por partes en gpu usando unfold
# subimos la matriz 2d a gpu y creamos una vista deslizante sin duplicar memoria en ram
matriz_t = torch.from_numpy(matriz).to(dispositivo)
vista_t = matriz_t.unfold(0, L, 1).permute(0, 2, 1)  # (n - L + 1, L, cant_canales)

def predecir(indices):
    # para cada instante t devuelve (real, prediccion), ambos de dimension (n, 6)
    #resto indices - L para apuntar a la ventana q corresponde al instante t
    idx_gpu = torch.from_numpy(indices - L).to(dispositivo)
    preds = []
    with torch.no_grad():
        for i in range(0, len(idx_gpu), tam_parte):
            xb = vista_t[idx_gpu[i:i + tam_parte]] #recorto en gpu
            y_hat = modelo(xb).cpu().numpy() #se procesan las ventanas en la lstm y 
            #se hacen las predicciones
            preds.append(y_hat) #guardo la prediccion
    real = matriz[indices, :cant_target] #me traigo los valores reales
    return real.astype(np.float32), np.concatenate(preds) #junto las predicciones con los reales

def armar_tabla(indices, real, pred):
    # arma el dataframe indexado por fecha p alinear el residuo al instante temporal exacto
    err = np.abs(real - pred)  # residuo instantaneo absoluto por canal
    tabla = pd.DataFrame(index=datos.index[indices]) #df con indices fechas correspondientes a indices
    for j, n in enumerate(num): #recorro los canales target
        tabla[f"real_{n}"] = real[:, j] #guardo el valor real
        tabla[f"pred_{n}"] = pred[:, j] #guardo la prediccion
        tabla[f"err_{n}"] = err[:, j] #guardo el error
    # promedio de los 6 canales escalados
    tabla["err_medio"] = err.mean(axis=1)
    # incorporamos etiquetas reales de la etapa 1
    cols_etiq = [f"etiq_{n}" for n in num] + ["anomalia"]
    tabla[cols_etiq] = datos[cols_etiq].values[indices]
    return tabla, err

print("\n=== 2) Generacion de predicciones (Val y Test) ===")
tablas = {}
residuos_guardados = {}
predicciones_guardadas = {}
reales_guardados = {}

for nombre in ["val", "test"]: #ejecuto para val y test
    idx = np.load(os.path.join(DIR_VENT, f"idx_{nombre}.npy")) #cargo pos validas 
    #serian las q guarde idx_val y idx_test en la 2da vuelta
    real, pred = predecir(idx)
    tablas[nombre], err = armar_tabla(idx, real, pred)
    residuos_guardados[nombre] = err
    predicciones_guardadas[nombre] = pred
    reales_guardados[nombre] = real
    # guardamos dataframe indexado en pkl
    ruta_pkl = os.path.join(DIR_SALIDA, f"pred_{nombre}.pkl")
    tablas[nombre].to_pickle(ruta_pkl)
    print(f"{nombre:>5}: {len(idx):,} predicciones | MSE = {np.mean((real - pred) ** 2):.4f}")

# guardado compatible en arrays npy p abastecer los analisis siguientes (ewma, metricas y som)
np.save(os.path.join(DIR_SALIDA, "error_val.npy"), residuos_guardados["val"])
np.save(os.path.join(DIR_SALIDA, "error_test.npy"), residuos_guardados["test"])
np.save(os.path.join(DIR_SALIDA, "y_test_real.npy"), reales_guardados["test"])
np.save(os.path.join(DIR_SALIDA, "y_test_pred.npy"), predicciones_guardadas["test"])
idx_test = np.load(os.path.join(DIR_VENT, "idx_test.npy"))
timestamps_test = datos.index[idx_test].to_numpy()
np.save(os.path.join(DIR_SALIDA, "timestamps_test.npy"), timestamps_test)
val, test = tablas["val"], tablas["test"]

# 3) error en regimen normal (val) vs error en eventos etiquetados (test)
# val representa el regimen sano no visto: nos da la linea base contra la q comparamos
mu0 = val["err_medio"].mean() #error promedio en reg sano
sigma0 = val["err_medio"].std() #desvio en reg sano
p99 = val["err_medio"].quantile(0.99) #p 99 del error en reg sano
print(f"\n=== 3) Error medio en Val (normal): mu0 = {mu0:.3f} | sigma0 = {sigma0:.3f} | ")

normal_test = test.loc[~test["anomalia"], "err_medio"] #filtro el periodo donde no hay anomalias
print(f"Test fuera de eventos: media = {normal_test.mean():.3f} | "
      f"{(normal_test > p99).mean() * 100:.1f}% de instantes superan el p99 de Val")

filas_resumen = []
for ev_id, ev in eventos.iterrows():
    tramo = test.loc[ev["StartTime"]:ev["EndTime"], "err_medio"] #recorta los errores de prediccion
    #q cayeron entre la hora de inicio y de fin de ese evento
    if len(tramo) == 0:  
        continue
    filas_resumen.append({
        "ID": ev_id,
        "Category": ev["Category"], # si es falla o evento raro
        "inicio": ev["StartTime"].date(), # fecha de inicio
        "horas": ev["duracion_h"], # duracion en horas
        "instantes": len(tramo), # instantes dentro del evento
        "err_medio": float(tramo.mean()), # error promedio
        "err_max": float(tramo.max()), # error maximo
        "pct_sobre_p99": (tramo > p99).mean() * 100 # % de instantes por encima del p99
    })

resumen = pd.DataFrame(filas_resumen).set_index("ID")
ruta_resumen = os.path.join(DIR_SALIDA, "resumen_eventos.csv")
resumen.to_csv(ruta_resumen)
print("\nError dentro de cada evento del Test:")
print(resumen.round(2).to_string())

# 4) graficos de diagnostico visual
color_cat = {"Anomaly": "red", "Rare Event": "orange"}

def sombrear(ax_grafico, desde=None, hasta=None):
    # sombrea las franjas temporales de fallas y eventos raros
    for _, ev in eventos.iterrows():
        if desde is not None and (ev["EndTime"] < desde or ev["StartTime"] > hasta):
            continue
        ax_grafico.axvspan(ev["StartTime"], ev["EndTime"],
                           color=color_cat.get(ev["Category"], "gray"), alpha=0.3, lw=0)

# figura 7: error a lo largo de todo el conjunto test (maximo por hora en escala logaritmica)
por_hora = test["err_medio"].resample("1h").max()
fig, ax = plt.subplots(figsize=(15, 4.5))
ax.plot(por_hora.index, por_hora, lw=0.6, color="tab:blue", label="error medio (max por hora)")
ax.axhline(p99, color="green", ls="--", label=f"p99 de Val = {p99:.2f}")
sombrear(ax)
ax.set_yscale("log")
ax.set_ylabel("|y - y_hat| (escala log)")
ax.set_title("Error de prediccion en Test | rojo = Anomaly, naranja = Rare Event")
ax.legend(loc="upper left")
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
ruta_fig7 = os.path.join(DIR_SALIDA, "fig7_error_test.png")
plt.savefig(ruta_fig7, dpi=120)
print(f"\nGrafico guardado: {ruta_fig7}")
plt.show()

# figura 8: zoom sobre eventos seleccionados (real vs prediccion y residuos multivariados)
def zoom(ev_id, margen="12h"):
    ev = eventos.loc[ev_id]
    desde = ev["StartTime"] - pd.Timedelta(margen)
    hasta = ev["EndTime"] + pd.Timedelta(margen)
    tramo = test.loc[desde:hasta]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    # canales 41 y 43 de referencia
    for ax_sub, n in zip(axes[:2], ["41", "43"]):
        ax_sub.plot(tramo.index, tramo[f"real_{n}"], lw=0.7, label="real")
        ax_sub.plot(tramo.index, tramo[f"pred_{n}"], lw=0.7, label="LSTM")
        ax_sub.set_ylabel(f"ch{n} (escalado)")
        sombrear(ax_sub, desde, hasta)
        ax_sub.grid(alpha=0.3)
    axes[0].legend(loc="upper left")

    # canal de residuos individuales y combinado
    for n in num:
        axes[2].plot(tramo.index, tramo[f"err_{n}"], lw=0.5, alpha=0.6, label=f"ch{n}")
    axes[2].plot(tramo.index, tramo["err_medio"], lw=1.2, color="black", label="medio")
    axes[2].axhline(p99, color="green", ls="--", label="p99 Val")
    axes[2].set_ylabel("|y - y_hat|")
    sombrear(axes[2], desde, hasta)
    axes[2].legend(loc="upper left", ncol=8, fontsize=8)
    axes[2].grid(alpha=0.3)
    axes[0].set_title(f"Evento {ev_id} ({ev['Category']}, {ev['duracion_h']:.1f} h)")
    plt.tight_layout()
    ruta_zoom = os.path.join(DIR_SALIDA, f"fig8_zoom_{ev_id}.png")
    plt.savefig(ruta_zoom, dpi=120)
    print(f"Grafico guardado: {ruta_zoom}")
    plt.show()

for ev_id in eventos_zoom:
    zoom(ev_id)

# guardamos tambien fig6_prediccion_vs_real p mantener compatibilidad con referencias previas
ruta_fig6 = os.path.join(DIR_SALIDA, "fig6_prediccion_vs_real.png")
if os.path.exists(os.path.join(DIR_SALIDA, "fig8_zoom_id_96.png")):
    import shutil
    shutil.copyfile(os.path.join(DIR_SALIDA, "fig8_zoom_id_96.png"), ruta_fig6)

print(f"\nPredicciones, metricas y graficos guardados exitosamente en {DIR_SALIDA}")
