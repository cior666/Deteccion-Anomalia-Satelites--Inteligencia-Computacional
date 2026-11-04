import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 0) configuracion y rutas principales
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(DIR_BASE, "Datos")
DIR_SALIDA = os.path.join(DIR_BASE, "datos_procesados")
os.makedirs(DIR_SALIDA, exist_ok=True)

canales_objetivo = [f"channel_{i}" for i in range(41, 47)]  # los 6 canales del subsistema 5 a predecir
canal_contexto = "channel_1"  # canal de contexto operativo global
canales = canales_objetivo + [canal_contexto]  # los 7 canales de entrada del modelo

fecha_inicio = "2004-01-01"
fecha_fin = "2006-01-01"  # excluyente: abarca los periodos 2004 y 2005 completos
paso = "90s"  # resolucion temporal uniforme
maximo_relleno = 10  # limite maximo de pasos (10 x 90s = 15 min) p aplicar ffill

# particion cronologica estricta
fecha_fin_train = "2004-06-15"  # train: 2004-01-01 -> 2004-06-15
fecha_fin_validacion = "2004-08-01"  # val: 2004-06-15 -> 2004-08-01
# test: 2004-08-01 -> 2005-12-31

# 1) lectura de los canales individuales
def leer_canal(nombre):
    ruta = os.path.join(DIR_DATOS, nombre)
    df = pd.read_pickle(ruta)
    # extraemos la columna del sensor como serie temporal indexada por fecha
    serie = df[nombre].loc[fecha_inicio:fecha_fin]
    # filtramos p asegurar q no supere la fecha de fin definida
    serie = serie[serie.index < fecha_fin]
    return serie

def describir_muestreo(serie):
    # calcula el intervalo temporal tipico y el salto temporal maximo de la serie
    dt = serie.index.to_series().diff().dt.total_seconds().dropna()
    return dt.median(), dt.max()

print("=== 1) Lectura de canales (2004-2005) ===")
series = {}  # diccionario p almacenar las series individuales leidas
for c in canales:
    s = leer_canal(c)
    mediana_dt, max_dt = describir_muestreo(s)
    print(f"{c:>11}: {len(s):>9,} mediciones | paso tipico {mediana_dt:5.1f}s | "
          f"hueco maximo {max_dt/3600:6.2f} h | rango [{s.min():.3f}, {s.max():.3f}]")
    series[c] = s

# 2) alineacion en grilla temporal comun de 90 segundos
grilla = pd.date_range(fecha_inicio, fecha_fin, freq=paso, inclusive="left")
columnas = {}
for c in canales:
    # promediamos las mediciones dentro de cada ventana regular de 90s y reindexamos a la grilla comun
    columnas[c] = series[c].resample(paso, origin=fecha_inicio).mean().reindex(grilla)

datos = pd.DataFrame(columnas, index=grilla)
datos.index.name = "datetime"
del series

# 3) tratamiento de huecos con ffill acotado
# rellenamos faltantes breves hasta 15 min; los huecos mayores quedan marcados como invalidos
datos = datos.ffill(limit=maximo_relleno)
datos["hueco"] = datos[canales].isna().any(axis=1)
pct_hueco = datos["hueco"].mean() * 100
print(f"\nTras ffill (max {maximo_relleno} pasos): {pct_hueco:.2f}% de instantes con huecos residuales")

# 4) cruce con etiquetas reales (ground truth)
print("\n=== 4) Cruce con etiquetas ===")
labels = pd.read_csv(os.path.join(DIR_DATOS, "labels.csv"), parse_dates=["StartTime", "EndTime"])
tipos = pd.read_csv(os.path.join(DIR_DATOS, "anomaly_types.csv"))
labels = labels.merge(tipos[["ID", "Category"]], on="ID", how="left")

# removemos la zona horaria p evitar desajustes al comparar timestamps de pandas
labels["StartTime"] = labels["StartTime"].dt.tz_localize(None)
labels["EndTime"] = labels["EndTime"].dt.tz_localize(None)

# filtramos los eventos q afectan a los canales objetivo dentro del periodo de analisis
eventos = labels[
    labels["Channel"].isin(canales_objetivo)
    & (labels["StartTime"] < fecha_fin)
    & (labels["EndTime"] >= fecha_inicio)
].copy()

# construimos banderas booleanas de falla canal por canal
for c in canales_objetivo:
    col = "etiq_" + c.split("_")[1]
    datos[col] = False
    for _, ev in eventos[eventos["Channel"] == c].iterrows():
        ini = ev["StartTime"].floor(paso)
        fin = ev["EndTime"].floor(paso)
        datos.loc[ini:fin, col] = True

cols_etiq = [c for c in datos.columns if c.startswith("etiq_")]
datos["anomalia"] = datos[cols_etiq].any(axis=1)  # bandera global de anomalia en el subsistema 5

# agrupamos por ID p consolidar el catalogo de eventos de falla
resumen = (eventos.groupby("ID")
           .agg(StartTime=("StartTime", "min"), EndTime=("EndTime", "max"),
                Category=("Category", "first"),
                canales=("Channel", "nunique"),
                segmentos=("Channel", "count"))
           .sort_values("StartTime"))
resumen["duracion_h"] = ((resumen["EndTime"] - resumen["StartTime"]).dt.total_seconds() / 3600).round(2)

print(f"{len(resumen)} eventos afectan a los canales 41-46 en 2004-2005:")
print(resumen.to_string())
print(f"Instantes etiquetados como anomalos: {datos['anomalia'].sum():,} ({datos['anomalia'].mean()*100:.2f}% del total)")

ruta_eventos = os.path.join(DIR_SALIDA, "eventos_2004_2005.csv")
resumen.to_csv(ruta_eventos)

# 5) particion cronologica de conjuntos (train, val y test)
datos["conjunto"] = "test"
datos.loc[datos.index < fecha_fin_validacion, "conjunto"] = "val"
datos.loc[datos.index < fecha_fin_train, "conjunto"] = "train"

print("\n=== 5) Particion ===")
for nombre in ["train", "val", "test"]:
    parte = datos[datos["conjunto"] == nombre]
    print(f"{nombre:>5}: {parte.index[0].date()} -> {parte.index[-1].date()} | "
          f"{len(parte):>7,} instantes | anomalos: {parte['anomalia'].sum():>5,} | "
          f"huecos: {parte['hueco'].sum():>6,}")

# 6) graficos de verificacion visual (panorama temporal)
color_cat = {"Anomaly": "red", "Rare Event": "orange"}

def sombrear_eventos(ax_grafico, desde=None, hasta=None):
    # sombrea las regiones temporales donde ocurrieron fallas documentadas
    for _, ev in resumen.iterrows():
        if desde is not None and (ev["EndTime"] < desde or ev["StartTime"] > hasta):
            continue
        ax_grafico.axvspan(ev["StartTime"], ev["EndTime"],
                           color=color_cat.get(ev["Category"], "gray"), alpha=0.35, lw=0)

# promedio horario p optimizar velocidad de graficacion
horario = datos[canales].resample("1h").mean()
fig, axes = plt.subplots(len(canales), 1, figsize=(15, 13), sharex=True)
for ax, c in zip(axes, canales):
    ax.plot(horario.index, horario[c], lw=0.5, color="tab:blue")
    sombrear_eventos(ax)
    ax.axvline(pd.Timestamp(fecha_fin_train), color="green", ls="--", lw=1.2)
    ax.axvline(pd.Timestamp(fecha_fin_validacion), color="purple", ls="--", lw=1.2)
    ax.set_ylabel(c.replace("channel_", "ch"), rotation=0, labelpad=20)
    ax.grid(alpha=0.3)
axes[0].set_title("Canales 41-46 + 1 (prom. horario) | rojo=Anomaly, naranja=Rare Event | verde=fin Train, violeta=fin Val")
plt.tight_layout()
ruta_fig1 = os.path.join(DIR_SALIDA, "fig1_panorama.png")
plt.savefig(ruta_fig1, dpi=120)
print(f"Guardado: {ruta_fig1}")
plt.show()

# 7) guardado de datos procesados y resumen de eventos
ruta_pkl = os.path.join(DIR_SALIDA, "datos_alineados.pkl")
datos.to_pickle(ruta_pkl)
print(f"\n=== 7) Guardado final exitoso ===")
print(f"Archivo pickle: {ruta_pkl}")
print("Columnas generadas:", list(datos.columns))
