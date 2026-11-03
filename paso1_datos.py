
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(DIR_BASE, "Datos")
DIR_SALIDA = os.path.join(DIR_BASE, "datos_procesados")
os.makedirs(DIR_SALIDA, exist_ok=True)
CANALES_TARGET = [f"channel_{i}" for i in range(41, 47)]  # los 6 del subsistema a predecir
CANAL_CONTEXTO = "channel_1"                              # canal de contexto global
CANALES = CANALES_TARGET + [CANAL_CONTEXTO]               # 7 canales de entrada
FECHA_INICIO = "2004-01-01"
FECHA_FIN = "2006-01-01"   # excluyente: abarca 2004 y 2005 completos
PASO = "90s"               # resolucion temporal comun
MAX_RELLENO = 10           # maximo de pasos (10 x 90s = 15 min) para ffill
# Particion cronologica
FIN_TRAIN = "2004-06-15"   # Train: 2004-01-01 -> 2004-06-15
FIN_VAL = "2004-08-01"     # Val:   2004-06-15 -> 2004-08-01
# Test:  2004-08-01 -> 2005-12-31

# 1) LECTURA DE LOS CANALES
def leer_canal(nombre):
    """Lee el DataFrame serializado (pickle) directamente desde la carpeta Datos.
    Recorta inmediatamente al rango 2004-2005 para optimizar el uso de memoria."""
    ruta = os.path.join(DIR_DATOS, nombre)
    df = pd.read_pickle(ruta)
    serie = df[nombre].loc[FECHA_INICIO:FECHA_FIN] #se extrae la col. del sensor como series de pandas (col indexada al tiempo)
    #uso .loc para recortar el rango temporal q quiero analziar
    serie = serie[serie.index < FECHA_FIN] #filtro para quedarme solo hasta fecha fin q defini mas arriba
    return serie

def describir_muestreo(serie):
    """Calcula la mediana y el maximo salto temporal entre mediciones (en segundos)."""
    dt = serie.index.to_series().diff().dt.total_seconds().dropna()
    return dt.median(), dt.max()

print("=== 1) Lectura de canales (2004-2005) ===")
series = {} #hacemos una dicc para guardar las series y q dsp queden los canales con sus nombres como clave
for c in CANALES:
    s = leer_canal(c)
    mediana_dt, max_dt = describir_muestreo(s)
    print(f"{c:>11}: {len(s):>9,} mediciones | paso tipico {mediana_dt:5.1f}s | "
          f"hueco maximo {max_dt/3600:6.2f} h | rango [{s.min():.3f}, {s.max():.3f}]") #para mi lo sacamos a la pija
    series[c] = s

# 2) ALINEACION EN GRILLA COMUN DE 90 SEGUNDOS
print(f"\n=== 2) Alineacion a grilla de {PASO} ===")
grilla = pd.date_range(FECHA_INICIO, FECHA_FIN, freq=PASO, inclusive="left") #armo la secuencia de fecha y horas 
#de manera regular osea con el paso q defini antes 90seg entre las fechas q nos interesan
columnas = {} #armo diccionario tb con las col alineadas antes de armar la tabla final
for c in CANALES:
    columnas[c] = series[c].resample(PASO, origin=FECHA_INICIO).mean().reindex(grilla) # divido toda la linea de tiempo 
    #en ventanas de 90seg, calculo el promedio de los valores dentro de cada ventana y relleno los huecos con eso.
    #dsp se hace  el reindex para alinearlo con la grilla q acabamos de definir
datos = pd.DataFrame(columnas, index=grilla) #construyo el df 
datos.index.name = "datetime"
del series
print(f"Grilla: {len(datos):,} instantes x {datos.shape[1]} canales")
print("Faltantes pre relleno (% de instantes):")
print((datos.isna().mean() * 100).round(2).to_string())

# completado de datos con el max de relleno q definimos antes
datos = datos.ffill(limit=MAX_RELLENO)
datos["hueco"] = datos[CANALES].isna().any(axis=1)
print(f"\nTras ffill (max {MAX_RELLENO} pasos): {datos['hueco'].mean()*100:.2f}% "f"de instantes siguen con algun canal faltante quedan representados por el nombre hueco")

# 4) CRUCE CON labels.csv (GROUND TRUTH)
print("\n=== 4) Cruce con etiquetas ===")
labels = pd.read_csv(os.path.join(DIR_DATOS, "labels.csv"),parse_dates=["StartTime", "EndTime"]) #leo el labels y convierto a fecha y hora
tipos = pd.read_csv(os.path.join(DIR_DATOS, "anomaly_types.csv")) #leo y esto contiene el tipo de evento asociado a un id
labels = labels.merge(tipos[["ID", "Category"]], on="ID", how="left") #aca le hago un join por id para saber que tipo de evento es

#a las fechas hay q sacarle la zona horaria porq no tienen zona horaria asociada y si se la dejo me tira error al comparar
labels["StartTime"] = labels["StartTime"].dt.tz_localize(None)
labels["EndTime"] = labels["EndTime"].dt.tz_localize(None)

# filtramos los 6 canales en el periodo de estudio
#nos quedamos solo con los eventos que afectan a nuestros canales y dentro de nuestro periodo de estudio
eventos = labels[labels["Channel"].isin(CANALES_TARGET)& (labels["StartTime"] < FECHA_FIN)& (labels["EndTime"] >= FECHA_INICIO)].copy()

# hacemos las etiqueteas booleanas canal x canal
for c in CANALES_TARGET:
    col = "etiq_" + c.split("_")[1]
    datos[col] = False #ponemos todo en false al ppio
    for _, ev in eventos[eventos["Channel"] == c].iterrows():
        ini = ev["StartTime"].floor(PASO) #redondea para abajo al paso q definimos
        fin = ev["EndTime"].floor(PASO) #lo mismo
        datos.loc[ini:fin, col] = True #ponemos en true todas las filas q caen dentro del periodo de anomalia

cols_etiq = [c for c in datos.columns if c.startswith("etiq_")]
datos["anomalia"] = datos[cols_etiq].any(axis=1) #hacemos la col anomalias en tiempo t p cualq canal del subs 5 q esta fallando (true) o no (false)

#agrupar eventos x ID y armar resumen de eventos
resumen = (eventos.groupby("ID")
           .agg(StartTime=("StartTime", "min"), EndTime=("EndTime", "max"),
                Category=("Category", "first"),
                canales=("Channel", "nunique"),
                segmentos=("Channel", "count"))
           .sort_values("StartTime"))
resumen["duracion_h"] = ((resumen["EndTime"] - resumen["StartTime"]).dt.total_seconds() / 3600).round(2)

print(f"{len(resumen)} eventos afectan a los canales 41-46 en 2004-2005:")
print(resumen.to_string())
print(f"Instantes etiquetados: {datos['anomalia'].sum():,} "f"({datos['anomalia'].mean()*100:.2f}% del total)")

ruta_eventos = os.path.join(DIR_SALIDA, "eventos_2004_2005.csv")
resumen.to_csv(ruta_eventos)

# 5) PARTICION CRONOLOGICA
datos["conjunto"] = "test"
datos.loc[datos.index < FIN_VAL, "conjunto"] = "val"
datos.loc[datos.index < FIN_TRAIN, "conjunto"] = "train"

print("\n=== 5) Particion ===")
for nombre in ["train", "val", "test"]:
    parte = datos[datos["conjunto"] == nombre]
    print(f"{nombre:>5}: {parte.index[0].date()} -> {parte.index[-1].date()} | "
          f"{len(parte):>7,} instantes | anomalos: {parte['anomalia'].sum():>5,} | "
          f"huecos: {parte['hueco'].sum():>6,}")

# 6) GRAFICOS DE VERIFICACION
print("\n=== 6) Generando graficos de verificacion ===")
COLOR_CAT = {"Anomaly": "red", "Rare Event": "orange"}

def sombrear_eventos(ax, desde=None, hasta=None):
    for _, ev in resumen.iterrows():
        if desde is not None and (ev["EndTime"] < desde or ev["StartTime"] > hasta):
            continue
        ax.axvspan(ev["StartTime"], ev["EndTime"],
                   color=COLOR_CAT.get(ev["Category"], "gray"), alpha=0.35, lw=0)

# Figura 1: Panorama general de los 2 anios (promedio por hora para optimizar render)
horario = datos[CANALES].resample("1h").mean()
fig, axes = plt.subplots(len(CANALES), 1, figsize=(15, 13), sharex=True)
for ax, c in zip(axes, CANALES):
    ax.plot(horario.index, horario[c], lw=0.5, color="tab:blue")
    sombrear_eventos(ax)
    ax.axvline(pd.Timestamp(FIN_TRAIN), color="green", ls="--", lw=1.2)
    ax.axvline(pd.Timestamp(FIN_VAL), color="purple", ls="--", lw=1.2)
    ax.set_ylabel(c.replace("channel_", "ch"), rotation=0, labelpad=20)
    ax.grid(alpha=0.3)
axes[0].set_title("Canales 41-46 + 1 (prom. horario) | rojo=Anomaly, naranja=Rare Event | ""verde=fin Train, violeta=fin Val")
plt.tight_layout()
ruta_fig1 = os.path.join(DIR_SALIDA, "fig1_panorama.png")
plt.savefig(ruta_fig1, dpi=120)
plt.close(fig)
print(f"Guardado: {ruta_fig1}")

# Figura 2: Zoom a resolucion real (90 s) sobre anomalia del test (id_96)
ev = resumen.loc["id_96"]
desde = ev["StartTime"] - pd.Timedelta("2D")
hasta = ev["EndTime"] + pd.Timedelta("2D")
tramo = datos.loc[desde:hasta]
fig, axes = plt.subplots(len(CANALES), 1, figsize=(15, 12), sharex=True)
for ax, c in zip(axes, CANALES):
    ax.plot(tramo.index, tramo[c], lw=0.6, color="tab:blue")
    sombrear_eventos(ax, desde, hasta)
    ax.set_ylabel(c.replace("channel_", "ch"), rotation=0, labelpad=20)
    ax.grid(alpha=0.3)
axes[0].set_title(f"Zoom evento id_96 ({ev['Category']}) a resolucion de 90 s")
plt.tight_layout()
ruta_fig2 = os.path.join(DIR_SALIDA, "fig2_zoom_evento.png")
plt.savefig(ruta_fig2, dpi=120)
plt.close(fig)
print(f"Guardado: {ruta_fig2}")

# 7) GUARDADO DE DATOS PROCESADOS
ruta_pkl = os.path.join(DIR_SALIDA, "datos_alineados.pkl")
datos.to_pickle(ruta_pkl)
print(f"\n=== 7) Guardado final exitoso ===")
print(f"Archivo pickle: {ruta_pkl}")
print("Columnas generadas:", list(datos.columns))
