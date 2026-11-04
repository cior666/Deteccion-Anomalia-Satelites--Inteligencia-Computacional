import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from ventanas import armar_ventanas

# configuracion
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
# DIR_TP no requerido
DIR_VENT = os.path.join(DIR_BASE, "datos_procesados", "ventanas")
DIR_MODELOS = os.path.join(DIR_BASE, "datos_procesados", "modelo")
os.makedirs(DIR_MODELOS, exist_ok=True)

SEMILLA = 0
torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)

# Hiperparametros
capas_ocultas = 64 # neuronas de cada capa LSTM = largo de los vectores h y C
tasa_apagado = 0.2 # fraccion de neuronas que se "apagan" al azar en cada batch
BATCH = 64 # ventanas por mini-batch
tasa_aprendizaje = 1e-3 # tasa de aprendizaje inicial de Adam
epocas = 50
paciencia = 8 # epocas sin mejorar en Val antes de cortar p hacer el early stopping digamos
mejora_minima = 1e-4 # mejora minima que cuenta como "mejora"
CLIP = 1.0 # tope para la norma del gradiente (ver punto 4)

# GPU 
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", dispositivo,
      f"({torch.cuda.get_device_name(0)})" if dispositivo.type == "cuda" else "")


# datos
with open(os.path.join(DIR_VENT, "config_ventanas.json")) as f:
    cfg = json.load(f)
L = cfg["L"]
cant_canales = len(cfg["canales"])           # 7 (entrada)
cant_objetivos = len(cfg["canales_target"])     # 6 (salida)

matriz = np.load(os.path.join(DIR_VENT, "matriz_escalada.npy"))
idx_train = np.load(os.path.join(DIR_VENT, "idx_train.npy"))
idx_val = np.load(os.path.join(DIR_VENT, "idx_val.npy"))

X_train, y_train = armar_ventanas(matriz, idx_train, L, cant_objetivos)
X_val, y_val = armar_ventanas(matriz, idx_val, L, cant_objetivos)
print(f"X_train {X_train.shape} | y_train {y_train.shape}")
print(f"X_val   {X_val.shape} | y_val   {y_val.shape}")

# Subimos todo a la GPU de una vez (~460 MB, entra en los 4 GB de la placa).
# Asi no hay que copiar de RAM a GPU en cada batch, que seria lo mas lento.
X_train_t = torch.from_numpy(X_train).to(dispositivo)
y_train_t = torch.from_numpy(y_train).to(dispositivo)
X_val_t = torch.from_numpy(X_val).to(dispositivo)
y_val_t = torch.from_numpy(y_val).to(dispositivo)
del X_train, X_val


# arquitectura
class ModeloLSTM(nn.Module):
    """
    Entrada  x: (batch, L, 7)
      lstm1 -> recorre los L pasos y devuelve h en CADA paso: (batch, L, 64)
               (equivale a return_sequences=True)
      lstm2 -> recorre esa secuencia de L vectores; nos quedamos solo con el
               h del ULTIMO paso: (batch, 64)   (return_sequences=False)
      salida -> capa lineal 64 -> 6: una prediccion por canal target
    Salida y_hat: (batch, 6)
    """
    # batch es el numero de ventanas que se procesan en paralelo.
    # batch_first=True -> los tensores son (batch, tiempo, canales)
    # return_sequences=True -> la capa devuelve la secuencia completa de estados ocultos h_1..h_L
    # return_sequences=False -> la capa devuelve solo el estado oculto final h_L
    def __init__(self, n_entrada, n_ocultas, n_salida, dropout):
        super().__init__()
        self.lstm1 = nn.LSTM(n_entrada, n_ocultas, batch_first=True) # Primera LSTM: aprende representaciones temporales a partir de los canales de telemetria
        self.drop1 = nn.Dropout(dropout)  # se apaga aleatoriamente el 20% de los valores de salida de esta capa,
        # p evitar q la red dependa demasiado de determinadas neuronas y reduce el sobreajuste
        self.lstm2 = nn.LSTM(n_ocultas, n_ocultas, batch_first=True) # Segunda LSTM: procesa esas representaciones para obtener una descripcion temporal mas elaborada
        self.drop2 = nn.Dropout(dropout)
        # Capa de salida: proyecta la representacion final (64 dimensiones) a los 6 canales de telemetria que queremos predecir
        # y* = W* h_ultimo + b
        self.salida = nn.Linear(n_ocultas, n_salida)   # activacion lineal

    # h (dim=64) es el estado oculto, q representa la informacion q la LSTM utiliza como rep del instante actual
    # C (dim=64) es el estado de la celda, q representa la informacion q la LSTM utiliza como memoria a largo plazo
    # h(0) y c(0) = 0
    # las compuertas f (olvido), i (entrada) y o (salida) son compuertas q controlan el flujo de informacion q entra, sale o se olvida de la celda de memoria C
    # C_t = f_t * C_{t-1} + i_t * C~   // C~ = Memoria candidata para incorporar a la memoria
    # f_t * C_{t-1} -> cuanto consevar de la memoria anterior
    # i_t * C~ -> q tanta nueva informacion entra
    # h_t = o_t * tanh(C_t) -> q tanta informacion sale

    def forward(self, x):
        # nn.LSTM devuelve (secuencia_de_h, (h_final, C_final)).
        # Los estados h0 y C0 arrancan en cero en cada ventana (por defecto).
        sec1, _ = self.lstm1(x)          # (batch, L, 64)  h_1 ... h_L de la capa 1
        sec1 = self.drop1(sec1)          # se produce el dropout
        sec2, _ = self.lstm2(sec1)       # (batch, L, 64)  h_1 ... h_L de la capa 2
        h_ultimo = sec2[:, -1, :]        # (batch, 64)     solo h_L
        h_ultimo = self.drop2(h_ultimo)  # se produce el dropout
        return self.salida(h_ultimo)     # (batch, 6)
# el estado oculto del ultimo instante de la segunda LSTM resume la informacion procesada a lo largo de los 80 pasos de la ventana
# la red utiliza esa representación para generar la prediccion de los 6 canales en el instante siguiente.

modelo = ModeloLSTM(cant_canales, capas_ocultas, cant_objetivos, tasa_apagado).to(dispositivo)
n_param = sum(p.numel() for p in modelo.parameters())
print(f"\n{modelo}\nParametros entrenables: {n_param:,}")
# Por que tantos parametros: cada capa LSTM tiene 4 bloques (f, i, C~, o),
# cada uno con su matriz para la entrada, su matriz para h_{t-1} y su bias.
#   lstm1: 4 * (64*7 + 64*64 + 2*64) = 18.688
#   lstm2: 4 * (64*64 + 64*64 + 2*64) = 33.280
#   salida: 64*6 + 6 = 390
# En total hay 52.368 parametros.
# 64*7 debido a 64 neuronas ocultas y 7 canales de entrada 
# 64*64 debido a 64 neuronas ocultas y 64 estados ocultos
# 2*64 debido a los dos bias y 64 neuronas ocultas

criterio = nn.MSELoss() # la funcion de perdida mide la diferencia entre la prediccion del modelo y el valor real
optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje) # Adam es un optimizador que ajusta los pesos del modelo para minimizar la funcion de perdida


# modelo persistencia:
# el valor en t va a ser igual al de t-1. No aprende nada. Si la LSTM no
# le gana claramente a esto, no sirve para detectar anomalias
mse_persistencia = criterio(X_val_t[:, -1, :cant_objetivos], y_val_t).item()
print(f"\nMSE de persistencia en Val (referencia a superar): {mse_persistencia:.4f}")


def evaluar(X, y):
    """MSE sobre un conjunto completo, por partes para no llenar la GPU."""
    modelo.eval()                       # apaga el Dropout
    total = 0.0
    with torch.no_grad():               # no hace falta calcular gradientes
        for i in range(0, len(X), 4096):
            pred = modelo(X[i:i + 4096])
            total += criterio(pred, y[i:i + 4096]).item() * len(pred) # promedio por cantidad de muestras
    return total / len(X) # promedio de todos los batches


# etapa entrenamiento
# Una EPOCA = pasar una vez por las ~159.000 ventanas de Train, de a BATCH.

# El orden temporal importa DENTRO de
# cada ventana (los 80 pasos van en orden), pero cada ventana es independiente:
# la LSTM arranca con h=0 y C=0 en cada una. Mezclarlas evita que los batches
# consecutivos sean casi identicos (ventanas vecinas se solapan en 79 pasos).

# Recorte de gradiente: en BPTT el gradiente se propaga hacia atras
# por los 80 pasos y a veces "explota". Si su norma supera CLIP, se lo achica
# manteniendo la direccion.
print("\n=== Entrenamiento ===")
historial = {"train": [], "val": []}
mejor_val, mejor_epoca, sin_mejorar = np.inf, 0, 0
n = len(X_train_t)

for epoca in range(1, epocas + 1):
    t0 = time.time()
    modelo.train()                                  # activa el Dropout
    orden = torch.randperm(n, device=dispositivo)   # mezcla las ventanas 
    suma = 0.0

    for i in range(0, n, BATCH):  # n=159026, batch=64
        lote = orden[i:i + BATCH]
        xb, yb = X_train_t[lote], y_train_t[lote]   # (64, 80, 7) y (64, 6)

        optimizador.zero_grad()           #  borrar gradientes del batch anterior, no los pesos
        pred = modelo(xb)                 #  forward: 80 pasos por las 2 LSTM / pred(64, 6)
        perdida = criterio(pred, yb)      #  MSE promedio del batch
        perdida.backward()                #  BPTT: se calcula el gradiente de cada peso, propaga el error a traves de las capas y los instantes pasados
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), CLIP) # limita el tamaño del gradiente 
        optimizador.step()                #  adam utiliza los gradientes calculados y mantiene información de gradientes anteriores para adaptar las actualizaciones de cada parámetro.

        suma += perdida.item() * len(lote) # para calcular el mse promedio de toda la epoca

    mse_train = suma / n # mse promedio de toda la epoca
    mse_val = evaluar(X_val_t, y_val_t) # mse promedio de la validacion
    historial["train"].append(mse_train)
    historial["val"].append(mse_val)

    # early stopping: guardamos los pesos solo si Val mejoro
    marca = ""
    if mse_val < mejor_val - mejora_minima:
        mejor_val, mejor_epoca, sin_mejorar = mse_val, epoca, 0
        torch.save(modelo.state_dict(), os.path.join(DIR_MODELOS, "mejores_pesos.tmp")) # guardamos parametros
        marca = "  <- mejor"
    else:
        sin_mejorar += 1

    print(f"Epoca {epoca:2d} | MSE train {mse_train:.4f} | MSE val {mse_val:.4f} | "
          f"{time.time() - t0:5.1f}s{marca}")

    if sin_mejorar >= paciencia:
        print(f"Parada temprana: {paciencia} epocas sin mejorar en Val.")
        break

# Nos quedamos con los pesos de la mejor epoca, no con los de la ultima
ruta_tmp = os.path.join(DIR_MODELOS, "mejores_pesos.tmp")
modelo.load_state_dict(torch.load(ruta_tmp))
os.remove(ruta_tmp)
print(f"\nMejor epoca: {mejor_epoca} | MSE val {mejor_val:.4f} | "
      f"persistencia {mse_persistencia:.4f} "
      f"-> la LSTM reduce el error un {100 * (1 - mejor_val / mse_persistencia):.0f}%")


# grafico
# curva de aprendizaje
ep = np.arange(1, len(historial["train"]) + 1)
plt.figure(figsize=(9, 4))
plt.plot(ep, historial["train"], "o-", label="Train")
plt.plot(ep, historial["val"], "o-", label="Val")
plt.axhline(mse_persistencia, color="gray", ls=":", label="Persistencia (Val)")
plt.axvline(mejor_epoca, color="green", ls="--", label=f"mejor epoca ({mejor_epoca})")
plt.xlabel("epoca")
plt.ylabel("MSE (datos escalados)")
plt.title("Curva de aprendizaje")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR_MODELOS, "fig5_curva_aprendizaje.png"), dpi=120)

#  Real vs predicho en un dia de Val (960 pasos de 90 s = 24 h)
modelo.eval()
with torch.no_grad():
    pred_val = modelo(X_val_t[:960]).cpu().numpy()
real_val = y_val_t[:960].cpu().numpy()

fig, axes = plt.subplots(cant_objetivos, 1, figsize=(14, 11), sharex=True)
horas = np.arange(960) * 90 / 3600
for j, ax in enumerate(axes):
    ax.plot(horas, real_val[:, j], lw=0.8, label="real")
    ax.plot(horas, pred_val[:, j], lw=0.8, label="LSTM")
    ax.set_ylabel(cfg["canales_target"][j].replace("channel_", "ch"),
                  rotation=0, labelpad=20)
    ax.grid(alpha=0.3)
axes[0].legend(loc="upper right")
axes[0].set_title("Validacion: real vs prediccion un paso adelante (primer dia de Val)")
axes[-1].set_xlabel("horas")
plt.tight_layout()
plt.savefig(os.path.join(DIR_MODELOS, "fig6_prediccion_val.png"), dpi=120)


# guardo
torch.save({
    "pesos": modelo.state_dict(),
    "hiperparametros": {"n_entrada": cant_canales, "n_ocultas": capas_ocultas,
                        "n_salida": cant_objetivos, "dropout": tasa_aprendizaje, "L": L},
    "mejor_epoca": mejor_epoca, "mse_val": mejor_val,
    "mse_persistencia": mse_persistencia,
}, os.path.join(DIR_MODELOS, "modelo_lstm.pt"))
with open(os.path.join(DIR_MODELOS, "historial.json"), "w") as f:
    json.dump(historial, f, indent=2)
print(f"Modelo guardado en {DIR_MODELOS}")

plt.show()