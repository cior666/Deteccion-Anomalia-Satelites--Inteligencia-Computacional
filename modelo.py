import os
import torch
import torch.nn as nn

# 0) configuracion y rutas por defecto
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(DIR_BASE, "datos_procesados")
DIR_MODELO_DEFECTO = os.path.join(DIR_DATOS, "modelo", "modelo_lstm.pt")

# 1) definicion de la arquitectura lstm (2 capas recurrentes + proyeccion lineal)
class ModeloLSTM(nn.Module):
    def __init__(self, n_entrada, n_ocultas, n_salida, dropout=0.0):
        super().__init__()
        # primera lstm: extrae dinamicas temporales a lo largo de la ventana de 80 pasos
        self.lstm1 = nn.LSTM(n_entrada, n_ocultas, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        # segunda lstm: procesa la secuencia intermedia y resume la informacion
        self.lstm2 = nn.LSTM(n_ocultas, n_ocultas, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        # capa lineal final: predice los canales objetivo en el instante siguiente
        self.salida = nn.Linear(n_ocultas, n_salida)

    def forward(self, x):
        sec1, _ = self.lstm1(x)
        sec1 = self.drop1(sec1)
        sec2, _ = self.lstm2(sec1)
        # tomamos unicamente el ultimo estado oculto h_L q resume toda la ventana
        h_ultimo = sec2[:, -1, :]
        h_ultimo = self.drop2(h_ultimo)
        return self.salida(h_ultimo)

# 2) funcion utilitaria p cargar el modelo entrenado y sus pesos
def cargar_modelo(ruta_modelo=None, dispositivo="cpu"):
    if ruta_modelo is None or not os.path.exists(ruta_modelo):
        if os.path.exists(DIR_MODELO_DEFECTO):
            ruta_modelo = DIR_MODELO_DEFECTO

    ckpt = torch.load(ruta_modelo, map_location=dispositivo)
    hp = ckpt.get("hiperparametros", {})
    n_entrada = hp.get("n_entrada", 7)
    n_ocultas = hp.get("n_ocultas", 64)
    n_salida = hp.get("n_salida", 6)

    modelo = ModeloLSTM(n_entrada, n_ocultas, n_salida, dropout=0.0).to(dispositivo)
    pesos = ckpt["pesos"] if "pesos" in ckpt else ckpt
    modelo.load_state_dict(pesos)
    modelo.eval()
    return modelo, ckpt
