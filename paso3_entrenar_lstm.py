import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# 0) CONFIGURACION Y DISPOSITIVO
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_VENTANAS = os.path.join(DIR_BASE, "datos_procesados", "ventanas")
DIR_SALIDA = os.path.join(DIR_BASE, "datos_procesados", "modelo")
os.makedirs(DIR_SALIDA, exist_ok=True)
# Fijar semillas para reproducibilidad cientifica
torch.manual_seed(42)
np.random.seed(42)

# Hiperparametros
BATCH_SIZE = 128
LR = 0.001
MAX_EPOCHS = 40
PATIENCE = 6         # epocas a esperar sin mejora antes de early stopping
# 1) CARGA DE TENSORES
print("\n=== 1) Carga de tensores ===")
X_train = np.load(os.path.join(DIR_VENTANAS, "X_train.npy"))
y_train = np.load(os.path.join(DIR_VENTANAS, "y_train.npy"))
X_val   = np.load(os.path.join(DIR_VENTANAS, "X_val.npy"))
y_val   = np.load(os.path.join(DIR_VENTANAS, "y_val.npy"))

print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}   | y_val:   {y_val.shape}")

# Crear DataLoaders de PyTorch para iterar por lotes eficientemente
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
#aca no estamos mezclando los tiempos, sino las venatanas 
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 2) ARQUITECTURA DE LA RED LSTM
class PredictorLSTM(nn.Module):
    #el dropout apaga el 20% de las conexiones para evitar el sobreajuste durante el entrenamiento
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, output_dim=6, dropout=0.2):
        super(PredictorLSTM, self).__init__()
        # Capa recurrente LSTM: procesa secuencias de tamano (batch, seq_len=80, input_dim=7)
        self.lstm = nn.LSTM(input_size=input_dim,hidden_size=hidden_dim,num_layers=num_layers,batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0)
        # Capa de salida lineal: proyecta el ultimo estado oculto hacia los 6 canales target
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # x: (batch_size, 80, 7)
        out, (hn, cn) = self.lstm(x)
        # out[:, -1, :]: tomamos el estado del ultimo paso de tiempo (paso 79 -> instante t-1)
        pred = self.fc(out[:, -1, :])
        return pred

model = PredictorLSTM().to(DEVICE)
print("\n=== 2) Arquitectura del Modelo ===")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total de parametros entrenables: {total_params:,}")

# 3) FUNCION DE PERDIDA Y OPTIMIZADOR
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 4) ENTRENAMIENTO CON EARLY STOPPING
print(f"\n=== 4) Iniciando entrenamiento (max {MAX_EPOCHS} epocas, patience={PATIENCE}) ===")

historial = {"train_loss": [], "val_loss": []}
best_val_loss = float("inf")
patience_counter = 0
ruta_modelo = os.path.join(DIR_SALIDA, "modelo_lstm.pt")

t_inicio_total = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    t_inicio_epoch = time.time()
    # --- FASE DE ENTRENAMIENTO ---
    model.train()
    running_train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * len(batch_x)
    train_loss = running_train_loss / len(train_dataset)
    
    # --- FASE DE VALIDACION ---
    model.eval()
    running_val_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            running_val_loss += loss.item() * len(batch_x)
    val_loss = running_val_loss / len(val_dataset)
    
    duracion = time.time() - t_inicio_epoch
    historial["train_loss"].append(train_loss)
    historial["val_loss"].append(val_loss)
    
    print(f"Epoca [{epoch:02d}/{MAX_EPOCHS}] - {duracion:.1f}s | "
          f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}", end="")
    
    # Control de Early Stopping y guardado del mejor modelo
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), ruta_modelo)
        print(" -> Mejor modelo guardado (*)")
    else:
        patience_counter += 1
        print(f" -> Sin mejora ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print(f"\nEarly Stopping activado en la epoca {epoch}: no hubo mejora en validacion.")
            break

t_total = time.time() - t_inicio_total
print(f"\nEntrenamiento finalizado en {t_total/60:.2f} minutos.")
print(f"Mejor Val MSE alcanzado: {best_val_loss:.6f}")

# 5) GRAFICO DE LA CURVA DE APRENDIZAJE Y GUARDADO
# Guardar historial en formato JSON
with open(os.path.join(DIR_SALIDA, "historial_entrenamiento.json"), "w") as f:
    json.dump(historial, f, indent=2)

# Graficar perdida epoca por epoca
plt.figure(figsize=(9, 4.5))
plt.plot(range(1, len(historial["train_loss"]) + 1), historial["train_loss"], "o-", label="Train Loss (MSE)", color="tab:blue")
plt.plot(range(1, len(historial["val_loss"]) + 1), historial["val_loss"], "s-", label="Val Loss (MSE)", color="tab:orange")
plt.xlabel("Epoca")
plt.ylabel("Error Cuadratico Medio (MSE)")
plt.title("Curva de Aprendizaje - Modelo LSTM (Regimen Sano)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
ruta_fig5 = os.path.join(DIR_SALIDA, "fig5_curva_perdida.png")
plt.savefig(ruta_fig5, dpi=120)
print(f"Grafico guardado: {ruta_fig5}")
print(f"Pesos del modelo guardados: {ruta_modelo}")
plt.show()
