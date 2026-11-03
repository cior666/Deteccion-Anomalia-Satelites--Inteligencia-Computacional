"""
Funciones compartidas para armar ventanas deslizantes.
Las usan paso2_ventanas.py, paso3 (entrenamiento) y paso4 (prediccion).
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def indices_validos(malo, L):
    malo = np.asarray(malo, dtype=np.int64)
    acum = np.concatenate([[0], np.cumsum(malo)])     # acum[k] = malos en [0, k)
    t = np.arange(L, len(malo))                       # los primeros L no tienen historia
    cuenta = acum[t + 1] - acum[t - L]                # malos en [t-L, t]
    return t[cuenta == 0]


def armar_ventanas(matriz, indices_t, L, n_target):
    vista = sliding_window_view(matriz, window_shape=L, axis=0)  # creo las ventanas deslizantes a lo largo del eje
    #temporal (N-L+1, canales, L)
    X = vista[indices_t - L]                  # aca recien se copian los datos
    X = np.transpose(X, (0, 2, 1))            # -> (muestras, L, canales)
    y = matriz[indices_t, :n_target]
    return np.ascontiguousarray(X, dtype=np.float32), y.astype(np.float32)
