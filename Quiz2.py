import numpy as np
import pandas as pd
matriz_4d = np.random.rand(20, 30, 20, 100)  
matriz_3d = matriz_4d[0]
def see_atributos(m):
    print("\nAtributos de la matriz:")
    print(f"Forma: {m.shape}")
    print(f"Dimensiones: {m.ndim}D")
    print(f"Tamaño total: {m.size} elementos")
    print(f"Tipo de datos: {m.dtype}")
matriz_2d = matriz_3d.reshape(-1, matriz_3d.shape[-1])