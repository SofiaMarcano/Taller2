import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
matriz_4d = np.random.rand(20, 30, 20, 100)  
matriz_3d = matriz_4d[0]
def see_atributos(m):
    print("\nAtributos de la matriz:")
    print(f"Forma: {m.shape}")
    print(f"Dimensiones: {m.ndim}D")
    print(f"Tamaño total: {m.size} elementos")
    print(f"Tipo de datos: {m.dtype}")
matriz_2d = matriz_3d.reshape(-1, matriz_3d.shape[-1])
def conv_dataframe(matriz):
    return pd.DataFrame(matriz)

def cargar_mat(path):
    return loadmat(path)

def cargar_csv(path):
    return pd.read_csv(path)

def suma(array, eje=None): 
    return np.sum(array, axis=eje)

def rest(array, eje=None): 
    return np.subtract.reduce(array, axis=eje) if len(array) > 1 else array

def mult(array, eje=None): 
    return np.prod(array, axis=eje)

def div(array, eje=None): 
    return np.divide(array[0], array[1]) if len(array) == 2 else array

def log(array): 
    return np.log(array)

def media(array, eje=None): 
    return np.mean(array, axis=eje)

def des(array, eje=None): 
    return np.std(array, axis=eje)