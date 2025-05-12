import numpy as np
import pandas as pd
from scipy.io import loadmat
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

def suma(array, eje=None): return np.sum(array, axis=eje)

def rest(array, eje=None): return np.subtract.reduce(array, axis=eje) if len(array) > 1 else array

def mult(array, eje=None): return np.prod(array, axis=eje)

def div(array, eje=None): return np.divide(array[0], array[1]) if len(array) == 2 else array

def log(array): return np.log(array)

def media(array, eje=None): return np.mean(array, axis=eje)

def des(array, eje=None): return np.std(array, axis=eje)

def analisis_kaggle_csv(ruta_csv, columnas, operacion):
    try:
        df = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        raise FileNotFoundError(f"El archivo {ruta_csv} no se encuentra en la ruta proporcionada.")

    if not all(col in df.columns for col in columnas):
        raise ValueError("Una o más columnas no existen en el archivo")

    if operacion == 'suma':
        resultado = df[columnas].sum()
    elif operacion == 'resta':
        resultado = df[columnas[0]] - df[columnas[1]]
    elif operacion == 'multiplicacion':
        resultado = df[columnas].prod(axis=1)
    elif operacion == 'division':
        resultado = df[columnas[0]] / df[columnas[1]]
    elif operacion == 'logaritmo':
        resultado = np.log(df[columnas])
    elif operacion == 'promedio':
        resultado = df[columnas].mean()
    elif operacion == 'desviacion':
        resultado = df[columnas].std()
    else:
        raise ValueError("Operación no válida")

    print(f"\nResultado de '{operacion}' sobre {columnas}:")
    print(resultado)
    return resultado
def conf_gra(titulo, xlabel, ylabel):
    plt.title(titulo, fontsize=14, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def ask_titulos():
    titulo = input("Título del gráfico [default: 'Gráfico']: ") or "Gráfico"
    x = input("Etiqueta eje X [default: 'X']: ") or "X"
    y = input("Etiqueta eje Y [default: 'Y']: ") or "Y"
    return titulo, x, y