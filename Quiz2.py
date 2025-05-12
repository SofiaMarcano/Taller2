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
def gra_histo(data):
    titulo, x, y = ask_titulos()
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, color='#2ecc71', edgecolor='#27ae60', alpha=0.7)
    conf_gra(titulo, x, y)

def gra_stem(data):
    plt.figure(figsize=(10, 5))
    markerline, stemlines, baseline = plt.stem(data, linefmt='#2980b9', markerfmt='o', basefmt='#7f8c8d', use_line_collection=True)
    plt.setp(markerline, markersize=6, markeredgecolor='#2c3e50', markerfacecolor='#e74c3c')
    plt.setp(stemlines, linewidth=0.8, alpha=0.6)
    titulo, x, y = ask_titulos()
    conf_gra(titulo, x, y)

def graf_barras(data):
    titulo, x, y = ask_titulos()
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(data)), data, color='#9b59b6', alpha=0.7)
    conf_gra(titulo, x, y)

def graf_pie(data, column='Outcome', title=None, colors=None):
    counts = data[column].value_counts()
    labels = ['No Diabetes', 'Diabetes'] if column == 'Outcome' else counts.index.astype(str)
    if colors is None:
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(counts)]
    plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, shadow=True, explode=[0.05] + [0]*(len(counts)-1))
    plt.title(title or f"Distribución de {column}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()
