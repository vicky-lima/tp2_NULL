'''
 Grupo: "NULL"
 Integralentes: Lucía Murillo, Victoria Florencia Lima, María Cecilia Marchesano y Emma Victoria Tripaldi
 Archivo con el código del analisis exploratorio de los datos proporcionados por la materia, junto con los
 ejercicios de elección, entrenamiento y desempeño del modelo de predicción de clases.
'''

import duckdb as dd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score 
from sklearn.tree import DecisionTreeClassifier

SEMILLA = 21

#%%============================================================================================================
#Leemos los archivos
carpeta_principal = os.path.dirname(os.path.abspath(__file__))
clases = pd.read_csv(os.path.join(carpeta_principal, "clases.csv"))
imagenes = pd.read_csv(os.path.join(carpeta_principal, "imagenes.csv"))
evaluacion = pd.read_csv(os.path.join(carpeta_principal, "evaluacion.csv"))

#%%============================================================================================================
#Ejercicio 1 - Análisis exploratorio de los datos
#separamos las clases, de los pixeles
pix_imagenes = imagenes.drop(columns=['label'])
labels = imagenes['label']
#vemos cuántas imagenes hay, cuántos píxeles tiene cada imágen y el rango de intensidades de los píxeles
print("Cantidad de imágenes", imagenes.shape[0])
print("Cantidad de pixeles", pix_imagenes.shape[1])
print("Valores de píxeles: mínimo", pix_imagenes.values.min(), "- máximo", pix_imagenes.values.max())

#Vemos cuántas imagenes hay de cada clase
#Vemos si las clases están balanceadas
cantidad_imagenes_por_clase_sql = """
                SELECT label, COUNT(*) AS cantidad_dibujos
                FROM imagenes
                GROUP BY label
                ORDER BY cantidad_dibujos DESC
"""

cant_imagenes_por_clase_df = dd.sql(cantidad_imagenes_por_clase_sql).df() # SACAR INDEX
print(cant_imagenes_por_clase_df)
#%%============================================================================================================
#Plot imagen aleatoria (elegimos el 8 porque si)
img = np.array(pix_imagenes.iloc[8]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()

#%%============================================================================================================
#1a) atributos relevantes para predecir el objeto al que corresponde el dibujo
# Nos interesa ver si hay renglones en las imagenes que siempre estan vacios (columnas en el csv) 
# Hacemos un mapa de calor para comparar en qué pixeles se dibuja más (según la media por pixel)
# Cuidado: si hay muchas imagenes de una clase puede estar influenciado el mapa de calor generico (pero sabemos que están balanceadas)

matriz_pixeles = pix_imagenes.to_numpy(dtype=float)

media_pixeles = matriz_pixeles.mean(axis=0)

mapa_calor = media_pixeles.reshape((28, 28))

plt.figure(figsize=(8, 6))
plt.imshow(mapa_calor, cmap="inferno")
plt.colorbar(label="Intensidad promedio")
plt.title("Mapa de calor de los píxeles")
plt.show()

#se puede ver que los bordes tienen intensidad baja

# Como la media puede estar influenciada por valores extremos, vemos desvio estandar por pixel, función std()
desvio_est_por_pixel = pix_imagenes.std()
print("Pixeles con desvío 0:", (desvio_est_por_pixel == 0).sum(), "de", pix_imagenes.shape[1])
# El desvio estandar igual a cero evalua si ese pixel tiene la misma intensidad en todas las imagenes
# No existen pixeles con desvio 0, por lo tanto no podemos descartar los bordes de las imagenes
#%%============================================================================================================
#1b) clases parecidas

# Plot mapas de calor de todas las clases en una sola figura
labels = pd.to_numeric(labels, errors="raise")

mapas_calor = []

for i in range(10):
    clase = imagenes[imagenes["label"] == i]
    matriz_pixeles = clase.drop(columns=["label"]).to_numpy(dtype=float)
    media_pixeles = matriz_pixeles.mean(axis=0)
    mapas_calor.append(media_pixeles.reshape((28, 28)))

valor_minimo = min(mapa.min() for mapa in mapas_calor)
valor_maximo = max(mapa.max() for mapa in mapas_calor)

fig, ejes = plt.subplots(
    2,
    5,
    figsize=(14, 6),
    constrained_layout=True,
)

for i, ax in enumerate(ejes.flat):
    imagen = ax.imshow(
        mapas_calor[i],
        cmap="inferno",
        vmin=valor_minimo,
        vmax=valor_maximo,
    )
    ax.set_title(f"Clase {i}")
    ax.axis("off")

fig.colorbar(
    imagen,
    ax=ejes.ravel().tolist(),
    label="Intensidad promedio",
    pad=0.02,
)
fig.suptitle("Mapas de calor por clase")
plt.show()

#SIMILITUDES:
#Cuadrado~circulo~remera (vacíos en el centro)
#Pez~colectivo (rectangulo arriba)

#%%============================================================================================================
# Graficamos las diferencias entre la media por pixel de las imágenes de pez vs colectivo y pez vs circulo

#1 calcular la diferencia para Pez (clase 3) vs Colectivo (clase 5)
clase_3_media = imagenes[imagenes["label"] == 3].drop(columns=["label"]).mean(axis=0)
clase_5_media = imagenes[imagenes["label"] == 5].drop(columns=["label"]).mean(axis=0)
diferencia_colectivo = ((clase_3_media - clase_5_media)**2).sort_values(ascending=False)

#2 calcular la diferencia para Pez (clase 3) vs Círculo (clase 0)
clase_0_media = imagenes[imagenes["label"] == 0].drop(columns=["label"]).mean(axis=0)
diferencia_circulo = ((clase_3_media - clase_0_media)**2).sort_values(ascending=False)

#3 graficar ambas curvas en la misma figura
eje_pixeles = np.arange(1, 785)

plt.figure(figsize=(10, 6))
plt.plot(eje_pixeles, diferencia_colectivo.values, label="Pez vs Colectivo", color="dodgerblue", alpha=0.8)
plt.plot(eje_pixeles, diferencia_circulo.values, label="Pez vs Círculo", color="crimson", alpha=0.8)


plt.xlabel("Pixeles")
plt.ylabel("Diferencia de intensidad media al cuadrado")
plt.title("Comparación de diferencia de intensidad media con Pez")
plt.axhline(0, color="black", linestyle="--", alpha=0.5)
plt.legend() 
plt.show()

#%%============================================================================================================
# 1c) dispersion dentro de una clase
#falta dispersion por clase
# IMAGENES DE MUESTRA

# Diez imágenes de las clases 0, 3, 8
clases_muestra = [0, 3, 8]
for c in clases_muestra:
    clase = imagenes[imagenes["label"] == c].drop(columns=["label"])

    cantidad = min(10, len(clase))
    fig, ejes = plt.subplots(2, 5, figsize=(10, 5))

    for i, ax in enumerate(ejes.flat):
        if i < cantidad:
            img = clase.iloc[i].to_numpy(dtype=float).reshape((28, 28))
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Imagen {i + 1}")
        ax.axis("off")

    fig.suptitle(f"Diez imágenes de la clase {c}")
    plt.tight_layout()
    plt.show()

# DISPERSION DE LAS CLASES CON BOXPLOT
imagen_promedio_por_clase = pix_imagenes.groupby(labels).mean()

clases_a_comparar = [0,3,8]
distancias_a_su_media = {}

for clase in clases_a_comparar:
    imagenes_clase = pix_imagenes[labels == clase]
    diferencias = imagenes_clase - imagen_promedio_por_clase.loc[clase]
    distancias_a_su_media[clase] = np.sqrt((diferencias **2 ).sum(axis=1)).values

plt.boxplot(distancias_a_su_media.values(), tick_labels=[f"Clase {c}" for c in clases_a_comparar])
plt.ylabel('Distancia  de cada imagen a la imagen promedio de su clase')
plt.title('Dispersión dentro de cada clase')
plt.show()

#%%============================================================================================================
#Ejercicio 2 - Clasificación binaria
#Cuadrado vs casa
clase2y9 = imagenes[(imagenes["label"] == 2) | (imagenes["label"] == 9)]

balance_sql = """
                SELECT label, COUNT(*) AS cantidad
                FROM clase2y9
                GROUP BY label
                """
balance_2y9 = dd.sql(balance_sql).df()
print(balance_2y9)



# clases 2 y 9 difieren de 140 imagenes, consideramos que están balanceadas

# como hay balance, usamos accuracy (2b) TP + TN  / TP + TN + FP + FN, 
#justificar en el docs por que elegimos esa
# positivo: clase 2 (cuadrado), negativo: clase 9 (casa)
# creemos que es más grave FP, que diga que es un cuadrado y sea una casa
# (que tenga un triangulo arriba del cuadrado y no lo vea), PRECISION alto
#%%============================================================================================================