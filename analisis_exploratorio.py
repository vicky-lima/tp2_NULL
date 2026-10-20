#ANALISIS EXPLORATORIO

import numpy as np
import matplotlib.pyplot as plt
import duckdb as dd
import pandas as pd
import os

carpeta_principal = os.path.dirname(os.path.abspath(__file__))
clases = pd.read_csv(os.path.join(carpeta_principal, "clases.csv"))
imagenes = pd.read_csv(os.path.join(carpeta_principal, "imagenes.csv"))
evaluacion = pd.read_csv(os.path.join(carpeta_principal, "evaluacion.csv"))


#%%=====================================================================================================
#Intentamos graficar una fila = un dibujo
# Plot imagen

copia_imagenes = imagenes.drop(columns=['label'])
labels = imagenes['label']

#print(copia_imagenes[:100])
img = np.array(copia_imagenes.iloc[8]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()
# %%

#Nos interesa ver si hay renglones en las imagenes que siempre estan vacios (columnas en el csv) 
#Podemos hacer un mapa de color para comparar en qué pixeles se dibuja más
#Cuidado: si hay muchas imagenes de una clase puede estar influenciado el mapa de calor generico

matriz_pixeles = copia_imagenes.to_numpy(dtype=float)

promedio_pixeles = matriz_pixeles.mean(axis=0)

mapa_calor = promedio_pixeles.reshape((28, 28))

plt.figure(figsize=(8, 6))
plt.imshow(mapa_calor, cmap="inferno")
plt.colorbar(label="Intensidad promedio")
plt.title("Mapa de calor de los píxeles")
plt.show()


# %%========================================================================================================
#Vemos el balance de las clases
consulta_sql = """
                SELECT label, COUNT(*) AS cantidad_dibujos
                FROM imagenes
                GROUP BY label
                ORDER BY cantidad_dibujos DESC

"""

dataframe = dd.sql(consulta_sql).df()
print(dataframe)

#Cada clase tiene aproximadamente la misma cantidad de dibujos: 
#MAX = 3120(6), MIN = 2870 (5)  

#%%====================================================================================================000
# Mapas de calor de todas las clases en una sola figura

imagenes["label"] = pd.to_numeric(imagenes["label"], errors="raise")

mapas_calor = []

for i in range(10):
    clase = imagenes[imagenes["label"] == i]
    matriz_pixeles = clase.drop(columns=["label"]).to_numpy(dtype=float)
    promedio_pixeles = matriz_pixeles.mean(axis=0)
    mapas_calor.append(promedio_pixeles.reshape((28, 28)))

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
#Cuadrado=circulo~remera (vacíos en el centro)
#Pez=colectivo (rectangulo arriba)

#%%============================================================================
# Diez imágenes de la clase 3
clase3 = imagenes[imagenes["label"] == 3].drop(columns=["label"])

cantidad = min(10, len(clase3))
fig, ejes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(ejes.flat):
    if i < cantidad:
        img = clase3.iloc[i].to_numpy(dtype=float).reshape((28, 28))
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Imagen {i + 1}")
    ax.axis("off")

fig.suptitle("Diez imágenes de la clase 3")
plt.tight_layout()
plt.show()
#%%========================================================================
# Diez imágenes de la clase 3
clase3 = imagenes[imagenes["label"] == 0].drop(columns=["label"])

cantidad = min(10, len(clase3))
fig, ejes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(ejes.flat):
    if i < cantidad:
        img = clase3.iloc[i].to_numpy(dtype=float).reshape((28, 28))
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Imagen {i + 1}")
    ax.axis("off")

fig.suptitle("Diez imágenes de la clase 0")
plt.tight_layout()
plt.show()
#%%==================================================
# Diez imágenes de la clase 3
clase3 = imagenes[imagenes["label"] == 8].drop(columns=["label"])

cantidad = min(10, len(clase3))
fig, ejes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(ejes.flat):
    if i < cantidad:
        img = clase3.iloc[i].to_numpy(dtype=float).reshape((28, 28))
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Imagen {i + 1}")
    ax.axis("off")

fig.suptitle("Diez imágenes de la clase 8")
plt.tight_layout()
plt.show()
#%%%=================================================