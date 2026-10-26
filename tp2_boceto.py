#ANALISIS EXPLORATORIO

import numpy as np
import matplotlib.pyplot as plt
import duckdb as dd
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

SEMILLA = 21

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
#c)
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
#OBS: 
    #mean(axis=0) → hace promedio de cada columna: promedio de cada píxel
    #mean(axis=1) → hace promedio de cada fila: promedio de cada imagen

#Graficamos la diferencia entre la media de los pixeles de la clase 3(pez) y 5(colectivo)

pixeles_clase3 = imagenes[imagenes["label"] == 3].drop(columns=["label"])
clase_3_media = pixeles_clase3.mean(axis=0)

pixeles_clase5 = imagenes[imagenes["label"] == 5].drop(columns=["label"])
clase_5_media = pixeles_clase5.mean(axis=0)

diferencia_intensidad = (clase_3_media - clase_5_media)

eje_pixeles = np.arange(1,785)
plt.plot(eje_pixeles, diferencia_intensidad)
plt.xlabel("Píxeles")
plt.ylabel("Diferencia de intensidad media")
plt.title("Diferencia de intensidad media de pixeles entre pez y colectivo")
plt.axhline(0, color="black", linestyle="--")
plt.show()


#Graficamos la diferencia ahora con con un scatterplot
eje_pixeles = np.arange(1, 785)

plt.figure(figsize=(14, 5))

plt.scatter(
    eje_pixeles,
    diferencia_intensidad,
    s=10,
    alpha=0.7
)

plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xlabel("Píxel")
plt.ylabel("Diferencia de intensidad media (clase 3 - clase 5)")
plt.title("Diferencia de intensidad media por píxel entre las clases 3 y 5")
plt.xlim(1, 784)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Graficamos cada clase con linea
eje_pixeles = np.arange(1,785)
plt.plot(eje_pixeles, clase_5_media, clase_3_media)
plt.xlabel("Píxeles")
plt.ylabel("Intensidad media")
plt.title("Intensidad media de pixeles pez y colectivo")
plt.axhline(0, color="black", linestyle="--")
plt.show()

#Graficamos cada clase con scatterplot

#Idea: si supieramos como acceder a los pixeles que queremos podriamos hacer un grafico que en vez de poner
#todos los pixeles en el eje x, comparemos por bloques que nos interesan

eje_pixeles = np.arange(1, 785)

fig, ejes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ejes[0].scatter(
    eje_pixeles,
    clase_3_media,
    s=10,
    alpha=0.7,
    color="blue"
)
ejes[0].set_title("Intensidad media por píxel - Clase 3")
ejes[0].set_ylabel("Intensidad media")
ejes[0].grid(True, alpha=0.3)

ejes[1].scatter(
    eje_pixeles,
    clase_5_media,
    s=10,
    alpha=0.7,
    color="red"
)
ejes[1].set_title("Intensidad media por píxel - Clase 5")
ejes[1].set_xlabel("Píxel")
ejes[1].set_ylabel("Intensidad media")
ejes[1].grid(True, alpha=0.3)

ejes[1].set_xlim(1, 784)

plt.tight_layout()
plt.show()

# Graficamos cada clase con scatterplot en un mismo grafico
plt.figure(figsize=(14,5))
plt.scatter(
    eje_pixeles, clase_3_media, color ="blue", label="Clase 2"
)
plt.scatter(
    eje_pixeles, clase_5_media, color ="green", label="Clase 5"
)

plt.xlabel("Píxel")
plt.ylabel("Intensidad media")
plt.title("Intensidad media por píxel: clases 2 y 9")
plt.xlim(1, 784)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% Hacemos graficos luego de las correcciones de Juan

# Diferencia entre pez y colectivo con linea y sin negativos ordenados
pixeles_clase3 = imagenes[imagenes["label"] == 3].drop(columns=["label"])
clase_3_media = pixeles_clase3.mean(axis=0)

pixeles_clase5 = imagenes[imagenes["label"] == 5].drop(columns=["label"])
clase_5_media = pixeles_clase5.mean(axis=0)

diferencia_intensidad = (clase_3_media - clase_5_media)**2
diferencia_intensidad = diferencia_intensidad.sort_values(ascending=False)

eje_pixeles = np.arange(1,785)
plt.plot(eje_pixeles, diferencia_intensidad)
plt.xlabel("Píxeles")
plt.ylabel("Diferencia de intensidad media")
plt.title("Diferencia de intensidad media de pixeles entre pez y colectivo")
plt.axhline(0, color="black", linestyle="--")
plt.show()

# Diferencia entre pez y circulo con linea y sin negativos ordenados
pixeles_clase3 = imagenes[imagenes["label"] == 3].drop(columns=["label"])
clase_3_media = pixeles_clase3.mean(axis=0)

pixeles_clase0 = imagenes[imagenes["label"] == 0].drop(columns=["label"])
clase_0_media = pixeles_clase0.mean(axis=0)

diferencia_intensidad = (clase_3_media - clase_0_media)**2
diferencia_intensidad = diferencia_intensidad.sort_values(ascending=False)

eje_pixeles = np.arange(1,785)
plt.plot(eje_pixeles, diferencia_intensidad)
plt.xlabel("Píxeles")
plt.ylabel("Diferencia de intensidad media")
plt.title("Diferencia de intensidad media de pixeles entre pez y circulo")
plt.axhline(0, color="black", linestyle="--")
plt.show()
#UNIFICAR LAS DOS RAYAS

"""
# acordarnos de promediar todos los pixeles para asignarle un unico valor a la imagen de la diferencia,
# asi podemos comparar en un dato cuan distintas o parecidas son las imagenes (es raro pero recomendacion de juan, a chequear)
clase_3_resumida = clase_3_media.mean() 
clase_5_resumida = clase_5_media.mean() 
clase_0_resumida = clase_0_media.mean() 

diferencia_resumida_3y5 = abs(clase_3_resumida - clase_5_resumida) #15.863169305031313
diferencia_resumida_3y0 = abs(clase_3_resumida - clase_0_resumida) #1.8753571950206123
"""


#%%============================================================================
#Quién es quién?

matriz = np.zeros((28,28))




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
# Ejercicio 2

clase2y9 = imagenes[(imagenes["label"] == 2) | (imagenes["label"] == 9)]

balance_sql = """
                SELECT label, COUNT(*) AS cantidad
                FROM clase2y9
                GROUP BY label
                """
balance_2y9 = dd.sql(balance_sql).df()
print(balance_2y9)



# clases 2 y 9 difieren de 140 imagenes, consideramos que están balanceadas

#  como hay balance, usamos accuracy (2b) TP + TN  / TP + TN + FP + FN, 
#justificar en el docs por que elegimos esa
# positivo: clase 2 (cuadrado), negativo: clase 9 (casa)
# creemos que es más grave FP, que diga que es un cuadrado y sea una casa
# (que tenga un triangulo arriba del cuadrado y no lo vea), PRECISION alto

#%%===================================================================
# 2c) separar en conjunto de entrenamiento - testeo
# para evitar que quede toda una clase en uno de los conjuntos, separamos las clases en distintos dfs
# y sacamos cierta proporción de cada una al azar, luego unimos todo y obtenemos los conjuntos que vamos a usar

datos = clase2y9.drop(columns=["label"])

clases = clase2y9["label"]

datos_train, datos_test, clases_train, clases_test = train_test_split(
    
    datos, clases, test_size = 0.2, random_state = SEMILLA, stratify = clases
    )

print(datos_train.shape)

#%%===================================================================
# 2d, elegimos k=5
# PENDIENTE, ver los conjuntos de pixeles que elegimos
clase2 = datos_train[clases_train == 2]
clase9 = datos_train[clases_train == 9]

diferencia = abs(clase2.mean() - clase9.mean())
ranking_pixeles_dif = diferencia.sort_values(ascending=False)
#print(ranking_pixeles_dif[:10])
'''
# knn con k=5 y los top 3 pixeles del ranking de diferencias
pixeles = ranking_pixeles_dif.index[:3]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(datos_train[pixeles], clases_train)
prediccion = knn.predict(datos_test[pixeles])

#testeamos accuracy y precision
acc = accuracy_score(clases_test, prediccion)
prec = precision_score(clases_test,prediccion, pos_label=2)

print("del 1 al 3")
print(acc) # 0,84
print(prec) # 0,9
#hay mas falsos positivos que falsos negativos
#print(clases_test)
#print(prediccion)


#PROBAMOS CON OTROS PIXELES
pixeles = ranking_pixeles_dif.index[3:6]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(datos_train[pixeles], clases_train)
prediccion = knn.predict(datos_test[pixeles])

#testeamos accuracy y precision
acc = accuracy_score(clases_test, prediccion)
prec = precision_score(clases_test,prediccion, pos_label=2)

print("del 3 al 6")
print(acc) #0,85
print(prec) #0,89


#PROBAMOS CON OTROS PIXELES
pixeles = ranking_pixeles_dif.index[6:9]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(datos_train[pixeles], clases_train)
prediccion = knn.predict(datos_test[pixeles])

#testeamos accuracy y precision
acc = accuracy_score(clases_test, prediccion)
prec = precision_score(clases_test,prediccion, pos_label=2)

print("del 6 al 9")
print(acc) #0,51
print(prec) #0,51


#PROBAMOS CON OTROS PIXELES (MAS ABAJO EN EL RANKING)
pixeles = ranking_pixeles_dif.index[100:103]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(datos_train[pixeles], clases_train)
prediccion = knn.predict(datos_test[pixeles])

#testeamos accuracy y precision
acc = accuracy_score(clases_test, prediccion)
prec = precision_score(clases_test,prediccion, pos_label=2)

print("del 100 al 102")
print(acc) #0,86
print(prec) #0,84


#PROBAMOS CON OTROS PIXELES (ULTIMOS 3)
pixeles = ranking_pixeles_dif.index[-3:]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(datos_train[pixeles], clases_train)
prediccion = knn.predict(datos_test[pixeles])

#testeamos accuracy y precision
acc = accuracy_score(clases_test, prediccion)
prec = precision_score(clases_test,prediccion, pos_label=2)

print("ultimos 3")
print(acc) #0,51
print(prec) #0,54


#PROBAMOS CON OTROS PIXELES (al azar)
pixeles = datos_train.sample(3, axis=1, random_state = SEMILLA).columns
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(datos_train[pixeles], clases_train)
prediccion = knn.predict(datos_test[pixeles])

#testeamos accuracy y precision
acc = accuracy_score(clases_test, prediccion)
prec = precision_score(clases_test,prediccion, pos_label=2)

print("al azar")
print(acc) #0,62
print(prec) #0,58
'''

#EJERCICIO 2e
ks = [2,3,4,5,10,50]
conjuntos = {
    'primeros 3' : ranking_pixeles_dif.index[:3],
    'primeros 5': ranking_pixeles_dif.index[:5],
    'primeros 9': ranking_pixeles_dif.index[:9],
    'del 6 al 8 (elegido)': ranking_pixeles_dif.index[6:9],
    'del 100 al 102': ranking_pixeles_dif.index[100:103],
    
}
#print(conjuntos)

resultados = []
for conjunto in conjuntos.items():
    for k in ks:
        pixeles = conjunto[1]
        #print(conjunto[0])
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(datos_train[pixeles], clases_train)
        prediccion = knn.predict(datos_test[pixeles])
        #testeamos accuracy y precision
        #print(f"k = {k}")
        acc = accuracy_score(clases_test, prediccion)
        #print(f"accuracy = {acc}")
        prec = precision_score(clases_test,prediccion, pos_label=2)
        #print(f"precision = {prec}")
        resultados.append((conjunto[0], len(pixeles), k, acc, prec))

resultados_df = pd.DataFrame(resultados, columns= ['atributos', 'cantidad atributos', 'K', 'accuracy', 'precision'])
resultados_df = resultados_df.round(3)
print(resultados_df)

#%%===================================================================================
#EJERCICIO 3
#3a: dividimos en dev y held out
datos = imagenes.drop(columns=["label"])

clases = imagenes["label"]

datos_dev, datos_held, clases_dev, clases_held = train_test_split(
    
    datos, clases, test_size = 0.2, random_state = SEMILLA, stratify = clases
    )

print(datos_dev.shape)
print(datos_held.shape)
print(imagenes.shape)
# %%==========================================================================
#3c: arboles de decision
#Partimos en k fold
cantidad_folds = 10
particion_folds = StratifiedKFold(n_splits=cantidad_folds, shuffle=True, random_state=SEMILLA)
'''
#Arbol 1: gini y best
profundidades = [2, 5, 7, 10]
for prof in profundidades:
    accuracy_por_fold = []
    for indices_train, indices_test in particion_folds.split(datos_dev, clases_dev):

        #volvemos a la division simplificada como en el punto 2, en cada vuelta del kfold
        datos_train = datos_dev.iloc[indices_train]
        clases_train = clases_dev.iloc[indices_train]
        datos_test = datos_dev.iloc[indices_test]
        clases_test = clases_dev.iloc[indices_test]

        arbol_prueba = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=prof, random_state=SEMILLA)
        arbol_prueba.fit(datos_train, clases_train)
        clases_decididas = arbol_prueba.predict(datos_test)

        accuracy_por_fold.append(accuracy_score(clases_test, clases_decididas))

    print(f"arbol profundidad{prof}")
    accuracy_promedio = np.mean(accuracy_por_fold)
    print(f"accuracy promedio = {accuracy_promedio}")
    print(f"accuracy por fold = {accuracy_por_fold}")
'''
#%%
#Arbol 2: entropia
profundidades = [2, 5, 7, 10]
for prof in profundidades:
    accuracy_por_fold = []
    for indices_train, indices_test in particion_folds.split(datos_dev, clases_dev):

        #volvemos a la division simplificada como en el punto 2, en cada vuelta del kfold
        datos_train = datos_dev.iloc[indices_train]
        clases_train = clases_dev.iloc[indices_train]
        datos_test = datos_dev.iloc[indices_test]
        clases_test = clases_dev.iloc[indices_test]

        arbol_prueba = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=prof, random_state=SEMILLA)
        arbol_prueba.fit(datos_train, clases_train)
        clases_decididas = arbol_prueba.predict(datos_test)

        accuracy_por_fold.append(accuracy_score(clases_test, clases_decididas))

    print(f"arbol profundidad{prof}")
    accuracy_promedio = np.mean(accuracy_por_fold)
    print(f"accuracy promedio = {accuracy_promedio}")
    print(f"accuracy por fold = {accuracy_por_fold}") 

#%% 
#Arbol 3: falta cambiar lo que queramos
profundidades = [2, 5, 7, 10]
for prof in profundidades:
    accuracy_por_fold = []
    for indices_train, indices_test in particion_folds.split(datos_dev, clases_dev):

        #volvemos a la division simplificada como en el punto 2, en cada vuelta del kfold
        datos_train = datos_dev.iloc[indices_train]
        clases_train = clases_dev.iloc[indices_train]
        datos_test = datos_dev.iloc[indices_test]
        clases_test = clases_dev.iloc[indices_test]

        arbol_prueba = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=prof, random_state=SEMILLA)
        arbol_prueba.fit(datos_train, clases_train)
        clases_decididas = arbol_prueba.predict(datos_test)

        accuracy_por_fold.append(accuracy_score(clases_test, clases_decididas))

    print(f"arbol profundidad{prof}")
    accuracy_promedio = np.mean(accuracy_por_fold)
    print(f"accuracy promedio = {accuracy_promedio}")
    print(f"accuracy por fold = {accuracy_por_fold}")