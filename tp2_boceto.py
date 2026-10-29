"""
ENCABEZADO
"""
import numpy as np
import matplotlib.pyplot as plt
import duckdb as dd
import pandas as pd
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score 
from sklearn.tree import DecisionTreeClassifier


SEMILLA = 21

carpeta_principal = os.path.dirname(os.path.abspath(__file__))
clases = pd.read_csv(os.path.join(carpeta_principal, "clases.csv"))
imagenes = pd.read_csv(os.path.join(carpeta_principal, "imagenes.csv"))
evaluacion = pd.read_csv(os.path.join(carpeta_principal, "evaluacion.csv"))

#%%=====================================================================================================
#Ejercicio 1 - EDA
datos = imagenes.drop(columns=['label'])
labels = imagenes['label']

print("Cantidad de imágenes", imagenes.shape[0])
print("Cantidad de pixeles", imagenes.shape[1])
print("Valores de píxeles: mínimo", imagenes.values.min(), "- máximo", imagenes.values.max())

# Vemos el balance de todas las clases, cuántas imagenes hay de cada una, como parte del analisis exploratorio
#Vemos el balance de las clases
consulta_sql = """
                SELECT label, COUNT(*) AS cantidad_dibujos
                FROM imagenes
                GROUP BY label
                ORDER BY cantidad_dibujos DESC

"""

dataframe = dd.sql(consulta_sql).df() # SACAR INDEX
print(dataframe)

#Cada clase tiene aproximadamente la misma cantidad de dibujos: 
#MAX = 3120(6), MIN = 2870 (5) 

#%%======================================================================================
#Intentamos graficar una fila = un dibujo
# Plot imagen

copia_imagenes = imagenes.drop(columns=['label'])
labels = imagenes['label']

#print(copia_imagenes[:100])
img = np.array(copia_imagenes.iloc[8]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()

# %% ====================================================================================
# a) 

#Nos interesa ver si hay renglones en las imagenes que siempre estan vacios (columnas en el csv) 
#Podemos hacer un mapa de color para comparar en qué pixeles se dibuja más
#Cuidado: si hay muchas imagenes de una clase puede estar influenciado el mapa de calor generico

#IMAGEN PROMEDIO EN HEATMAP
matriz_pixeles = copia_imagenes.to_numpy(dtype=float)

promedio_pixeles = matriz_pixeles.mean(axis=0)

mapa_calor = promedio_pixeles.reshape((28, 28))

plt.figure(figsize=(8, 6))
plt.imshow(mapa_calor, cmap="inferno")
plt.colorbar(label="Intensidad promedio")
plt.title("Mapa de calor de los píxeles")
plt.show()


datos = imagenes.drop(columns=['label'])
labels = imagenes['label']

#vemos desvio estandar por pixel, función std()
desvio_por_pixel = datos.std()
print("Pixeles con desvío 0:", (desvio_por_pixel == 0).sum(), "de", datos.shape[1])
# No existen pixeles con desvio 0, si bien en el mapa de calor vimos a los margenes como descartables, 
# esto es relativo dependiendo la clase


#%%====================================================================================================000
#b) clases parecidas

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



#%% =================================================================================================================
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


#%%============================================================================
# 1c) dispersion dentro de una clase

# IMAGENES DE MUESTRA

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

# Diez imágenes de la clase 0
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

# Diez imágenes de la clase 8
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


# DISPERSION DE LAS CLASES CON BOXPLOT
datos = imagenes.drop(columns=['label'])
labels = imagenes['label']
imagen_promedio_por_clase = datos.groupby(labels).mean()


clases_a_comparar = [3,0,8]
distancias_a_su_media = {}

for clase in clases_a_comparar:
    imagenes_clase = datos[labels == clase]
    diferencias = imagenes_clase - imagen_promedio_por_clase.loc[clase]
    distancias_a_su_media[clase] = np.sqrt((diferencias **2 ).sum(axis=1)).values

plt.boxplot(distancias_a_su_media.values(), labels=[f'Clase {c}' for c in clases_a_comparar])
plt.ylabel('Distancia  de cada imagen a la imagen promedio de su clase')
plt.title('Dispersión dentro de cada clase')
plt.show()
#%%%===========================================================================================================
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

#%%==============================================================================================================================
# 2c) separar en conjunto de entrenamiento - testeo
# para evitar que quede toda una clase en uno de los conjuntos, separamos las clases en distintos dfs
# y sacamos cierta proporción de cada una al azar, luego unimos todo y obtenemos los conjuntos que vamos a usar

datos = clase2y9.drop(columns=["label"])

clases = clase2y9["label"]

datos_train, datos_test, clases_train, clases_test = train_test_split(
    
    datos, clases, test_size = 0.2, random_state = SEMILLA, stratify = clases
    )

print(datos_train.shape)

#%%======================================================================================================================================
# 2d), elegimos k=5

clase2 = datos_train[clases_train == 2]
clase9 = datos_train[clases_train == 9]

diferencia = abs(clase2.mean() - clase9.mean())
ranking_pixeles_dif = diferencia.sort_values(ascending=False)
#print(ranking_pixeles_dif[:10])


conjuntos_2_pix = {
    'del 1 al 2' : ranking_pixeles_dif.index[:2],
    'del 2 al 4': ranking_pixeles_dif.index[2:4],
    'del 4 al 6': ranking_pixeles_dif.index[4:6],
    'del 100 al 102': ranking_pixeles_dif.index[100:102],
    'últimos 2': ranking_pixeles_dif.index[-2:],
    '2 random': datos_train.sample(2, axis=1, random_state = SEMILLA).columns
}

conjuntos_3_pix = {
    'del 1 al 3' : ranking_pixeles_dif.index[:3],
    'del 3 al 6': ranking_pixeles_dif.index[3:6],
    'del 6 al 8': ranking_pixeles_dif.index[6:9],
    'del 100 al 103': ranking_pixeles_dif.index[100:103],
    'últimos 3': ranking_pixeles_dif.index[-3:],
    '3 random': datos_train.sample(3, axis=1, random_state = SEMILLA).columns
}


conjuntos_5_pix = {
    'del 1 al 5' : ranking_pixeles_dif.index[:5],
    'del 5 al 10': ranking_pixeles_dif.index[5:10],
    'del 10 al 15': ranking_pixeles_dif.index[10:15],
    'del 100 al 105': ranking_pixeles_dif.index[100:105],
    'últimos 5': ranking_pixeles_dif.index[-5:],
    '5 random': datos_train.sample(5, axis=1, random_state = SEMILLA).columns
}

diccionarios =[conjuntos_2_pix, conjuntos_3_pix,  conjuntos_5_pix]

resultados = []
for diccionario in diccionarios:

    for conjunto in diccionario.items():
        
        pixeles = conjunto[1]
        #print(conjunto[0])
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(datos_train[pixeles], clases_train)
        prediccion = knn.predict(datos_test[pixeles])
        #testeamos accuracy y precision
        #print(f"k = {k}")
        acc = accuracy_score(clases_test, prediccion)
        #print(f"accuracy = {acc}")
        prec = precision_score(clases_test,prediccion, pos_label=2)
        #print(f"precision = {prec}")
        resultados.append((conjunto[0], len(pixeles), acc, prec))


resultados_df = pd.DataFrame(resultados, columns= ['atributos', 'cantidad atributos', 'accuracy', 'precision'])
resultados_df = resultados_df.round(3)
print(resultados_df)



#%%#======================================================================================================
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
#3a): dividimos en dev y held out
datos = imagenes.drop(columns=["label"])

clases = imagenes["label"]

datos_dev, datos_held, clases_dev, clases_held = train_test_split(
    
    datos, clases, test_size = 0.2, random_state = SEMILLA, stratify = clases
    )

print(datos_dev.shape)
print(datos_held.shape)
print(imagenes.shape)

#3b) matriz docs

# %%==========================================================================
#3c): arboles de decision
#Partimos en k fold
cantidad_folds = 10
particion_folds = StratifiedKFold(n_splits=cantidad_folds, shuffle=True, random_state=SEMILLA)
"""
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
"""
#%%=====================================================================================
#Arbol 2: entropia ELEGIDO! :)
"""
profundidades = [2, 5, 7, 10]
for prof in profundidades:
    accuracy_por_fold = []

    # A continuacion variamos el fold que probamos vs entrenamos:
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
"""

#%% ================================================================================================
#Arbol 3: falta cambiar lo que queramos
"""
profundidades = [2, 5, 7, 10]
for prof in profundidades:
    accuracy_por_fold = []
    for indices_train, indices_test in particion_folds.split(datos_dev, clases_dev):

        #volvemos a la division simplificada como en el punto 2, en cada vuelta del kfold
        datos_train = datos_dev.iloc[indices_train]
        clases_train = clases_dev.iloc[indices_train]
        datos_test = datos_dev.iloc[indices_test]
        clases_test = clases_dev.iloc[indices_test]

        arbol_prueba = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=prof, random_state=SEMILLA, min_samples_leaf=5)
        arbol_prueba.fit(datos_train, clases_train)
        clases_decididas = arbol_prueba.predict(datos_test)

        accuracy_por_fold.append(accuracy_score(clases_test, clases_decididas))

    print(f"arbol profundidad{prof}")
    accuracy_promedio = np.mean(accuracy_por_fold)
    print(f"accuracy promedio = {accuracy_promedio}")
    print(f"accuracy por fold = {accuracy_por_fold}")

"""

    #%%#=======================================================================================
#3 d) Nuestro mejor arbol es el de prof 10


arbol_final = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=10, random_state=SEMILLA)

#lo entreno con todo el conjunto de desarrollo (dev):
arbol_final.fit(datos_dev, clases_dev) 

#lo testeo con los datos que NUNCA toqué (held-out)
clases_decididas = arbol_final.predict(datos_held) #devuelve una lista con las clases predecidas de cada imagen

#calculamos la accuracy para el held
accuracy_arbol_final = accuracy_score(clases_held, clases_decididas)
print(clases_decididas, accuracy_arbol_final)

# La accuracy de nuestro arbol final, medida con el held out nos dio 0,792333...

#%%#===============================================================================================
# 3) e) NOS FALTA RUNNEARLO...MAÑANA CHEQUEAMOS EN CLASE

# La accuracy esperada para predecir el conjunto de evaluacion de la catedra es 0,792333...
# Ya que a este punto, entreno con el mismo conjunto de datos, y entre el heldout y la evaluacion, no hay ningun entrenamiento adicional
 
#clases_decididas = arbol_final.predict(evaluacion)

    #%%#=======================================================================================================