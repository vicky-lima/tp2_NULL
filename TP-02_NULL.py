'''
 Grupo: "NULL"
 Integrantes: Lucía Murillo, Victoria Florencia Lima, María Cecilia Marchesano y Emma Victoria Tripaldi
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
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
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
print("\nCantidad de imagenes de cada clase")
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

# Diez imágenes de la clase 3
clase = imagenes[imagenes["label"] == 3].drop(columns=["label"])

cantidad = min(10, len(clase))
fig, ejes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(ejes.flat):
    if i < cantidad:
        img = clase.iloc[i].to_numpy(dtype=float).reshape((28, 28))
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Imagen {i + 1}")
    ax.axis("off")

fig.suptitle(f"Diez imágenes de la clase 3")
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
#2a) nuevo dataframe solo con las imagenes de las clases 2 y 9
imagenes_clases_2y9 = imagenes[(imagenes["label"] == 2) | (imagenes["label"] == 9)]

cant_imagenes_clases_2y9_sql = """
                SELECT label, COUNT(*) AS cantidad
                FROM imagenes_clases_2y9
                GROUP BY label
                ORDER BY label
                """
balance_2y9 = dd.sql(cant_imagenes_clases_2y9_sql).df()

print("\nCantidad de imagenes clases 2 y 9")
print(balance_2y9)

# clases 2 y 9 difieren de 140 imagenes, consideramos que están balanceadas

# como hay balance, usamos accuracy (2b) TP + TN  / TP + TN + FP + FN, 

# positivo: clase 2 (cuadrado), negativo: clase 9 (casa)
# creemos que es más grave FP, que diga que es un cuadrado y sea una casa
# (que tenga un triangulo arriba del cuadrado y no lo vea), PRECISION alto
#%%============================================================================================================
#2c) separar en conjuntos de train y test
# stratify separa balanceando las clases (no hay riesgo de tener todas las imagenes de una clase en solo uno de los dos conjuntos)
datos_2y9 = imagenes_clases_2y9.drop(columns=["label"])

clases_2y9 = imagenes_clases_2y9["label"]

datos_2y9_train, datos_2y9_test, clases_2y9_train, clases_2y9_test = train_test_split(
    
    datos_2y9, clases_2y9, test_size = 0.2, random_state = SEMILLA, stratify = clases_2y9
    )

#%%============================================================================================================
# 2d) Ajustar un modelo de KNN. Elegimos k=5

clase2 = datos_2y9_train[clases_2y9_train == 2]
clase9 = datos_2y9_train[clases_2y9_train == 9]

diferencia_media_por_pixel = abs(clase2.mean() - clase9.mean())
ranking_pixeles_dif = diferencia_media_por_pixel.sort_values(ascending=False)
print("\nRanking de la diferencia de la media por píxeles")
print(ranking_pixeles_dif[:10])

#Armamos conjuntos de x cantidad de pixeles variando los pixeles elegidos del ranking

conjuntos_2_pix = {
    'del 1 al 2' : ranking_pixeles_dif.index[:2],
    'del 2 al 4': ranking_pixeles_dif.index[2:4],
    'del 4 al 6': ranking_pixeles_dif.index[4:6],
    'del 100 al 102': ranking_pixeles_dif.index[100:102],
    'últimos 2': ranking_pixeles_dif.index[-2:],
    '2 random': datos_2y9_train.sample(2, axis=1, random_state = SEMILLA).columns
}

conjuntos_3_pix = {
    'del 1 al 3' : ranking_pixeles_dif.index[:3],
    'del 3 al 6': ranking_pixeles_dif.index[3:6],
    'del 6 al 8': ranking_pixeles_dif.index[6:9],
    'del 100 al 103': ranking_pixeles_dif.index[100:103],
    'últimos 3': ranking_pixeles_dif.index[-3:],
    '3 random': datos_2y9_train.sample(3, axis=1, random_state = SEMILLA).columns
}

conjuntos_5_pix = {
    'del 1 al 5' : ranking_pixeles_dif.index[:5],
    'del 5 al 10': ranking_pixeles_dif.index[5:10],
    'del 10 al 15': ranking_pixeles_dif.index[10:15],
    'del 100 al 105': ranking_pixeles_dif.index[100:105],
    'últimos 5': ranking_pixeles_dif.index[-5:],
    '5 random': datos_2y9_train.sample(5, axis=1, random_state = SEMILLA).columns
}

diccionarios_conj =[conjuntos_2_pix, conjuntos_3_pix,  conjuntos_5_pix]

resultados = []
for diccionario in diccionarios_conj:

    for conjunto in diccionario.items():
        
        pixeles = conjunto[1]
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(datos_2y9_train[pixeles], clases_2y9_train)
        prediccion = knn.predict(datos_2y9_test[pixeles])
        #testeamos accuracy y precision
        acc = accuracy_score(clases_2y9_test, prediccion)
        prec = precision_score(clases_2y9_test,prediccion, pos_label=2)
        resultados.append((conjunto[0], len(pixeles), acc, prec))


resultados_2y9_df = pd.DataFrame(resultados, columns= ['atributos', 'cantidad atributos', 'accuracy', 'precision'])
resultados_2y9_df = resultados_2y9_df.round(3)
print("\nResultados predicción clase 2 y 9 con KNN, k = 5")
print(resultados_2y9_df)

# PLOT PIXELES MEJORES RESULTADOS
# Identificamos a los pixeles candidatos en una grilla de 28x28 que simula la imagen.
# Ponemos un 1 en los del grupo ‘1 a 3’ y un 2 en los del grupo ‘6 a 9’

matriz = np.zeros((28,28))

for pixel in conjuntos_3_pix['del 1 al 3']:
    pixel_numero = int(pixel.replace('pixel', ''))
    fila_pixel = pixel_numero // 28
    columna_pixel = pixel_numero % 28
    matriz[fila_pixel][columna_pixel] = 1

for pixel in conjuntos_3_pix['del 6 al 8']:
    pixel_numero = int(pixel.replace('pixel', ''))
    fila_pixel = pixel_numero // 28
    columna_pixel = pixel_numero % 28
    matriz[fila_pixel][columna_pixel] = 2


colores = ListedColormap(['whitesmoke', 'red', 'royalblue'])
plt.figure(figsize=(6,6))
plt.imshow(matriz, cmap=colores, vmin=-0.5, vmax=2.5, interpolation='nearest')
plt.legend(handles=[Patch(color='red', label= 'Puestos del 1 al 3'),
                    Patch(color='royalblue', label= 'Puestos del 6 al 8')],
                    loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2
                    )
plt.ylabel('Fila')
plt.xlabel('Columna')
plt.title('Ubicación de los píxeles elegidos en la imagen de 28x28')
plt.show()


#%%============================================================================================================
# 2e) comparar modelos de KNN usando distintos atributos y distintos valores de k

ks = [2,3,4,5,10,50]
conjuntos = {
    'primeros 3' : ranking_pixeles_dif.index[:3],
    'primeros 5': ranking_pixeles_dif.index[:5],
    'primeros 9': ranking_pixeles_dif.index[:9],
    'del 6 al 8 (elegido)': ranking_pixeles_dif.index[6:9],
    'del 100 al 102': ranking_pixeles_dif.index[100:103],
}

resultados = []
for conjunto in conjuntos.items():
    for k in ks:
        pixeles = conjunto[1]
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(datos_2y9_train[pixeles], clases_2y9_train)
        prediccion = knn.predict(datos_2y9_test[pixeles])
        #testeamos accuracy y precision
        acc = accuracy_score(clases_2y9_test, prediccion)
        prec = precision_score(clases_2y9_test,prediccion, pos_label=2)
        resultados.append((conjunto[0], len(pixeles), k, acc, prec))

comparacion_modelos_2y9_df = pd.DataFrame(resultados, columns= ['atributos', 'cantidad atributos', 'K', 'accuracy', 'precision'])
comparacion_modelos_2y9_df = comparacion_modelos_2y9_df.round(3)
print("\nComparación modelos KNN predicción clase 2 y 9")
print(comparacion_modelos_2y9_df)

#%%============================================================================================================
#Ejercicio 3 - Clasificación multiclase
# 3a) Separar en conjuntos de desarrollo y control
# dividimos en dev y held out
datos = imagenes.drop(columns=["label"])
clases = imagenes["label"]

datos_dev, datos_held, clases_dev, clases_held = train_test_split(
    
    datos, clases, test_size = 0.2, random_state = SEMILLA, stratify = clases
    )

#%%============================================================================================================
# 3c) comparar y seleccionar árboles de decisión con distintos hiperparámetros, 
# incluyendo distintas profundidades (limitarse a profundidades entre 1 y 10)
# usando k-fold

# Partimos en k fold
cantidad_folds = 10
particion_folds = StratifiedKFold(n_splits=cantidad_folds, shuffle=True, random_state=SEMILLA)

profundidades = [2, 5, 7, 10]
resultados_arboles = []
'''
#Arbol 1: gini y best
arbol = ["gini"]
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
    
    accuracy_promedio = np.mean(accuracy_por_fold)
    arbol.append(accuracy_promedio)
resultados_arboles.append(arbol)

#Arbol 2: entropia ELEGIDO!
arbol = ["entropia"]
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

    accuracy_promedio = np.mean(accuracy_por_fold)
    arbol.append(accuracy_promedio)
resultados_arboles.append(arbol)

#Arbol 3: gini, min_samples_leaf = 5
arbol = ["gini con min_samples_leaf"]
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

    accuracy_promedio = np.mean(accuracy_por_fold)
    arbol.append(accuracy_promedio)
resultados_arboles.append(arbol)

#Arbol 4: entropia, min_samples_leaf = 5
arbol = ["entropia con min_samples_leaf"]
for prof in profundidades:
    accuracy_por_fold = []
    for indices_train, indices_test in particion_folds.split(datos_dev, clases_dev):

        #volvemos a la division simplificada como en el punto 2, en cada vuelta del kfold
        datos_train = datos_dev.iloc[indices_train]
        clases_train = clases_dev.iloc[indices_train]
        datos_test = datos_dev.iloc[indices_test]
        clases_test = clases_dev.iloc[indices_test]

        arbol_prueba = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=prof, random_state=SEMILLA, min_samples_leaf=5)
        arbol_prueba.fit(datos_train, clases_train)
        clases_decididas = arbol_prueba.predict(datos_test)

        accuracy_por_fold.append(accuracy_score(clases_test, clases_decididas))

    accuracy_promedio = np.mean(accuracy_por_fold)
    arbol.append(accuracy_promedio)
resultados_arboles.append(arbol)

resultados_arboles_df = pd.DataFrame(resultados_arboles, columns= ['hiperparametros', 'prof 2', 'prof 5', 'prof 7', 'prof 10'])
resultados_arboles_df = resultados_arboles_df.round(4)
print("\nResultados accuracy clasificación con arboles binarios")
print(resultados_arboles_df)
'''

#%%============================================================================================================
# Matriz de confusion del árbol elegido con el conjunto de entrenamiento

arbol_prueba = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=10, random_state=SEMILLA)
predicciones_fold = []
clases_test_fold = []

for indices_train, indices_test in particion_folds.split(datos_dev, clases_dev):

    #volvemos a la division simplificada como en el punto 2, en cada vuelta del kfold
    datos_train = datos_dev.iloc[indices_train]
    clases_train = clases_dev.iloc[indices_train]
    datos_test = datos_dev.iloc[indices_test]
    clases_test = clases_dev.iloc[indices_test]

    arbol_prueba.fit(datos_train, clases_train)
    clases_decididas = arbol_prueba.predict(datos_test)
    predicciones_fold += list(clases_decididas)
    clases_test_fold += list(clases_test)

matriz_de_confusion = confusion_matrix(clases_test_fold , predicciones_fold)
print("\nMatriz de confusión de los datos de entrenamiento")
print(matriz_de_confusion)

#%%============================================================================================================
# 3d) Entrenar el modelo elegido
# Nuestro mejor arbol es el de prof 10 con entropia

arbol_final = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=10, random_state=SEMILLA)

#lo entrenamos con todo el conjunto de desarrollo (dev):
arbol_final.fit(datos_dev, clases_dev) 

#lo testeamos con los datos que NUNCA tocamos (held-out)
clases_decididas = arbol_final.predict(datos_held) #devuelve una lista con las clases predecidas de cada imagen

#calculamos la accuracy para el held
accuracy_arbol_final = accuracy_score(clases_held, clases_decididas)

print(f"accuracy arbol final: {round(accuracy_arbol_final, 4)}")

# La accuracy de nuestro arbol final, medida con el held out nos dio 0,792333...

#%%============================================================================================================
# 3e) Estimación de desempeño
# NOS FALTA RUNNEARLO...MAÑANA CHEQUEAMOS EN CLASE

# La accuracy esperada para predecir el conjunto de evaluacion de la catedra es 0,792333...
# Ya que a este punto, entreno con el mismo conjunto de datos, y entre el heldout y la evaluacion, no hay ningun entrenamiento adicional

'''
evaluacion_ids = evaluacion["id"]
evaluacion_datos = evaluacion.drop(columns=["id"])
clases_decididas = arbol_final.predict(evaluacion_datos)

predicciones_evaluacion_df = pd.DataFrame({"id": evaluacion_ids, "label": clases_decididas})
predicciones_evaluacion_df.to_csv(
    os.path.join(carpeta_principal, "TP-02-0.7923-NULL.csv"),
    index=False
)'''
#%%============================================================================================================