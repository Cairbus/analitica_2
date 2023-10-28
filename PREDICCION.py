# Cargar librerías
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sb
from scipy import stats

%matplotlib inline
plt.rcParams['figure.figsize'] = (16, 9)
plt.style use('ggplot')

from sklearn.model_selection import StratifiedKFold
# librerías Árboles de Decisión
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from sklearn import preprocessing

# librerías Gaussian Naive Bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer

dataframe.drop(['OBJECTID','RADICADO','Shape','MES_NOMBRE','DIA_NOMBRE','FECHA','HORA','DIRECCION','DIRECCION_ENC','TIPO_GEOCOD','BARRIO','X_MAGNAMED','Y_MAGNAMED','PERIODO'], axis=1, inplace=True)
dataframe = dataframe.replace('Sin inf',0)

label_encoding = preprocessing.LabelEncoder()
dataframe['clasenum'] = label_encoding.fit_transform(dataframe['CLASE'].astype(str))
dataframe['gravenum'] = label_encoding.fit_transform(dataframe['GRAVEDAD'].astype(str))
dataframe['comunum'] = label_encoding fit.transform(dataframe['COMUNA'].astype(str))
dataframe['disenum'] = label_encoding.fit_transform(dataframe['DISENO'].astype(str))
dataframe['CBML'] = label_encoding.fit_transform(dataframe['CBML'].astype(str)) #por algun motivo cbml estaba totalmente codificado como string

# Se crea un nuevo dataframe para trabajar sobre el solo con las variables numéricas
dataframe2 = dataframe.select_dtypes(include='number')
dataframe2.replace([np.inf, -np.inf], np.nan, inplace=True)
dataframe2 = dataframe2.fillna(0)
dataframe2 = dataframe2.sample(frac=1).reset_index(drop=True)

# Característica objetivo: Droga

# Mezclar datos
dataframe2 = dataframe2.sample(frac=1).reset_index(drop=True)

# Todos los datos excepto la caracteristica Droga
X = dataframe2.drop(['clasenum'], axis=1)

# CARACTERÍSTICA OBJETIVO
y = dataframe2['clasenum']

# ¿Cuáles serían las características/atributos/variables que puede ser posibles candidatos al nodo raíz?
# Se debe analizar una hipótesis con el análisis exploratorio

X = np.array(X)
y = np.array(y)

cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=1)
accuracies = list()
max_attributes = len(list(dataframe))
depth_range = range(1, max_attributes + 1)

# Testearemos la profundidad de 1 a cantidad de atributos +1
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(criterion='gini',min_samples_split=20,  min_sam   ples_leaf=5, max_depth=depth, class_weight='balanced')
    for train_index, valid_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[valid_index]
        y_train, y_test = y[train_index], y[valid_index]

        model = tree_model.fit(X_train, y_train)
        valid_acc = model.score(X_test, y_test) # calculamos la precisión con el segmento de validación
        fold_accuracy.append(valid_acc)

    # almacenamos acc promedio para cada profundidad
    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)

# Mostramos los resultados obtenidos
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]

X = dataframe2.drop(['clasenum'], axis=1)

# Atributo OBJETIVO
y = dataframe2['clasenum']

# Dividimos los datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, # 80% datos de entrenamiento, 20% prueba
                                                    stratify=y,
                                                    random_state=1)

# Datos de entrenaminto
conteo = X_train.copy()
conteo['label'] = y_train
conteo.groupby('label').size()
decision_tree = tree.DecisionTreeClassifier(criterion='gini',
                                            min_samples_split=20,
                                            min_samples_leaf=5,
                                            max_depth=depth,
                                            class_weight='balanced')

# Ajustamos el modelo con los datos
decision_tree.fit(X_train, y_train)
y_pred_train = decision_tree.predict(X_train)

# Porcentaje de Exactitud con los datos de entrenamiento
acc_train = accuracy_score(y_train, y_pred_train)
#print("Exactitud con datos de entrenamiento: {:.2f}".format(acc_train))

y_pred_test = decision_tree.predict(X_test)

# Porcentaje de Exactitud con pruebas
acc_test = accuracy_score(y_test, y_pred_test)
#print("Exactitud con datos de pruebas: {:.2f}".format(acc_test))

# Todos los datos excepto la caracteristica objetivo.
X = dataframe2.drop(['clasenum'], axis=1)
# CARACTERÍSTICA OBJETIVO

y = dataframe2['clasenum']

best = SelectKBest(k=len(list(dataframe2))-1)

X_new = best.fit_transform(X, y)
X_new.shape
selected = best.get_support(indices=True)
#print(X.columns[selected])
# Todos los datos excepto la caracteristica objetivo.
X = dataframe2.drop(['clasenum'], axis=1)

# CARACTERÍSTICA OBJETIVO
y = dataframe2['clasenum']

best = SelectKBest(k=4)

X_new = best.fit_transform(X, y)
X_new.shape
selected = best.get_support(indices=True)
#print(X.columns[selected])
# Todos los datos excepto la caracteristica clasenum (se omiten los atributos seleccionados por Kbest)
X = dataframe2.drop(['DIA','CBML','LONGITUD','comunum'], axis=1)

# Atributo OBJETIVO
y = dataframe2['clasenum']

# Dividimos los datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.8, # 80% datos de entrenamiento, 20% prueba
