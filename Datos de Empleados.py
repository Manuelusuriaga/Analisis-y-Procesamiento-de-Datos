import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
# 1.Preprocesamiento de Datos:

# Cargar el dataset
url = "https://drive.google.com/uc?id=1sLvCEUOmbL341MApio5RC05nHpgGrq06"
df_resultados = pd.read_csv(url)

# Verificar si existen valores faltantes.  (No hay valores faltantes)
print(df_resultados.isna())

# Convertir la columna 'LeaveOrNot' de valores binarios (0 y 1) a las etiquetas categóricas Leave o Not Leave, dónde Leave corresponde a 1 y Not Leave a 0.

# Diccionario de mapeo de valores binarios a etiquetas categóricas
mapeo = {1: 'Leave', 0: 'Not Leave'}
# Aplicar el mapeo a la columna 'LeaveOrNot'
df_resultados['LeaveOrNot'] = df_resultados['LeaveOrNot'].map(mapeo)



#Eliminar filas que tengan valores faltantes en las colúmnas ExperienceInCurrentDomain y JoiningYear.
df_resultados = df_resultados.dropna(subset=['ExperienceInCurrentDomain', 'JoiningYear'])

#Imputar datos faltantes de la colúmna Age con la media.
# A.Calcular la media de la columna 'Age'
media_edad = df_resultados['Age'].mean()
# B. Imputar
df_resultados['Age'].fillna(media_edad, inplace=True)

# Imputar datos faltantes de la colúmna PaymentTier con la moda.
# A.Calcular la moda de la columna 'PaymentTier'
moda_payment_tier = df_resultados['PaymentTier'].mode()[0]
# B.Imputar
df_resultados['PaymentTier'].fillna(moda_payment_tier, inplace=True)

# Obtener solo las columnas numéricas (excluyendo 'LeaveOrNot')
columnas_numericas = df_resultados.select_dtypes(include=['number']).columns

# Calcular el IQR para cada columna numérica
Q1 = df_resultados[columnas_numericas].quantile(0.25)
Q3 = df_resultados[columnas_numericas].quantile(0.75)
IQR = Q3 - Q1

# Determinar los límites inferior y superior para cada columna
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtrar el DataFrame para excluir los registros con valores atípicos
df_filtrado = df_resultados.copy()
for columna in columnas_numericas:
    df_filtrado = df_filtrado[(df_filtrado[columna] >= limite_inferior[columna]) & (df_filtrado[columna] <= limite_superior[columna])]
print(df_filtrado)

#--------------------------------------------------------------------------------------------------------

# 2. Análisis Exploratorio de Datos (EDA):

#Graficar la distribución de los sexos con un gráfico de torta

# Contar el número de ocurrencias de cada categoría en la columna 'Gender'
conteo_sexos = df_resultados['Gender'].value_counts()

# Graficar el gráfico de torta
plt.pie(conteo_sexos, labels=conteo_sexos.index, autopct='%1.1f%%', startangle=140) # (autopct='%1.1f%%'), para mostrar el porcentaje.
plt.title('Distribución de Sexos') # Titulo
plt.legend(title = "Distribución de Sexos", loc ="center left", bbox_to_anchor = (0.9,0.5)) #Leyenda
plt.axis('equal') # Para dar forma circular
plt.show() # Imprimir

#Graficar la Distribución de niveles de estudio de los empleados usando subplots para tener un histograma a la izquierda y a su derecha una gráfica de torta.

# Crear subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# Histograma
axs[0].hist(df_resultados['Education'], bins=range(len(df_resultados['Education'].unique()) + 1), edgecolor='black')
axs[0].set_title('Distribución de Niveles de Estudio') #título para el  primer subplot
axs[0].set_xlabel('Education')
axs[0].set_ylabel('Frecuencia')
Education_counts = df_resultados['Education'].value_counts()
axs[1].pie(Education_counts, labels=Education_counts.index, autopct='%1.1f%%', startangle=140)
axs[1].set_title('Distribución de Niveles de Estudio') #título para el segundo subplot
plt.tight_layout() # Ajusta automáticamente los subplots
plt.show()

# Filtrar los datos de los empleados que han tomado licencias
empleados_con_licencias = df_resultados[df_resultados['LeaveOrNot'] == 'Leave']
# Crear un histograma de la edad de los empleados con licencias
plt.hist(empleados_con_licencias['Age'], bins=10, edgecolor='black')
plt.title('Distribución de Edad de Empleados con Licencias')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

# Calcular la frecuencia de la columna 'LeaveOrNot' (nuestra clase)
frecuencia_clases = df_resultados['LeaveOrNot'].value_counts()

# Graficar la distribución de clases
plt.bar(frecuencia_clases.index, frecuencia_clases.values)
plt.title('Distribución de la Clase "LeaveOrNot"')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)  # Rotar etiquetas del eje x para mayor legibilidad
plt.show()

# Verificar si el dataset está balanceado
balanceado = frecuencia_clases.min() / frecuencia_clases.max() > 0.8  # Si la proporción mínima respecto a la máxima es mayor al 80%
print('¿Está el dataset balanceado?:', balanceado)

#--------------------------------------------------------------------------------------------------------
# 3.Modelado de Datos:
# Preparar los datos para el modelado:

# Guardar la columna objetivo en una variable separada
objetivo = df_resultados['LeaveOrNot']

# Elimina la colúmna objetivo del dataframe y sácala a otra variable para entrenar el modelo.
df = df_resultados.drop(columns=['LeaveOrNot'])

# Convierte a variables dummies todas las variables categóricas.
  # Obtener las columnas categóricas
columnas_categoricas = df.select_dtypes(include=['object']).columns
df_dummies = pd.get_dummies(df, columns=columnas_categoricas) # Conversion de variables

# Realiza una partición estratificada del dataset con train_test_split usando el argumento stratify.
X_train, X_test, y_train, y_test = train_test_split(df_dummies, objetivo, test_size=0.2, stratify=objetivo, random_state=42)

#Entrenar dos RandomForests, uno sin cambios y otro usando el argumento class_weight="balanced".

# Modelo Random Forest sin cambios
rf_sin_balance = RandomForestClassifier(random_state=42)
rf_sin_balance.fit(X_train, y_train)
predicciones_sin_balance = rf_sin_balance.predict(X_test)
accuracy_sin_balance = accuracy_score(y_test, predicciones_sin_balance)
print("Precisión del modelo Random Forest sin cambios:", accuracy_sin_balance)

# Modelo Random Forest con class_weight="balanced"
rf_con_balance = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_con_balance.fit(X_train, y_train)
predicciones_con_balance = rf_con_balance.predict(X_test)
accuracy_con_balance = accuracy_score(y_test, predicciones_con_balance)
print("Precisión del modelo Random Forest con class_weight=\"balanced\":", accuracy_con_balance)

#Calcular métricas de desempeño para ambos modelos:
#Accuracy en el conjunto de entrenamiento

# Calcular la precisión en el conjunto de entrenamiento para el modelo sin cambios
predicciones_train_sin_balance = rf_sin_balance.predict(X_train)
accuracy_train_sin_balance = accuracy_score(y_train, predicciones_train_sin_balance)
print("Precisión en el conjunto de entrenamiento para el modelo sin cambios:", accuracy_train_sin_balance)

# Calcular la precisión en el conjunto de entrenamiento para el modelo con class_weight="balanced"
predicciones_train_con_balance = rf_con_balance.predict(X_train)
accuracy_train_con_balance = accuracy_score(y_train, predicciones_train_con_balance)

#Accuracy en el conjunto de test

# Calcular la precisión en el conjunto de prueba para el modelo sin cambios
accuracy_test_sin_balance = accuracy_score(y_test, predicciones_sin_balance)
print("Precisión en el conjunto de prueba para el modelo sin cambios:", accuracy_test_sin_balance)

# Calcular la precisión en el conjunto de prueba para el modelo con class_weight="balanced"
accuracy_test_con_balance = accuracy_score(y_test, predicciones_con_balance)
print("Precisión en el conjunto de prueba para el modelo con class_weight=\"balanced\":", accuracy_test_con_balance)

# Matriz de confusión (Grafícala ayudándote con ConfusionMatrixDisplay de Scikit-Learn)

# Calcular la matriz de confusión para el modelo sin cambios
cm_sin_balance = confusion_matrix(y_test, predicciones_sin_balance)

# Graficar la matriz de confusión para el modelo sin cambios
disp_sin_balance = ConfusionMatrixDisplay(confusion_matrix=cm_sin_balance, display_labels=['Not Leave', 'Leave'])
disp_sin_balance.plot(cmap='Blues')
plt.title("Matriz de Confusión - Modelo sin cambios")
plt.show()

# Calcular la matriz de confusión para el modelo con class_weight="balanced"
cm_con_balance = confusion_matrix(y_test, predicciones_con_balance)

# Graficar la matriz de confusión para el modelo con class_weight="balanced"
disp_con_balance = ConfusionMatrixDisplay(confusion_matrix=cm_con_balance, display_labels=['Not Leave', 'Leave'])
disp_con_balance.plot(cmap='Blues')
plt.title("Matriz de Confusión - Modelo con class_weight=\"balanced\"")
plt.show()

# Calcular el F1 Score para el modelo sin cambios
f1_sin_balance = f1_score(y_test, predicciones_sin_balance, pos_label='Leave')
print("F1 Score para el modelo sin cambios:", f1_sin_balance)

# Calcular el F1 Score para el modelo con class_weight="balanced"
f1_con_balance = f1_score(y_test, predicciones_con_balance, pos_label='Leave')
print("F1 Score para el modelo con class_weight=\"balanced\":", f1_con_balance)

