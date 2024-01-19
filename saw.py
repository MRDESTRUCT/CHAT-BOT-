import json
import nltk
import matplotlib.pyplot as plt
from textblob.classifiers import NaiveBayesClassifier
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import tkinter as tk
from tkinter import filedialog

# Download the stopwords corpus from NLTK
nltk.download('stopwords')

# Cargar el clasificador de sentimientos preentrenado
archivo_datos = 'B:/PROYECTO IA CLASIFICADORA/datossp.json'
with open(archivo_datos, 'r', encoding='utf-8') as fp:
    datos_entrenamiento = json.load(fp)
datos_entrenamiento_formateados = [(item['text'], item['label']) for item in datos_entrenamiento]
cl = NaiveBayesClassifier(datos_entrenamiento_formateados)

# Crear una ventana de Tkinter
raiz = tk.Tk()
raiz.title('Aplicación de Análisis de Sentimientos')
raiz.configure(bg='black')  # Establecer el color de fondo en negro

# Función para analizar el sentimiento cuando se presiona el botón
def analizar_sentimiento():
    archivo_texto = filedialog.askopenfilename(filetypes=[("Archivos de texto", "*.txt")])
    with open(archivo_texto, 'r', encoding='utf-8') as archivo:
        texto = archivo.read()

    oraciones = sent_tokenize(texto)
    palabras = word_tokenize(texto)

    # Filtrar las stopwords
    stop_words = set(stopwords.words('spanish'))
    palabras_filtradas = [word.lower() for word in palabras if word.isalpha() and word.lower() not in stop_words]

    # Análisis de frecuencia de palabras
    frecuencia_palabras = Counter(palabras_filtradas)
    palabra_mas_usada = frecuencia_palabras.most_common(1)[0][0]

    conteo_positivo = 0
    conteo_negativo = 0

    for oracion in oraciones:
        clasificacion = cl.classify(oracion)
        if clasificacion == 'pos':
            conteo_positivo += 1
        elif clasificacion == 'neg':
            conteo_negativo += 1

    total_oraciones = len(oraciones)

    porcentaje_positivo = (conteo_positivo / total_oraciones) * 100
    porcentaje_negativo = (conteo_negativo / total_oraciones) * 100

    etiquetas = ['Positivo', 'Negativo']
    porcentajes = [porcentaje_positivo, porcentaje_negativo]

    plt.bar(etiquetas, porcentajes, color=['green', 'red'])
    plt.title('Resultados del Análisis de Sentimientos')
    plt.xlabel('Sentimiento')
    plt.ylabel('Porcentaje')
    plt.ylim([0, 100])  # Establecer el rango del eje y de 0 a 100
    plt.show()

    print(f"Palabra más usada: {palabra_mas_usada}")

# Crear un botón para analizar el sentimiento
boton_analizar = tk.Button(raiz, text='Analizar Sentimiento', command=analizar_sentimiento, bg='black', fg='white', font=('Arial', 14))
boton_analizar.pack()

# Ejecutar el bucle principal de Tkinter
raiz.mainloop()
