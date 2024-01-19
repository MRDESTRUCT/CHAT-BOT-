import json
import nltk
import matplotlib.pyplot as plt
from textblob.classifiers import NaiveBayesClassifier
from nltk.tokenize import sent_tokenize
import tkinter as tk
from tkinter import filedialog

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

    conteo_positivo = 0
    conteo_negativo = 0
    resultados_clasificacion = []

    for oracion in oraciones:
        clasificacion = cl.classify(oracion)
        resultados_clasificacion.append(clasificacion)
        if clasificacion == 'pos':
            conteo_positivo += 1
        elif clasificacion == 'neg':
            conteo_negativo += 1

    etiquetas = ['Positivo', 'Negativo']
    conteos = [conteo_positivo, conteo_negativo]

    plt.bar(etiquetas, conteos, color=['green', 'red'])
    plt.title('Resultados del Análisis de Sentimientos')
    plt.xlabel('Sentimiento')
    plt.ylabel('Conteo')
    plt.show()

# Crear un botón para analizar el sentimiento
boton_analizar = tk.Button(raiz, text='Analizar Sentimiento', command=analizar_sentimiento, bg='black', fg='white', font=('Arial', 14))
boton_analizar.pack()

# Ejecutar el bucle principal de Tkinter
raiz.mainloop()
