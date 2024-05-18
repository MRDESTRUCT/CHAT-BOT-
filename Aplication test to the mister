import json
import nltk
import matplotlib.pyplot as plt
from textblob.classifiers import NaiveBayesClassifier
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
import os

# Download the stopwords corpus from NLTK
nltk.download('stopwords')

# Variables globales
cl = None  # Se inicializará con el clasificador después de cargar los datos de entrenamiento

# Función para cargar datos de entrenamiento desde un archivo JSON
def cargar_datos_entrenamiento(archivo_json):
    global cl

    with open(archivo_json, 'r', encoding='utf-8') as fp:
        datos_entrenamiento = json.load(fp)

    datos_entrenamiento_formateados = [(item['text'], item['label']) for item in datos_entrenamiento]
    cl = NaiveBayesClassifier(datos_entrenamiento_formateados)

# Función para analizar el sentimiento y mostrar las palabras más mencionadas
def analizar(archivo_texto):
    with open(archivo_texto, 'r', encoding='utf-8') as archivo:
        texto = archivo.read()

    oraciones = sent_tokenize(texto)
    palabras = word_tokenize(texto)

    # Filtrar las stopwords
    stop_words = set(stopwords.words('spanish'))
    palabras_filtradas = [word.lower() for word in palabras if word.isalpha() and word.lower() not in stop_words]

    # Análisis de frecuencia de palabras
    frecuencia_palabras = Counter(palabras_filtradas)
    palabras_mas_mencionadas = frecuencia_palabras.most_common()

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

    fig, ax = plt.subplots()
    ax.bar(etiquetas, porcentajes, color=['green', 'red'])
    ax.set_title('Resultados del Análisis de Datos')
    ax.set_xlabel('Aceptacion')
    ax.set_ylabel('Porcentaje')
    ax.set_ylim([0, 100])  # Establecer el rango del eje y de 0 a 100
    ax.set_yticks(range(0, 101, 10))  # Establecer los ticks del eje y en incrementos de 10
    ax.grid(True)  # Mostrar rejilla
    plt.show()

# Función para guardar las palabras más mencionadas en un archivo de texto
def save_common_words_to_file(palabras_mas_mencionadas, archivo_salida):
    with open(archivo_salida, 'w', encoding='utf-8') as file:
        for palabra, frecuencia in palabras_mas_mencionadas:
            file.write(f"{palabra}: {frecuencia}\n")

class MainApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Etiqueta de título
        title_label = Label(text='Análisis de los datos', size_hint=(1, 0.1), font_size=24, bold=True)
        layout.add_widget(title_label)

        # Botón para cargar datos de entrenamiento
        button_load_training_data = Button(text='Cargar Datos de Entrenamiento', size_hint=(1, 0.1), background_color=(0.2, 0.6, 1, 1))
        button_load_training_data.bind(on_press=self.load_training_data)
        layout.add_widget(button_load_training_data)

        # Botón para cargar archivo de texto para análisis
        button_analyze_text = Button(text='Cargar Archivo de Texto para Análisis', size_hint=(1, 0.1), background_color=(0.2, 0.6, 1, 1))
        button_analyze_text.bind(on_press=self.analyze_text)
        layout.add_widget(button_analyze_text)

        # Botón para mostrar palabras más mencionadas
        button_show_common_words = Button(text='Mostrar Palabras Más Mencionadas', size_hint=(1, 0.1), background_color=(0.2, 0.6, 1, 1))
        button_show_common_words.bind(on_press=self.show_common_words)
        layout.add_widget(button_show_common_words)

        return layout

    # Función para cargar datos de entrenamiento
    def load_training_data(self, instance):
        file_chooser = FileChooserListView()
        popup = Popup(title='Seleccionar archivo JSON para entrenamiento', content=file_chooser, size_hint=(0.9, 0.9))
        file_chooser.bind(on_submit=self.on_file_submit)
        popup.open()

    # Función llamada cuando se selecciona un archivo para cargar datos de entrenamiento
    def on_file_submit(self, instance, *args):
        archivo_json = instance.selection[0]
        cargar_datos_entrenamiento(archivo_json)
        Popup(title='Datos de entrenamiento cargados exitosamente', content=Label(text='Los datos de entrenamiento se cargaron exitosamente.'), size_hint=(None, None), size=(400, 200)).open()

    # Función para cargar archivo de texto y realizar el análisis de sentimientos
    def analyze_text(self, instance):
        file_chooser = FileChooserListView()
        popup = Popup(title='Seleccionar archivo de texto para análisis', content=file_chooser, size_hint=(0.9, 0.9))
        file_chooser.bind(on_submit=self.on_analyze_submit)
        popup.open()

    # Función llamada cuando se selecciona un archivo para análisis
    def on_analyze_submit(self, instance, *args):
        archivo_texto = instance.selection[0]
        analizar(archivo_texto)

    # Función para mostrar las palabras más mencionadas y guardarlas en un archivo
    def show_common_words(self, instance):
        file_chooser = FileChooserListView()
        popup = Popup(title='Seleccionar archivo de texto para mostrar palabras más mencionadas', content=file_chooser, size_hint=(0.9, 0.9))
        file_chooser.bind(on_submit=self.on_common_words_submit)
        popup.open()

    # Función llamada cuando se selecciona un archivo para mostrar palabras más mencionadas y guardarlas
    def on_common_words_submit(self, instance, *args):
        archivo_texto = instance.selection[0]
        with open(archivo_texto, 'r', encoding='utf-8') as archivo:
            texto = archivo.read()

        palabras = word_tokenize(texto)

        # Filtrar las stopwords
        stop_words = set(stopwords.words('spanish'))
        palabras_filtradas = [word.lower() for word in palabras if word.isalpha() and word.lower() not in stop_words]

        # Contar la frecuencia de las palabras
        frecuencia_palabras = Counter(palabras_filtradas)
        palabras_mas_mencionadas = frecuencia_palabras.most_common()

        # Guardar las palabras más mencionadas en un archivo de texto
        nombre_archivo_salida = os.path.splitext(os.path.basename(archivo_texto))[0] + '_palabras_mencionadas.txt'
        save_common_words_to_file(palabras_mas_mencionadas, nombre_archivo_salida)

        # Mostrar un mensaje de éxito
        mensaje_exito = f"Palabras más mencionadas guardadas en {nombre_archivo_salida}"
        Popup(title='Palabras Más Mencionadas Guardadas', content=Label(text=mensaje_exito), size_hint=(None, None), size=(400, 200)).open()

# Ejecutar la aplicación
if __name__ == '__main__':
    MainApp().run()
