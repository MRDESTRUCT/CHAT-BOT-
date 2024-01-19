import json
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from textblob.classifiers import NaiveBayesClassifier

# Función para analizar un archivo de texto y generar un gráfico
def analyze_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        return

    with open(file_path, 'r', encoding='utf-8') as fp:
        sentences = fp.read().split('\n')

    positive_count = 0
    negative_count = 0

    for sentence in sentences:
        classification = cl.classify(sentence)
        if classification == 'positive':
            positive_count += 1
        else:
            negative_count += 1

    # Generar un gráfico de barras
    labels = ['Positive', 'Negative']
    counts = [positive_count, negative_count]

    plt.bar(labels, counts)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Analysis Results')
    plt.show()

# Función para realizar una clasificación de una oración ingresada por el usuario
def classify_sentence():
    user_input = input_entry.get()

    classification = cl.classify(user_input)
    if classification == 'positive':
        result_label.config(text=f"The sentence is classified as positive: {user_input}")
    else:
        result_label.config(text=f"The sentence is classified as negative: {user_input}")

# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("Sentiment Analysis App")

# Página de bienvenida
welcome_label = tk.Label(root, text="Welcome! Please enter your name:")
welcome_label.pack()
user_name_entry = tk.Entry(root)
user_name_entry.pack()

# Botón para identificarse
identify_button = tk.Button(root, text="Identify", command=lambda: identify_user())
identify_button.pack()

# Página principal
main_label = tk.Label(root, text="Enter a sentence for sentiment analysis:")
main_label.pack()
input_entry = tk.Entry(root)
input_entry.pack()

classify_button = tk.Button(root, text="Classify", command=lambda: classify_sentence())
classify_button.pack()
result_label = tk.Label(root, text="")
result_label.pack()

# Botón para analizar un archivo de texto
analyze_file_button = tk.Button(root, text="Analyze Text File", command=lambda: analyze_file())
analyze_file_button.pack()

# Función para identificar al usuario
def identify_user():
    user_name = user_name_entry.get()
    welcome_label.config(text=f"Welcome, {user_name}!")

# Cargar datos de entrenamiento
file_path = 'B:/PROYECTO IA CLASIFICADORA/datossp.json'
with open(file_path, 'r', encoding='utf-8') as fp:
    training_data = json.load(fp)

training_data_formatted = [(item['text'], item['label']) for item in training_data]
cl = NaiveBayesClassifier(training_data_formatted)

root.mainloop()
