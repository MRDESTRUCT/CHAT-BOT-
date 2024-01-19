import json
from textblob.classifiers import NaiveBayesClassifier


file_path = 'B:/PROYECTO IA CLASIFICADORA/datossp.json'

with open(file_path, 'r', encoding='utf-8') as fp:
    training_data = json.load(fp)


training_data_formatted = [(item['text'], item['label']) for item in training_data]


cl = NaiveBayesClassifier(training_data_formatted)

while True:
    user_input = input("Me encanto la comida")

    if user_input.lower() == 'exit':
        break

    classification = cl.classify(user_input)
    if classification == 'positive':
        print(f"The sentence is classified as positive: {user_input}")
    else:
        print(f"The sentence is classified as negative: {user_input}")
