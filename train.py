import numpy as np
from pymongo import MongoClient
import json

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#Importamos las funciones de nltk_utils
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet


client = MongoClient("mongodb+srv://Angie:62426426848648@cluster0.abxoiod.mongodb.net/?retryWrites=true&w=majority")
db = client.get_database('Chatbot')
intents = db.intents


# Todas las palabras
all_words = []
# Etiquetas
tags = []
# Patrones
xy = []
# Recorrera cada oracion en nuestros patrones de intents
for intent in intents.find():
    tag = intent['tag']
    # Añadir a la matriz tags
    tags.append(tag)
    for pattern in intent['patterns']:
        # ignorar caracteres especiales
        ignore_words = '¿?,.¡!'
        res = pattern.translate(str.maketrans('', '', ignore_words))
        # Tokenizar cada palabra en la oracion
        w = tokenize(res)
        # Añadir en en nuestra lista de palabras all_words
        all_words.extend(w)
        # Se añade cada oracion tokenizada a su tag
        xy.append((w, tag))

# stem y convertir
all_words = [stem(w) for w in all_words]
# Eliminar palabras duplicadas
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)


# Creacion de los datos de entrenamiento
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)


X_train = np.array(X_train)
y_train = np.array(y_train)


# Hiper-parametros
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Perdida y optimizacion
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # hacia adelante
        outputs = model(words)
        loss = criterion(outputs, labels)
        # hacia atrás y optimizacion
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

# Guardado del modelo de datos a un Diccionario
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}
FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')

