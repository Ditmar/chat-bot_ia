from pymongo import MongoClient
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

client = MongoClient("mongodb+srv://Angie:62426426848648@cluster0.abxoiod.mongodb.net/?retryWrites=true&w=majority")
db = client.get_database('Chatbot')
intents = db.intents
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Chatbot"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.85:
        for intent in intents.find():
            if tag == intent["tag"]:
                return (intent['responses'])
    records = db.messages
    payload = {'msg':msg}
    records.insert_one(payload)
    return ("Aun no tengo respuesta para esa peticion")


if __name__ == "__main__":
    print("Hola! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)