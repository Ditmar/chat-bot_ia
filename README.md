# SISTEMA CHATBOT PARA SUMINISTRAR INFORMACIÓN ACADÉMICA/ADMINISTRATIVA MEDIANTE LA PÁGINA WEB DE LA CARRERA DE INGENIERÍA DE SISTEMAS DE LA UNIVERSIDAD AUTÓNOMA TOMÁS FRÍAS

Implemenacion de sistema Chatbot

## Configuración inicial:

Clonar repositorio y crear un entorno virtual

```
$ git clone https://github.com/Ditmar/chat-bot_ia.git
$ cd chatbot-deployment
$ python3 -m venv venv
$ . venv/bin/activate
```
Instalar dependencias
```
$ (venv) pip install Flask torch torchvision nltk
```
Instalar el paquete NLTK
```
$ (venv) python
>>> import nltk
>>> nltk.download('punkt')
```
Ingresar `intents.json` a base de datos MongoDB

Run
```
$ (venv) python train.py
```
Esto volcará el archivo data.pth. y luego corre
el siguiente comando para probarlo en la consola.
```
$ (venv) python chat.py
```




