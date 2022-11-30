# SISTEMA CHATBOT PARA SUMINISTRAR INFORMACIÓN ACADÉMICA/ADMINISTRATIVA MEDIANTE LA PÁGINA WEB DE LA CARRERA DE INGENIERÍA DE SISTEMAS DE LA UNIVERSIDAD AUTÓNOMA TOMÁS FRÍAS

Implemenacion de sistema Chatbot

Esto da 2 opciones de implementación:
- Despliegue dentro de la aplicación Flask con la plantilla jinja2
- Sirva solo la API de predicción de Flask. Los archivos html y javascript utilizados se pueden incluir en cualquier aplicación Frontend (con solo una ligera modificación) y se pueden ejecutar completamente separados de la aplicación Flask.

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




