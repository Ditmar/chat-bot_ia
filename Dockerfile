FROM nvcr.io/nvidia/pytorch:21.06-py3
RUN pip install --upgrade pip && pip install \
    timm \
    nltk \
    numpy \
    pymongo \
    flask
WORKDIR /workspace
COPY . /workspace
CMD ["python3", "app.py"]