FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

COPY . .

EXPOSE 4000

CMD ["flask", "run", "--host=0.0.0.0", "--port=4000"]