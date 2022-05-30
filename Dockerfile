FROM python:3.8

RUN mkdir /app/

WORKDIR /app/

COPY . .

RUN pip install -r requirements.txt && apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install gstreamer1.0-tools -y && apt-get install -y gstreamer1.0-plugins-good -y && apt-get install -y gstreamer1.0-plugins-ugly && apt install -y gstreamer1.0-plugins-bad

CMD ["python", "main.py", "docker"]

