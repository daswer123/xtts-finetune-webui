FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Copy the requirements.txt file to the container
COPY requirements.txt /root/

# Install the packages from requirements.txt
RUN pip install --no-cache-dir -r /root/requirements.txt

ENV GRADIO_SERVER_NAME=0.0.0.0

WORKDIR /root/tts

CMD ["/usr/bin/python3", "xtts_demo.py"]
