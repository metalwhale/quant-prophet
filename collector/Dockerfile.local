FROM python:3.10.12

RUN apt update -y

# Build qlib. See: https://qlib.readthedocs.io/en/latest/start/installation.html#qlib-installation
RUN apt install -y cmake
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
RUN cd /usr/src/ \
    && git clone --branch v0.9.2 https://github.com/microsoft/qlib.git \
    && cd qlib/ \
    && python setup.py install
