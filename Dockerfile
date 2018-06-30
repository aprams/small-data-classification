FROM ubuntu:16.04

LABEL maintainer="Craig Citro <craigcitro@google.com>"

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
	python3-tk\
        python3 \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
	libsm6 \
	libxrender1 \
	libfontconfig1 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn

# RUN ln -s -f /usr/bin/python3 /usr/bin/python#
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

COPY . /app/

WORKDIR /app/

CMD ["python3", "server.py"]
