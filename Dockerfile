FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ENV PHP_VERSION=8.3
ENV PHP_INI_DIR=/usr/local/etc/php

ENV TZ=America/Sao_Paulo
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:ondrej/php && \
    apt update

RUN apt install -y \
    php${PHP_VERSION}-cli \
    php${PHP_VERSION}-dev \
    build-essential \
    git \
    libtool && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/rusr/lib/ccache:${PATH}"
WORKDIR /usr/src/ext

CMD ["php", "-a"]