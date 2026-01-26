FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ENV PHP_VERSION=8.3
ENV PHP_INI_DIR=/usr/local/etc/php
ENV TZ=America/Sao_Paulo
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:ondrej/php && \
    apt update && \
    apt install -y \
    php${PHP_VERSION}-cli \
    php${PHP_VERSION}-dev \
    php${PHP_VERSION}-common \
    php${PHP_VERSION}-xml \
    php${PHP_VERSION}-zip \
    php${PHP_VERSION}-curl \
    php${PHP_VERSION}-mbstring \
    build-essential \
    git \
    libtool \
    unzip && \
    rm -rf /var/lib/apt/lists/*

COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

ENV PATH="/usr/lib/ccache:${PATH}"
WORKDIR /usr/src/ext

ENV COMPOSER_ALLOW_SUPERUSER=1

CMD ["php", "-a"]