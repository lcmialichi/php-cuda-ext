### PHP CUDA Extension

A native PHP extension that provides direct access to NVIDIA CUDA functionality, enabling high-performance GPU computing inside PHP applications.

### Requirements
To build and run this extension, you need:

- NVIDIA CUDA Toolkit (12.x recommended)
- Compatible NVIDIA GPU driver
- PHP 8.0+ with support for C extensions
- gcc / g++
- make / autoconf
- Linux (Ubuntu, Debian, CentOS, Arch, etc.)


### How to Compile
```bash
git clone https://github.com/lcmialichi/php-cuda-ext.git/
cd php-cuda-ext
```
Compile and install the extension by running:
```bash
./compile
```

The compile script automatically performs:
1. phpize
2. ./configure
3. make
4. make install

Verify that it loaded correctly:
```bash
php -m | grep cuda
```