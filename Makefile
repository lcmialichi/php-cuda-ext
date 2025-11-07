.PHONY: all build install clean test distclean clean-cuda dev uninstall status help

BUILD_DIR   := build
PHP_VERSION := $(shell php -r "echo PHP_MAJOR_VERSION . PHP_MINOR_VERSION;" 2>/dev/null)
EXT_NAME    := cuda
NVCC        := $(shell which nvcc 2>/dev/null)

all: build

build:
	@echo "Building PHP CUDA extension..."
	@mkdir -p $(BUILD_DIR)
	# Copia arquivos de configuração E todo o diretório src
	cp config.m4 config.w32 $(BUILD_DIR)/ 2>/dev/null || true
	cp -r src/* $(BUILD_DIR)/ 2>/dev/null || true
	cd $(BUILD_DIR) && \
	phpize && \
	./configure --with-cuda && \
	make

install: build
	@echo "Installing extension..."
	cd $(BUILD_DIR) && sudo make install
	@echo "Extension installed. Add 'extension=$(EXT_NAME).so' to your php.ini"

uninstall:
	@echo "🗑️  Uninstalling extension..."
	cd $(BUILD_DIR) && sudo make uninstall || true
	@echo "Extension uninstalled"

dev: clean
	@echo "Building for development..."
	@mkdir -p $(BUILD_DIR)
	cp config.m4 config.w32 $(BUILD_DIR)/ 2>/dev/null || true
	cp -r src/* $(BUILD_DIR)/ 2>/dev/null || true
	cd $(BUILD_DIR) && \
	phpize && \
	./configure --with-cuda --enable-debug CFLAGS="-g -O0" && \
	make

test: build
	@echo "Running tests..."
	cd $(BUILD_DIR) && make test

clean-cuda:
	@echo "Cleaning CUDA objects..."
	rm -f $(BUILD_DIR)/*.cu.o

clean: clean-cuda
	@echo "Cleaning build directory..."
	rm -rf $(BUILD_DIR)

distclean: clean
	@echo "Deep cleaning..."
	rm -f package.xml *.tgz

status:
	@echo "Checking extension status..."
	@php -m | grep $(EXT_NAME) || echo "❌ Extension not loaded"
	@php -i | grep "CUDA" || echo "ℹ️  CUDA info not available"

help:
	@echo "📦 PHP CUDA Extension Build System"
	@echo ""
	@echo "Commands:"
	@echo "  make build     - Build extension in ./build/"
	@echo "  make install   - Build and install system-wide"
	@echo "  make uninstall - Uninstall extension"
	@echo "  make dev       - Build for development (debug symbols)"
	@echo "  make test      - Build and run tests"
	@echo "  make clean     - Remove build directory and CUDA objects"
	@echo "  make clean-cuda - Remove only CUDA objects"
	@echo "  make distclean - Deep clean (including distribution files)"
	@echo "  make status    - Check extension status"
	@echo "  make help      - Show this help"

.DEFAULT_GOAL := help