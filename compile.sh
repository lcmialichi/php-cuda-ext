#!/bin/bash
set -e

EXT_NAME="cuda"
SRC_DIR="$(pwd)"
BUILD_DIR="./${EXT_NAME}_build"
INSTALL_EXT_DIR=$(php-config --extension-dir)

echo ""
echo "┌───────────────────────────────────────────────────────┐"
echo "│     PHP CUDA Extension — Build & Install Script       │"
echo "└───────────────────────────────────────────────────────┘"
echo ""

if [ "$EUID" -ne 0 ]; then
    echo "WARNING  Running without root privileges."
    echo "   Dependency installation will be skipped."
    echo "   sudo ./compile.sh"
    INSTALL_DEPS=0
else
    INSTALL_DEPS=1
fi

if [ ! -d "/usr/local/cuda" ]; then
    echo "ERROR: CUDA Toolkit not found at /usr/local/cuda"
    echo "   Install CUDA before continuing:"
    echo "   https://developer.nvidia.com/cuda-downloads"
    exit 1
fi


echo "✔ CUDA Toolkit found."

echo ""
echo "Preparing build directory..."
if [ ! -d "$BUILD_DIR" ]; then
   rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

cp config.m4 Makefile Makefile.frag "$BUILD_DIR"
cp -R src/ tests/ "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "Building PHP extension: $EXT_NAME"

if [ -f "Makefile" ]; then
    make clean || true
fi

phpize
./configure --with-cuda=/usr/local/cuda 

echo ""
echo "Compiling..."
make -j"$(nproc)"

echo ""
echo "Installing into PHP extension directory:"
echo "→ $(php-config --extension-dir)"
make install

echo ""
echo "Generating INI file..."

INI_FILE_NAME="$EXT_NAME.ini"
INI_FILE_TEMP="/tmp/$INI_FILE_NAME"

{
    echo "; PHP CUDA Extension Configuration"
    echo "extension=$EXT_NAME.so"
} > "$INI_FILE_TEMP"

PHP_INI_SCAN_DIR=$(php -i | grep 'Scan this dir for additional .ini files' | awk '{print $NF}')

if [ -d "$PHP_INI_SCAN_DIR" ]; then
    echo "→ Found ini scan directory: $PHP_INI_SCAN_DIR"

    if cp "$INI_FILE_TEMP" "$PHP_INI_SCAN_DIR/"; then
        echo "✔ INI file installed."
    else
        echo "WARNING:  Permission denied. Install manually:"
        echo "sudo cp $INI_FILE_TEMP $PHP_INI_SCAN_DIR/"
    fi

else
    echo "WARNING:  PHP scan directory could not be detected."
    echo "Create this file manually in the appropriate PHP config directory:"
    echo ""
    echo "$INI_FILE_NAME:"
    echo "extension=$EXT_NAME.so"
fi

echo ""
echo "✔ Build and installation complete!"
echo "   Restart Apache/FPM or your PHP environment to apply changes."
echo ""
