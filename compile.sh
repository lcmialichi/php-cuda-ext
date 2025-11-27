#!/bin/bash
set -e

detect_cuda_home() {
    if [ -d "/usr/local/cuda" ]; then
       echo "/usr/local/cuda"
       return 0
    fi

    if command -v nvcc >/dev/null 2>&1; then
        NVCC_PATH=$(command -v nvcc)
        CUDA_HOME=$(dirname "$(dirname "$NVCC_PATH")")
        if [ -d "$CUDA_HOME" ]; then
            echo "$CUDA_HOME"
            return 0
        fi
    fi

    if [ -d "/usr/lib/cuda" ]; then
        echo "/usr/lib/cuda"
       return 0

    fi

    if [ -d "/usr/lib64/nvidia/toolkit" ]; then
        echo "/usr/lib64/nvidia/toolkit"
        return 0

    fi

    if [ -d "/opt/cuda" ]; then
        echo "/opt/cuda"
        return 0

    fi

    if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
        echo "$CUDA_HOME"
        return 0

    fi

    return 1

}


echo ""
echo "┌───────────────────────────────────────────────────────┐"
echo "│     PHP CUDA Extension — Build & Install Script       │"
echo "└───────────────────────────────────────────────────────┘"
echo ""

EXT_NAME="cuda"
SRC_DIR="$(pwd)"
BUILD_DIR="./${EXT_NAME}_build"
INSTALL_EXT_DIR=$(php-config --extension-dir)
CUDA_HOME=$(detect_cuda_home)

if [ "$CUDA_HOME" = "1" ] || [ -z "$CUDA_HOME" ]; then
    echo "Unable to detect CUDA home. Make sure NVCC is installed."
    exit 1
fi

if [ "$EUID" -ne 0 ]; then
    echo "WARNING: Running without root privileges."
    INSTALL_DEPS=0
else
    INSTALL_DEPS=1
fi

if [ ! -d "/usr/local/cuda" ]; then
    echo "ERROR: CUDA Toolkit not found at /usr/local/cuda"
    exit 1
fi

echo "✔ CUDA Toolkit found."

echo ""
echo "Preparing build directory..."
[ -d "$BUILD_DIR" ] && rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cp config.m4 Makefile Makefile.frag "$BUILD_DIR"
cp -R src/ tests/ "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "Building PHP extension: $EXT_NAME"

[ -f "Makefile" ] && make clean || true

phpize
./configure --with-cuda=$CUDA_HOME

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

cat > "$INI_FILE_TEMP" <<EOF
; PHP CUDA Extension
extension=$EXT_NAME.so
EOF

TARGET_DIR=$(php --ini | grep "Scan for additional .ini files" | awk -F": " '{print $2}')

if [ ! -d "$TARGET_DIR" ] || [ "$TARGET_DIR" = "(none)" ]; then
    echo "Falling back to manual detection..."

    PHP_VERSION=$(php -r "echo PHP_MAJOR_VERSION.'.'.PHP_MINOR_VERSION;")

    if [ -d "/etc/php/$PHP_VERSION/mods-available" ]; then
        TARGET_DIR="/etc/php/$PHP_VERSION/mods-available"
    elif [ -d "/etc/php.d" ]; then
        TARGET_DIR="/etc/php.d"
    else
        echo "ERROR: Could not detect INI directory."
        echo "Copy manually: $INI_FILE_TEMP"
        exit 0
    fi
fi

echo "→ INI target directory: $TARGET_DIR"

if cp "$INI_FILE_TEMP" "$TARGET_DIR/"; then
    echo "✔ INI file installed."
else
    echo "WARNING: Permission denied. Install manually:"
    echo "sudo cp $INI_FILE_TEMP $TARGET_DIR/"
fi

if [[ "$TARGET_DIR" =~ mods-available ]]; then
    echo ""
    echo "Enabling INI via phpenmod..."
    phpenmod "$EXT_NAME" || echo "WARNING: phpenmod failed."
fi

echo ""
echo "✔ Build and installation complete!"
echo "Restart PHP-FPM/Apache to apply changes."
echo ""
