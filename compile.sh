#!/bin/bash
set -e

cd "$(dirname "$0")"
PHP_BIN=${PHP_BIN:-php}
PHPIZE=${PHPIZE:-phpize}
PHP_CONFIG=${PHP_CONFIG:-php-config}
INSTALL=0
if [ "${1:-}" = "--install" ]; then
    INSTALL=1
elif [ "$#" -ne 0 ]; then
    echo "Usage: $0 [--install]" >&2
    exit 2
fi

PHP_VERSION=$($PHP_CONFIG --version | cut -d. -f1,2)
RUNTIME_VERSION=$($PHP_BIN -r 'echo PHP_MAJOR_VERSION.".".PHP_MINOR_VERSION;')
if [ "$PHP_VERSION" != "$RUNTIME_VERSION" ]; then
    echo "PHP binary ($RUNTIME_VERSION) and php-config ($PHP_VERSION) do not match" >&2
    exit 1
fi

detect_cuda_home() {
    if [ -n "${CUDA_HOME:-}" ] && [ -f "$CUDA_HOME/include/cuda_runtime.h" ]; then
        echo "$CUDA_HOME"
        return 0
    fi

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
BUILD_DIR=${BUILD_DIR:-"./${EXT_NAME}_build-${PHP_VERSION}"}
CUDA_HOME=$(detect_cuda_home || true)

if [ "$CUDA_HOME" = "1" ] || [ -z "$CUDA_HOME" ]; then
    echo "Unable to detect CUDA home. Make sure NVCC is installed."
    exit 1
fi

if [ ! -f "$CUDA_HOME/include/cuda_runtime.h" ]; then
    echo "ERROR: CUDA Toolkit headers not found at $CUDA_HOME"
    exit 1
fi

echo "✔ CUDA Toolkit found."

echo ""
echo "Preparing build directory..."
mkdir -p "$BUILD_DIR"

cp config.m4 Makefile.frag "$BUILD_DIR"
cp -R src/ tests/ "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "Building PHP extension: $EXT_NAME"

"$PHPIZE"
./configure --with-php-config="$PHP_CONFIG" --with-cuda="$CUDA_HOME"

echo ""
echo "Compiling..."
make -j"$(nproc)"

if [ "$INSTALL" -eq 0 ]; then
    echo "Build complete: $BUILD_DIR/modules/cuda.so"
    echo "Run ./run-tests.sh --require-gpu to validate with NVIDIA driver access."
    exit 0
fi

echo ""
echo "Installing into PHP extension directory:"
echo "→ $($PHP_CONFIG --extension-dir)"
make install

echo ""
echo "Generating INI file..."

INI_FILE_NAME="$EXT_NAME.ini"
INI_FILE_TEMP="/tmp/$INI_FILE_NAME"

cat > "$INI_FILE_TEMP" <<EOF
; PHP CUDA Extension
extension=$EXT_NAME.so
EOF

TARGET_DIR=$($PHP_BIN --ini | grep "Scan for additional .ini files" | awk -F": " '{print $2}')

if [ ! -d "$TARGET_DIR" ] || [ "$TARGET_DIR" = "(none)" ]; then
    echo "Falling back to manual detection..."

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
