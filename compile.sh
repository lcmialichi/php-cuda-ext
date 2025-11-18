#!/bin/bash
set -e

EXT_NAME="cuda"
SRC_DIR="$(pwd)"
BUILD_DIR="/tmp/${EXT_NAME}_build" 
INSTALL_EXT_DIR=$(php-config --extension-dir)

find /usr -name "cudnn.h" 2>/dev/null
find /usr/local -name "cudnn.h" 2>/dev/null
find /opt -name "cudnn.h" 2>/dev/null

apt-get update && apt-get install -y libcudnn8-dev

cd $SRC_DIR 
echo "--- initializing extension build $EXT_NAME (C++/CUDA) ---"

if [ -f "Makefile" ]; then
    make clean 2>/dev/null
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cp -R "$SRC_DIR"/* ./

phpize
./configure --with-cuda=/usr/local/cuda

echo "Compiling $EXT_NAME..."
make

echo "installing $EXT_NAME.so at PHP (DIR: $(php-config --extension-dir))..."
make install

INSTALL_DIR=$(php-config --extension-dir)
TEST_INI="/tmp/$EXT_NAME-test.ini"

echo "extension=$EXT_NAME.so" > "$TEST_INI"
    INI_FILE_NAME="$EXT_NAME.ini"
    INI_FILE_TEMP="/tmp/$INI_FILE_NAME"
    echo "; Extension configuration $EXT_NAME" > "$INI_FILE_TEMP"
    echo "extension=$EXT_NAME.so" >> "$INI_FILE_TEMP"
    
    PHP_INI_SCAN_DIR=$(php -i | grep 'Scan this dir for additional .ini files' | awk '{print $NF}')
    
if [ -d "$PHP_INI_SCAN_DIR" ]; then
    if command -v phpenmod &> /dev/null; then
        echo "   -> Enabling extension..."
        
        PHP_VERSION=$(php -r "echo PHP_MAJOR_VERSION . '.' . PHP_MINOR_VERSION;")
        MODS_AVAILABLE_DIR="/etc/php/$PHP_VERSION/mods-available"
        
        if [ -d "$MODS_AVAILABLE_DIR" ]; then
            if cp "$INI_FILE_TEMP" "$MODS_AVAILABLE_DIR/"; then
                phpenmod "$EXT_NAME"
                echo "--- BUILD AND INSTALLATION SUCCESSFULLY COMPLETED! ---"
                echo "Remember to RESTART your web/FPM server to apply the changes."
            else
                echo "   \033[33mWARNING: Permission failure. Unable to copy the INI file to $MODS_AVAILABLE_DIR.\033[0m"
                echo "   \033[33m The extension has been compiled, but you must enable it manually with 'sudo':\033[0m"
                echo "   \$ \033[33msudo cp $INI_FILE_TEMP $MODS_AVAILABLE_DIR/\033[0m"
                echo "   \$ \033[33msudo phpenmod $EXT_NAME\033[0m"
            fi
            
        else
            if cp "$INI_FILE_TEMP" "$PHP_INI_SCAN_DIR/"; then
                echo "--- BUILD AND INSTALLATION SUCCESSFULLY COMPLETED! ---"
                echo "Extension $EXT_NAME configured in $PHP_INI_SCAN_DIR."
                echo "Remember to RESTART your web/FPM server to apply the changes."
            else
                echo "   \033[33mWARNING: Permission failure. Unable to copy the INI file to $PHP_INI_SCAN_DIR.\033[0m"
                echo "   \033[33mThe extension has been compiled, but you must enable it manually with 'sudo':\033[0m"
                echo "   (temp INI path: $INI_FILE_TEMP)"
                echo "   (src path: $PHP_INI_SCAN_DIR)"
                echo "   \$ \033[33msudo cp $INI_FILE_TEMP $PHP_INI_SCAN_DIR/\033[0m"
            fi
        fi

    else
        if cp "$INI_FILE_TEMP" "$PHP_INI_SCAN_DIR/"; then
            echo "--- BUILD AND INSTALLATION SUCCESSFULLY COMPLETED! ---"
            echo "Extension $EXT_NAME configured in $PHP_INI_SCAN_DIR."
            echo "Remember to RESTART your web/FPM server to apply the changes."
        else
            echo "   \033[33mAVISO: Falha de Permissão. Não foi possível copiar o INI para $PHP_INI_SCAN_DIR.\033[0m"
            echo "   \033[33mA extensão foi compilada, mas você deve habilitá-la manualmente com 'sudo':\033[0m"
            echo "   (temp INI path: $INI_FILE_TEMP)"
            echo "   (src path: $PHP_INI_SCAN_DIR)"
            echo "   \$ \033[33msudo cp $INI_FILE_TEMP $PHP_INI_SCAN_DIR/\033[0m"
        fi
    fi

else
    echo "--- BUILD AND INSTALLATION SUCCESSFULLY COMPLETED! ---"
    echo "The PHP scan directory could not be determined."
    echo "Create the file $EXT_NAME.ini in your PHP configuration directory with the following content:"
    echo "extension=$EXT_NAME.so"
fi