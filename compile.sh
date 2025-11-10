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
echo "--- Iniciando o Build da Extensão $EXT_NAME (C++/CUDA) ---"

echo "1. Limpando resíduos de builds anteriores..."
if [ -f "Makefile" ]; then
    make clean 2>/dev/null
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cp -R "$SRC_DIR"/* ./

echo "2. Executando phpize..."
phpize

echo "3. Executando ./configure (linkando CUDA)..."
./configure --with-cuda=/usr/local/cuda

echo "4. Compilando a extensão $EXT_NAME..."
make

echo "5. Instalando $EXT_NAME.so no PHP (Diretório: $(php-config --extension-dir))..."
make install

echo "6. Teste de Carregamento Forçado:"
INSTALL_DIR=$(php-config --extension-dir)
TEST_INI="/tmp/$EXT_NAME-test.ini"

echo "extension=$EXT_NAME.so" > "$TEST_INI"

    
    INI_FILE_NAME="$EXT_NAME.ini"
    INI_FILE_TEMP="/tmp/$INI_FILE_NAME"
    echo "; Configuração de carregamento para a extensão $EXT_NAME" > "$INI_FILE_TEMP"
    echo "extension=$EXT_NAME.so" >> "$INI_FILE_TEMP"
    
    PHP_INI_SCAN_DIR=$(php -i | grep 'Scan this dir for additional .ini files' | awk '{print $NF}')
    
if [ -d "$PHP_INI_SCAN_DIR" ]; then
    echo "   -> Diretório de configuração encontrado: $PHP_INI_SCAN_DIR"
    
    if command -v phpenmod &> /dev/null; then
        echo "   -> Usando 'phpenmod' para habilitar a extensão..."
        
        PHP_VERSION=$(php -r "echo PHP_MAJOR_VERSION . '.' . PHP_MINOR_VERSION;")
        MODS_AVAILABLE_DIR="/etc/php/$PHP_VERSION/mods-available"
        
        if [ -d "$MODS_AVAILABLE_DIR" ]; then
            echo "   -> Copiando $INI_FILE_TEMP para $MODS_AVAILABLE_DIR..."
            
            if cp "$INI_FILE_TEMP" "$MODS_AVAILABLE_DIR/"; then
                echo "   -> INI copiado. Habilitando com 'phpenmod'..."
                phpenmod "$EXT_NAME"
                echo "--- BUILD E INSTALAÇÃO PERMANENTE CONCLUÍDOS COM SUCESSO! ---"
                echo "Extensão $EXT_NAME habilitada permanentemente."
                echo "Lembre-se de REINICIAR o seu servidor web/FPM para aplicar as mudanças."
            else
                echo "   \033[33mAVISO: Falha de Permissão. Não foi possível copiar o INI para $MODS_AVAILABLE_DIR.\033[0m"
                echo "   \033[33mA extensão foi compilada, mas você deve habilitá-la manualmente com 'sudo':\033[0m"
                echo "   \$ \033[33msudo cp $INI_FILE_TEMP $MODS_AVAILABLE_DIR/\033[0m"
                echo "   \$ \033[33msudo phpenmod $EXT_NAME\033[0m"
            fi
            
        else
            echo "   -> Tentando copiar o INI para o diretório de scan: $PHP_INI_SCAN_DIR"
            
            if cp "$INI_FILE_TEMP" "$PHP_INI_SCAN_DIR/"; then
                echo "--- BUILD E INSTALAÇÃO PERMANENTE CONCLUÍDOS COM SUCESSO! ---"
                echo "Extensão $EXT_NAME configurada em $PHP_INI_SCAN_DIR."
                echo "Lembre-se de REINICIAR o seu servidor web/FPM para aplicar as mudanças."
            else
                echo "   \033[33mAVISO: Falha de Permissão. Não foi possível copiar o INI para $PHP_INI_SCAN_DIR.\033[0m"
                echo "   \033[33mA extensão foi compilada, mas você deve habilitá-la manualmente com 'sudo':\033[0m"
                echo "   (Caminho INI temporário: $INI_FILE_TEMP)"
                echo "   (Caminho de destino: $PHP_INI_SCAN_DIR)"
                echo "   \$ \033[33msudo cp $INI_FILE_TEMP $PHP_INI_SCAN_DIR/\033[0m"
            fi
        fi

    else
        echo "   -> 'phpenmod' não encontrado. Tentando copiar o INI para o diretório de scan: $PHP_INI_SCAN_DIR"
        
        if cp "$INI_FILE_TEMP" "$PHP_INI_SCAN_DIR/"; then
            echo "--- BUILD E INSTALAÇÃO PERMANENTE CONCLUÍDOS COM SUCESSO! ---"
            echo "Extensão $EXT_NAME configurada em $PHP_INI_SCAN_DIR."
            echo "Lembre-se de REINICIAR o seu servidor web/FPM para aplicar as mudanças."
        else
            echo "   \033[33mAVISO: Falha de Permissão. Não foi possível copiar o INI para $PHP_INI_SCAN_DIR.\033[0m"
            echo "   \033[33mA extensão foi compilada, mas você deve habilitá-la manualmente com 'sudo':\033[0m"
            echo "   (Caminho INI temporário: $INI_FILE_TEMP)"
            echo "   (Caminho de destino: $PHP_INI_SCAN_DIR)"
            echo "   \$ \033[33msudo cp $INI_FILE_TEMP $PHP_INI_SCAN_DIR/\033[0m"
        fi
    fi

else
    echo "--- BUILD CONCLUÍDO. CONFIGURAÇÃO MANUAL NECESSÁRIA ---"
    echo "Não foi possível determinar o diretório de scan do PHP."
    echo "Crie o arquivo $EXT_NAME.ini em seu diretório de configuração do PHP com o conteúdo:"
    echo "extension=$EXT_NAME.so"
fi