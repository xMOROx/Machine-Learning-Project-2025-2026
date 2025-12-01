#!/bin/bash
#
# This script downloads 4 datasets required for the XAI project.
# 1. CIFAR-10        (for CNNs)
# 2. GLUE SST-2      (for BERT)
# 3. Colorized MNIST (for SVM / Logistic Regression)
# 4. Adult           (for Random Forest / LightGBM)
#

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' 

set -e

DATA_ROOT="data"
mkdir -p $DATA_ROOT
echo -e "${YELLOW}Downloading all datasets to directory: ${CYAN}$DATA_ROOT${NC}"
echo -e "${YELLOW}============================================================${NC}"

echo -e "\n${YELLOW}[1/4] Downloading CIFAR-10 (for CNN)...${NC}"
PYTHON_CIFAR_TEMPLATE=$(cat << END_OF_SCRIPT
import torchvision
import os
path = os.path.join('$DATA_ROOT', 'cifar10')
print(f'Downloading CIFAR-10 to {path}...')
try:
    torchvision.datasets.CIFAR10(root=path, train=True, download=True)
    torchvision.datasets.CIFAR10(root=path, train=False, download=True)
    print(f'{GREEN}CIFAR-10 download complete.{NC}')
except Exception as e:
    print(f'{RED}Failed to download CIFAR-10. Check your connection or torchvision library. Error: {e}{NC}')
    exit(1)
END_OF_SCRIPT
)
python3 -c "$PYTHON_CIFAR_TEMPLATE"

echo -e "\n${YELLOW}[2/4] Downloading GLUE SST-2 (for BERT)...${NC}"
PYTHON_GLUE_TEMPLATE=$(cat << END_OF_SCRIPT
from datasets import load_dataset
import os
path = os.path.join('$DATA_ROOT', 'glue_sst2')
print(f'Downloading GLUE/SST-2 (cache) to {path}...')
try:
    load_dataset('glue', 'sst2', cache_dir=path)
    print(f'{GREEN}GLUE SST-2 download complete.{NC}')
except Exception as e:
    print(f'{RED}Failed to download GLUE. Check your connection or datasets library. Error: {e}{NC}')
    exit(1)
"
END_OF_SCRIPT
)
python3 -c "$PYTHON_GLUE_TEMPLATE"

echo -e "\n${YELLOW}[3/4] Cloning Colorized MNIST (for SVM/Logistic Regression)...${NC}"
COLOR_MNIST_DIR="$DATA_ROOT/colorized-MNIST"
COLOR_MNIST_REPO="https://github.com/jayaneetha/colorized-MNIST.git"

if [ -d "$COLOR_MNIST_DIR" ]; then
    echo -e "${YELLOW}Directory ${CYAN}$COLOR_MNIST_DIR${YELLOW} already exists, skipping clone.${NC}"
else
    echo -e "Cloning repository ${CYAN}$COLOR_MNIST_REPO${NC} into ${CYAN}$COLOR_MNIST_DIR${NC}..."
    git clone $COLOR_MNIST_REPO $COLOR_MNIST_DIR
    echo -e "${GREEN}Colorized MNIST repository cloned successfully.${NC}"
fi

echo -e "\n${YELLOW}[4/4] Downloading Adult (Census Income) (for Random Forest/LGBM)...${NC}"
ADULT_DIR="$DATA_ROOT/adult"
mkdir -p $ADULT_DIR
BASE_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
FILES_ADULT=("adult.data" "adult.test" "adult.names")

for file in "${FILES_ADULT[@]}"; do
    if [ ! -f "$ADULT_DIR/$file" ]; then
        echo -e "Downloading ${CYAN}$file${NC}..."
        curl -sL -o "$ADULT_DIR/$file" "$BASE_URL/$file" || wget -q -O "$ADULT_DIR/$file" "$BASE_URL/$file"
    else
        echo -e "${YELLOW}$file already exists, skipping.${NC}"
    fi
done
echo -e "${GREEN}Adult (Census) download complete.${NC}"

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}All dataset downloads finished successfully!${NC}"
echo -e "${YELLOW}Contents of the '$DATA_ROOT' directory:${NC}"
ls -l $DATA_ROOT