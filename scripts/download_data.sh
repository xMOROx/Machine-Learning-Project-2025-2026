#!/bin/bash
#
# This script downloads 4 datasets required for the XAI project.
# 1. CIFAR-10   (for CNNs)
# 2. GLUE SST-2 (for BERT)
# 3. MNIST      (for SVM / Logistic Regression)
# 4. Adult      (for Random Forest / LightGBM)
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
PYTHON_SCRIPT_TEMPLATE=$(cat << 'END_SCRIPT'
import torchvision
import os
path = os.path.join('$DATA_ROOT', 'cifar10')
print(f'Downloading CIFAR-10 to {path}...')
try:
    torchvision.datasets.CIFAR10(root=path, train=True, download=True)
    torchvision.datasets.CIFAR10(root=path, train=False, download=True)
    print('${GREEN}CIFAR-10 download complete.${NC}')
except Exception as e:
    print(f'${RED}Failed to download CIFAR-10. Check your connection or torchvision library. Error: {e}${NC}')
    exit(1)
END_SCRIPT
)

python3 -c "$PYTHON_SCRIPT_TEMPLATE"

echo -e "\n${YELLOW}[2/4] Downloading GLUE SST-2 (for BERT)...${NC}"
python3 -c "
from datasets import load_dataset
import os
path = os.path.join('$DATA_ROOT', 'glue_sst2')
print(f'Downloading GLUE/SST-2 (cache) to {path}...')
try:
    load_dataset('glue', 'sst2', cache_dir=path)
    print('${GREEN}GLUE SST-2 download complete.${NC}')
except Exception as e:
    print(f'${RED}Failed to download GLUE. Check your connection or datasets library. Error: {e}${NC}')
    exit(1)
"

echo -e "\n${YELLOW}[3/4] Downloading MNIST (for SVM/Logistic Regression)...${NC}"
MNIST_DIR="$DATA_ROOT/mnist"
mkdir -p $MNIST_DIR
BASE_URL="http://yann.lecun.com/exdb/mnist/"
FILES=("train-images-idx3-ubyte.gz" "train-labels-idx1-ubyte.gz" "t10k-images-idx3-ubyte.gz" "t10k-labels-idx1-ubyte.gz")

for file in "${FILES[@]}"; do
    if [ ! -f "$MNIST_DIR/$file" ]; then
        echo -e "Downloading ${CYAN}$file${NC}..."
        curl -sL -o "$MNIST_DIR/$file" "$BASE_URL$file" || wget -q -O "$MNIST_DIR/$file" "$BASE_URL$file"
    else
        echo -e "${YELLOW}$file already exists, skipping.${NC}"
    fi
done
echo -e "${GREEN}MNIST download complete (.gz files, to be unpacked by a loader).${NC}"

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