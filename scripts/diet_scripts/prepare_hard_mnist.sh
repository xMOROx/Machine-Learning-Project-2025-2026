#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Prepare Hard MNIST dataset from Colorized MNIST"
    echo ""
    echo "Options:"
    echo "  --data-dir    Custom data directory (default: ${DATA_DIR}/diet)"
    echo "  -h, --help    Show this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

DIET_DATA_DIR="${DATA_DIR}/diet"
COLORIZED_MNIST_DIR="${DIET_DATA_DIR}/colorized-MNIST"
HARD_MNIST_DIR="${DIET_DATA_DIR}/hard_mnist"

if [ ! -d "${COLORIZED_MNIST_DIR}" ]; then
    log_error "Colorized MNIST not found at ${COLORIZED_MNIST_DIR}"
    log_info "Run download_diet_data.sh --mnist first"
    exit 1
fi

log_info "Preparing Hard MNIST dataset..."

cd "${PROJECT_ROOT}"

python3 "${SCRIPT_DIR}/diet_scripts/python/prepare_hard_mnist.py" \
    --input-dir "${COLORIZED_MNIST_DIR}" \
    --output-dir "${HARD_MNIST_DIR}"

log_success "Hard MNIST dataset prepared at ${HARD_MNIST_DIR}"
exit 0

# Legacy inline script (not used)
python3 << EOF_UNUSED
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import random
import os

def construct_data(path, out_path):
    small_resize = transforms.Resize(56)
    resize = transforms.Resize(112)
    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()
    
    files = glob.glob(path)
    
    for i, f in enumerate(files):
        if not f.endswith(".png"):
            continue
        
        label = int(f.split("/")[-2])
        image = Image.open(f)
        t = to_tensor(resize(image))
        
        new_tensor = torch.ones((3, 224, 224)).to(t.dtype) * t[:, 0:1, 0:1]
        
        num_lines = torch.randint(low=0, high=3, size=(2,))
        for j in range(num_lines[0]):
            width = random.randint(0, 50)
            ind = random.randint(0, 223 - (width + 1))
            new_tensor[:, ind:ind+width, :] = torch.rand(3, width, 224)
        for j in range(num_lines[1]):
            width = random.randint(0, 50)
            ind = random.randint(0, 223 - (width + 1))
            new_tensor[:, :, ind:ind+width] = torch.rand(3, 224, width)
        
        f2 = files[random.randint(0, len(files) - 1)]
        while not f2.endswith(".png"):
            f2 = files[random.randint(0, len(files) - 1)]
        image2 = Image.open(f2)
        t2 = to_tensor(small_resize(image2))
        ind2 = torch.randint(low=0, high=164, size=(2,))
        small_digit_background = torch.where(torch.sum(t2, 0) <= 2, 0, 1)
        t2 *= small_digit_background
        new_tensor[:, ind2[0]:ind2[0]+56, ind2[1]:ind2[1]+56] += (0.98 * t2)
        
        ind = torch.randint(low=0, high=112, size=(2,))
        new_tensor[:, ind[0]:ind[0]+112, ind[1]:ind[1]+112] = t
        new_tensor = new_tensor.clamp(max=1, min=0)
        
        new_image = to_image(new_tensor)
        out_file = os.path.join(out_path, str(label), f"img_{i}.png")
        new_image.save(out_file)
        
        if i % 1000 == 0:
            print(f"Processed {i} images...")

colorized_mnist = "${COLORIZED_MNIST_DIR}"
hard_mnist = "${HARD_MNIST_DIR}"

print("Processing training data...")
construct_data(f"{colorized_mnist}/training/*/*.png", f"{hard_mnist}/training")

print("Processing testing data...")
construct_data(f"{colorized_mnist}/testing/*/*.png", f"{hard_mnist}/testing")

print("Hard MNIST preparation complete!")
EOF

log_success "Hard MNIST dataset prepared at ${HARD_MNIST_DIR}"
