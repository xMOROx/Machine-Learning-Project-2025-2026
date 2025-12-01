#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Train baseline models for DiET"
    echo ""
    echo "Options:"
    echo "  --dataset     Dataset to train on: mnist, xray, celeba (default: mnist)"
    echo "  --epochs      Number of training epochs (default: 10)"
    echo "  --batch-size  Batch size (default: 64)"
    echo "  --lr          Learning rate (default: 0.001)"
    echo "  --data-dir    Custom data directory"
    echo "  --output-dir  Custom output directory"
    echo "  --gpu         GPU device ID (default: 0)"
    echo "  -h, --help    Show this help message"
}

DATASET="mnist"
EPOCHS=10
BATCH_SIZE=64
LR=0.001
GPU=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUTS_DIR="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
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
DIET_MODELS_DIR="${OUTPUTS_DIR}/diet/trained_models"
ensure_dir "${DIET_MODELS_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"

log_info "Training baseline model for DiET"
log_info "Dataset: ${DATASET}"
log_info "Epochs: ${EPOCHS}"
log_info "Batch size: ${BATCH_SIZE}"
log_info "Learning rate: ${LR}"
log_info "GPU: ${GPU}"

cd "${PROJECT_ROOT}"

case ${DATASET} in
    mnist)
        DATA_PATH="${DIET_DATA_DIR}/hard_mnist/"
        MODEL_OUT="${DIET_MODELS_DIR}/hard_mnist_rn34.pth"
        ;;
    xray)
        DATA_PATH="${DIET_DATA_DIR}/chest-xray/"
        MODEL_OUT="${DIET_MODELS_DIR}/xray_rn34.pth"
        ;;
    celeba)
        DATA_PATH="${DIET_DATA_DIR}/celeba/"
        MODEL_OUT="${DIET_MODELS_DIR}/celeba_rn34.pth"
        ;;
    *)
        log_error "Unknown dataset: ${DATASET}"
        exit 1
        ;;
esac

python3 "${SCRIPT_DIR}/diet_scripts/python/train_baseline.py" \
    --dataset "${DATASET}" \
    --data-dir "${DATA_PATH}" \
    --output-path "${MODEL_OUT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}"

log_success "Training completed. Model saved to ${MODEL_OUT}"
exit 0

# Legacy inline script (not used)
python3 << EOF_UNUSED
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image
import glob
import os
import time

class DatasetFromDisk(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.data[idx]).convert('RGB'))
        return idx, img, self.labels[idx]

def load_data(data_path, dataset_type):
    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []
    
    if dataset_type == "mnist":
        train_files = glob.glob(data_path + "training/*/*")
        test_files = glob.glob(data_path + "testing/*/*")
        ext = "png"
        label_fn = lambda f: int(f.split("/")[-2])
    elif dataset_type == "xray":
        train_files = glob.glob(data_path + "train/*/*") + glob.glob(data_path + "val/*/*")
        test_files = glob.glob(data_path + "test/*/*")
        ext = "jpeg"
        label_fn = lambda f: 0 if f.split("/")[-2] == "NORMAL" else 1
    elif dataset_type == "celeba":
        train_files = glob.glob(data_path + "train/*")
        test_files = glob.glob(data_path + "test/*")
        ext = "jpg"
        label_fn = lambda f: int(f.split("/")[-1].split("_")[0])
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")
    
    for f in train_files:
        if f.endswith(ext):
            train_imgs.append(f)
            train_labels.append(label_fn(f))
    
    for f in test_files:
        if f.endswith(ext):
            test_imgs.append(f)
            test_labels.append(label_fn(f))
    
    return train_imgs, train_labels, test_imgs, test_labels

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for idx, inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for idx, inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(test_loader), 100. * correct / total

data_path = "${DATA_PATH}"
dataset_type = "${DATASET}"
model_out = "${MODEL_OUT}"
epochs = ${EPOCHS}
batch_size = ${BATCH_SIZE}
lr = ${LR}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading data...")
train_imgs, train_labels, test_imgs, test_labels = load_data(data_path, dataset_type)
num_classes = len(set(train_labels))
print(f"Train samples: {len(train_imgs)}, Test samples: {len(test_imgs)}, Classes: {num_classes}")

train_loader = torch.utils.data.DataLoader(
    DatasetFromDisk(train_imgs, train_labels), 
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    DatasetFromDisk(test_imgs, test_labels), 
    batch_size=batch_size, shuffle=False
)

model = resnet34(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

print("Starting training...")
for epoch in range(epochs):
    start = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    elapsed = time.time() - start
    
    print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

os.makedirs(os.path.dirname(model_out), exist_ok=True)
torch.save(model.state_dict(), model_out)
print(f"Model saved to {model_out}")
EOF

log_success "Training completed. Model saved to ${MODEL_OUT}"
