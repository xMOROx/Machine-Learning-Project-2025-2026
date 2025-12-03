import os
import argparse
import time
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image


class DatasetFromDisk(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.data[idx]).convert("RGB"))
        return idx, img, self.labels[idx]


def load_mnist_data(data_path):
    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []

    for f in glob.glob(os.path.join(data_path, "training", "*", "*.png")):
        train_imgs.append(f)
        train_labels.append(int(os.path.basename(os.path.dirname(f))))

    for f in glob.glob(os.path.join(data_path, "testing", "*", "*.png")):
        test_imgs.append(f)
        test_labels.append(int(os.path.basename(os.path.dirname(f))))

    return train_imgs, train_labels, test_imgs, test_labels


def load_xray_data(data_path):
    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []

    for split in ["train", "val"]:
        for f in glob.glob(os.path.join(data_path, split, "*", "*.jpeg")):
            train_imgs.append(f)
            label = 0 if os.path.basename(os.path.dirname(f)) == "NORMAL" else 1
            train_labels.append(label)

    for f in glob.glob(os.path.join(data_path, "test", "*", "*.jpeg")):
        test_imgs.append(f)
        label = 0 if os.path.basename(os.path.dirname(f)) == "NORMAL" else 1
        test_labels.append(label)

    return train_imgs, train_labels, test_imgs, test_labels


def load_celeba_data(data_path, attr_file=None):
    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []

    for f in glob.glob(os.path.join(data_path, "train", "*.jpg")):
        train_imgs.append(f)
        train_labels.append(0)

    for f in glob.glob(os.path.join(data_path, "test", "*.jpg")):
        test_imgs.append(f)
        test_labels.append(0)

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

    return total_loss / len(train_loader), 100.0 * correct / total


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

    return total_loss / len(test_loader), 100.0 * correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline model for DiET")
    parser.add_argument("--dataset", choices=["mnist", "xray", "celeba"], required=True)
    parser.add_argument("--data-dir", required=True, help="Path to dataset")
    parser.add_argument("--output-path", required=True, help="Path to save model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training baseline model for {args.dataset}")
    print(f"Device: {args.device}")

    if args.dataset == "mnist":
        train_imgs, train_labels, test_imgs, test_labels = load_mnist_data(
            args.data_dir
        )
        num_classes = 10
    elif args.dataset == "xray":
        train_imgs, train_labels, test_imgs, test_labels = load_xray_data(args.data_dir)
        num_classes = 2
    elif args.dataset == "celeba":
        train_imgs, train_labels, test_imgs, test_labels = load_celeba_data(
            args.data_dir
        )
        num_classes = 3

    print(
        f"Train samples: {len(train_imgs)}, Test samples: {len(test_imgs)}, Classes: {num_classes}"
    )

    train_loader = torch.utils.data.DataLoader(
        DatasetFromDisk(train_imgs, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        DatasetFromDisk(test_imgs, test_labels),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = resnet34(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
        )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_path)
    print(f"Model saved to {args.output_path}")


if __name__ == "__main__":
    main()
