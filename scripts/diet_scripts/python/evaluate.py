import os
import sys
import argparse
import glob
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image
from tqdm import tqdm


class DatasetFromDisk(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.data[idx]).convert("RGB"))
        return idx, image, self.labels[idx]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx, self.data[idx], self.labels[idx]


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


def load_xray_data(data_path, noise_class="NORMAL"):
    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    for split in ["train", "val"]:
        for i, f in enumerate(glob.glob(os.path.join(data_path, split, "*", "*.jpeg"))):
            img = transform(Image.open(f).convert("RGB"))
            label_name = os.path.basename(os.path.dirname(f))
            label = 0 if label_name == "NORMAL" else 1
            if label_name == noise_class:
                j = (i % 14) * 16
                img[:, :16, j : j + 16] += torch.normal(
                    mean=torch.zeros_like(img[:, :16, j : j + 16]),
                    std=0.05 * (img.max() - img.min()),
                )
            train_imgs.append(img)
            train_labels.append(label)

    for i, f in enumerate(glob.glob(os.path.join(data_path, "test", "*", "*.jpeg"))):
        img = transform(Image.open(f).convert("RGB"))
        label_name = os.path.basename(os.path.dirname(f))
        label = 0 if label_name == "NORMAL" else 1
        if label_name == noise_class:
            j = (i % 14) * 16
            img[:, :16, j : j + 16] += torch.normal(
                mean=torch.zeros_like(img[:, :16, j : j + 16]),
                std=0.05 * (img.max() - img.min()),
            )
        test_imgs.append(img)
        test_labels.append(label)

    return train_imgs, train_labels, test_imgs, test_labels


def pixel_perturbation(model, data_loader, mask, perturbation_percents, args):
    model.eval()
    ups = nn.Upsample(scale_factor=args.ups, mode="bilinear")
    sm = nn.Softmax(dim=1)

    results = {p: {"correct": 0, "total": 0} for p in perturbation_percents}

    with torch.no_grad():
        for idx, batch_x, batch_labels in tqdm(data_loader, desc="Evaluating"):
            batch_x = batch_x.to(args.device)
            batch_labels = batch_labels.to(args.device)
            batch_mask = ups(mask[idx]).to(args.device)

            for p in perturbation_percents:
                threshold = torch.quantile(
                    batch_mask.flatten(1), 1 - p / 100, dim=1, keepdim=True
                )
                threshold = threshold.unsqueeze(-1).unsqueeze(-1)
                binary_mask = (batch_mask >= threshold).float()

                perturbed = batch_x * binary_mask
                preds = model(perturbed)
                predicted = torch.argmax(preds, dim=1)

                results[p]["correct"] += (predicted == batch_labels).sum().item()
                results[p]["total"] += len(batch_labels)

    print("\nPixel Perturbation Results:")
    for p in perturbation_percents:
        acc = results[p]["correct"] / results[p]["total"] * 100
        print(f"  Top {p}%: {acc:.2f}%")

    return results


def compute_iou(model, data_loader, mask, args):
    model.eval()
    ups = nn.Upsample(scale_factor=args.ups, mode="bilinear")

    total_iou = 0
    count = 0

    with torch.no_grad():
        for idx, batch_x, batch_labels in tqdm(data_loader, desc="Computing IOU"):
            batch_x = batch_x.to(args.device)
            batch_mask = ups(mask[idx]).to(args.device)

            gt_mask = torch.where(
                torch.sum(batch_x, dim=1, keepdim=True) >= 2.5, 1.0, 0.0
            )
            pred_mask = (batch_mask > 0.5).float()

            intersection = (gt_mask * pred_mask).sum(dim=(1, 2, 3))
            union = ((gt_mask + pred_mask) > 0).float().sum(dim=(1, 2, 3))

            iou = intersection / (union + 1e-8)
            total_iou += iou.sum().item()
            count += len(batch_labels)

    mean_iou = total_iou / count
    print(f"\nMean IOU: {mean_iou:.4f}")
    return mean_iou


def parse_args():
    parser = argparse.ArgumentParser(description="DiET Evaluation")
    parser.add_argument("--dataset", choices=["mnist", "xray", "celeba"], required=True)
    parser.add_argument("--data-dir", required=True, help="Path to dataset")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--mask-path", required=True, help="Path to mask directory")
    parser.add_argument("--mask-num", type=int, default=1, help="Mask number to use")
    parser.add_argument("--ups", type=int, default=4, help="Upsample factor")
    parser.add_argument(
        "--eval-type", choices=["perturbation", "iou", "all"], default="all"
    )
    parser.add_argument(
        "--perturbations", type=int, nargs="+", default=[10, 20, 50, 100]
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--noise-class", default="NORMAL")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"DiET Evaluation: {args.dataset}")
    print(f"Evaluation type: {args.eval_type}")
    print(f"Device: {args.device}")

    if args.dataset == "mnist":
        train_imgs, train_labels, test_imgs, test_labels = load_mnist_data(
            args.data_dir
        )
        num_classes = 10
        from_disk = True
    elif args.dataset == "xray":
        train_imgs, train_labels, test_imgs, test_labels = load_xray_data(
            args.data_dir, args.noise_class
        )
        num_classes = 2
        from_disk = False
    else:
        raise ValueError(f"Dataset {args.dataset} not fully supported for evaluation")

    if from_disk:
        train_loader = torch.utils.data.DataLoader(
            DatasetFromDisk(train_imgs, train_labels),
            batch_size=args.batch_size,
            shuffle=False,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            Dataset(train_imgs, train_labels), batch_size=args.batch_size, shuffle=False
        )

    model = resnet34(weights=None)
    model.fc = nn.Linear(512, num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.to(args.device)
    model.eval()

    mask_file = os.path.join(args.mask_path, f"mask_{args.mask_num}.pt")
    if not os.path.exists(mask_file):
        mask_file = os.path.join(args.mask_path, "mask_0.pt")

    mask = torch.load(mask_file, map_location="cpu")
    print(f"Loaded mask from {mask_file}")

    if args.eval_type in ["perturbation", "all"]:
        pixel_perturbation(model, train_loader, mask, args.perturbations, args)

    if args.eval_type in ["iou", "all"]:
        if args.dataset == "mnist":
            compute_iou(model, train_loader, mask, args)


if __name__ == "__main__":
    main()
