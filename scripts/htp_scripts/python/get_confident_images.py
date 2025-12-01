import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform or Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
            ]
        )

        with open(data_file, "r") as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_path = parts[0]
                label = int(parts[1])
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        full_path = os.path.join(self.data_root, img_path)
        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class BatchNormalize:
    def __init__(self, mean, std, device=None):
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


def save_image(tensor, path):
    from torchvision.utils import save_image

    save_image(tensor, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Get confident images for GridPG")
    parser.add_argument("--model-config", required=True, help="Path to model config")
    parser.add_argument(
        "--model-checkpoint", required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--data-file", required=True, help="Path to data file list")
    parser.add_argument("--data-root", required=True, help="Path to data root")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for confident images"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95, help="Confidence threshold"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=100, help="Max images per class"
    )
    parser.add_argument(
        "--num-classes", type=int, default=1000, help="Number of classes"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-type", choices=["std", "bcos"], default="std")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    norm_fn = BatchNormalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), device=device
    )

    transform = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
        ]
    )

    dataset = ImageDataset(args.data_file, args.data_root, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    try:
        import mmpretrain

        model = mmpretrain.get_model(
            args.model_config, pretrained=args.model_checkpoint
        )
    except ImportError:
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        model.load_state_dict(torch.load(args.model_checkpoint, map_location="cpu"))

    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    class_counts = {cls: 0 for cls in range(args.num_classes)}
    sample_index = 0
    correct = 0

    print(f"Processing {len(dataset)} images...")

    for loader_idx, (data, target) in enumerate(loader):
        target = target.to(device)

        if class_counts.get(int(target[0]), 0) >= args.max_per_class:
            continue

        if args.model_type == "std":
            data_norm = norm_fn(data.clone().to(device))
            output = model(data_norm)
            output_probs = nn.Softmax(dim=1)(output)
        else:
            bcos_data = torch.cat([data.to(device), 1 - data.to(device)], dim=1)
            output = model(bcos_data)
            output_probs = nn.Sigmoid()(output)

        pred_correct = (output_probs.argmax(1) == target).sum().item()
        correct += pred_correct

        if pred_correct == 0 or float(output_probs.max()) < args.confidence:
            continue

        fname = f"{sample_index}_{target[0].cpu().item()}.png"
        save_image(data[0], os.path.join(args.output_dir, fname))

        sample_index += 1
        class_counts[int(target[0])] += 1

        if (loader_idx + 1) % 100 == 0:
            print(
                f"Processed {loader_idx + 1}/{len(loader)}, saved {sample_index} images, acc: {correct/(loader_idx+1):.4f}"
            )

    print(f"\nTotal confident images saved: {sample_index}")
    print(f"Overall accuracy: {correct/len(dataset):.4f}")


if __name__ == "__main__":
    main()
