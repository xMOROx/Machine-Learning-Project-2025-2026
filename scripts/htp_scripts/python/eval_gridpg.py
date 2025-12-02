import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    ToTensor,
    GaussianBlur,
)
from PIL import Image


class GridPGDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform or Compose([ToTensor()])

        with open(data_file, "r") as f:
            self.images = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_root, img_name)
        image = Image.open(img_path).convert("RGB")

        classes = img_name.split("_")[1:]
        classes = [int(c.split(".")[0]) for c in classes]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(classes)


class BatchNormalize:
    def __init__(self, mean, std, device=None):
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


def compute_gridpg_score(contribution_map, index, map_size=3, k=None):
    h, w = contribution_map.shape[:2]
    assert h == w, "height should be equal to width"

    contribution_map = torch.from_numpy(contribution_map)
    if k is None:
        contribution_map = F.avg_pool2d(
            contribution_map.unsqueeze(0), 5, padding=2, stride=1
        )[0]
    else:
        contribution_map = GaussianBlur(k, sigma=k / 4)(contribution_map.unsqueeze(0))[
            0
        ]
    contribution_map = contribution_map.numpy()
    contribution_map[contribution_map < 0] = 0

    grid_h = h // map_size
    grid_w = w // map_size

    row = index // map_size
    col = index % map_size

    target_region = contribution_map[
        row * grid_h : (row + 1) * grid_h, col * grid_w : (col + 1) * grid_w
    ]
    target_sum = target_region.sum()
    total_sum = contribution_map.sum() + 1e-8

    return target_sum / total_sum


def compute_entropy(contribution_map):
    contribution_map = np.abs(contribution_map)
    contribution_map = contribution_map / (contribution_map.sum() + 1e-8)
    contribution_map = contribution_map.flatten()
    contribution_map = contribution_map[contribution_map > 0]
    return -np.sum(contribution_map * np.log(contribution_map + 1e-8))


def compute_gini(contribution_map):
    contribution_map = np.abs(contribution_map).flatten()
    contribution_map = np.sort(contribution_map)
    n = len(contribution_map)
    cumsum = np.cumsum(contribution_map)
    return (n + 1 - 2 * np.sum(cumsum) / (cumsum[-1] + 1e-8)) / n


def get_gradcam_attribution(model, data, target_class, layer_name="layer4"):
    from captum.attr import LayerGradCam, LayerAttribution

    layer = getattr(model, layer_name) if hasattr(model, layer_name) else model.layer4
    gradcam = LayerGradCam(model, layer)

    attribution = gradcam.attribute(data, target=target_class)
    attribution = LayerAttribution.interpolate(
        attribution, (data.shape[2], data.shape[3])
    )
    return attribution.squeeze().detach().cpu().numpy()


def get_input_x_gradient_attribution(model, data, target_class):
    from captum.attr import InputXGradient

    ixg = InputXGradient(model)
    attribution = ixg.attribute(data, target=target_class)
    return attribution.squeeze().sum(0).detach().cpu().numpy()


def get_integrated_gradients_attribution(model, data, target_class, steps=50):
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)
    attribution = ig.attribute(data, target=target_class, n_steps=steps)
    return attribution.squeeze().sum(0).detach().cpu().numpy()


def get_guided_backprop_attribution(model, data, target_class):
    from captum.attr import GuidedBackprop

    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(data, target=target_class)
    return attribution.squeeze().sum(0).detach().cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GridPG, Compactness, and Complexity"
    )
    parser.add_argument("--model-config", help="Path to model config (for mmpretrain)")
    parser.add_argument(
        "--model-checkpoint", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-file", required=True, help="Path to grid image list file"
    )
    parser.add_argument(
        "--data-root", required=True, help="Path to grid images directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--attribution-methods",
        nargs="+",
        default=["gradcam", "ixg"],
        choices=["gradcam", "ixg", "intgrad", "gbp"],
    )
    parser.add_argument("--map-size", type=int, default=3, choices=[2, 3])
    parser.add_argument("--num-classes", type=int, default=1000)
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

    transform = Compose([ToTensor()])
    dataset = GridPGDataset(args.data_file, args.data_root, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    try:
        import mmpretrain

        model = mmpretrain.get_model(
            args.model_config, pretrained=args.model_checkpoint
        )
        model.data_preprocessor = None
    except (ImportError, TypeError):
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        model.load_state_dict(torch.load(args.model_checkpoint, map_location="cpu"))

    model.to(device)
    model.eval()

    results = {
        method: {"gridpg": [], "entropy": [], "gini": []}
        for method in args.attribution_methods
    }

    print(f"Evaluating {len(dataset)} grid images...")

    for idx, (data, classes) in enumerate(loader):
        data = data.to(device)
        data.requires_grad = True

        if args.model_type == "std":
            data_norm = norm_fn(data)
        else:
            data_norm = torch.cat([data, 1 - data], dim=1)

        num_cells = args.map_size * args.map_size

        for cell_idx in range(min(len(classes[0]), num_cells)):
            target_class = classes[0][cell_idx].item()

            for method in args.attribution_methods:
                try:
                    if method == "gradcam":
                        attribution = get_gradcam_attribution(
                            model, data_norm, target_class
                        )
                    elif method == "ixg":
                        attribution = get_input_x_gradient_attribution(
                            model, data_norm, target_class
                        )
                    elif method == "intgrad":
                        attribution = get_integrated_gradients_attribution(
                            model, data_norm, target_class
                        )
                    elif method == "gbp":
                        attribution = get_guided_backprop_attribution(
                            model, data_norm, target_class
                        )

                    gridpg = compute_gridpg_score(attribution, cell_idx, args.map_size)
                    entropy = compute_entropy(attribution)
                    gini = compute_gini(attribution)

                    results[method]["gridpg"].append(gridpg)
                    results[method]["entropy"].append(entropy)
                    results[method]["gini"].append(gini)
                except Exception as e:
                    print(
                        f"Error computing {method} for image {idx}, cell {cell_idx}: {e}"
                    )

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} images")

    print("\n=== Results ===")
    for method in args.attribution_methods:
        if results[method]["gridpg"]:
            mean_gridpg = np.mean(results[method]["gridpg"])
            mean_entropy = np.mean(results[method]["entropy"])
            mean_gini = np.mean(results[method]["gini"])
            print(
                f"{method}: GridPG={mean_gridpg:.4f}, Entropy={mean_entropy:.4f}, Gini={mean_gini:.4f}"
            )

    results_file = os.path.join(args.output_dir, "results.npy")
    np.save(results_file, results)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
