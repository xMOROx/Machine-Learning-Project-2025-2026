import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

OBJECT_CATEGORIES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class BatchNormalize:
    def __init__(self, mean, std, device=None):
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


def encode_labels_with_bbox(target):
    size = target["annotation"]["size"]
    orig_dims = (int(size["height"]), int(size["width"]))

    ls = target["annotation"]["object"]
    idx_to_bbox = {}
    labels = []

    objects = [ls] if isinstance(ls, dict) else ls

    for obj in objects:
        difficult = int(obj.get("difficult", 0))
        if difficult == 0:
            class_idx = OBJECT_CATEGORIES.index(obj["name"])
            labels.append(class_idx)

            if class_idx not in idx_to_bbox:
                idx_to_bbox[class_idx] = []

            bbox = obj["bndbox"]
            idx_to_bbox[class_idx].append(
                (
                    int(bbox["xmin"]),
                    int(bbox["ymin"]),
                    int(bbox["xmax"]),
                    int(bbox["ymax"]),
                )
            )

    label_vector = np.zeros(len(OBJECT_CATEGORIES))
    label_vector[labels] = 1

    return torch.from_numpy(label_vector), idx_to_bbox, orig_dims


def compute_epg_score(attribution, bboxes, orig_dims, target_size=224):
    h_scale = target_size / orig_dims[0]
    w_scale = target_size / orig_dims[1]

    mask = np.zeros((target_size, target_size))

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        xmin = int(xmin * w_scale)
        ymin = int(ymin * h_scale)
        xmax = int(xmax * w_scale)
        ymax = int(ymax * h_scale)
        mask[ymin:ymax, xmin:xmax] = 1

    attribution = np.abs(attribution)
    if attribution.ndim == 3:
        attribution = attribution.sum(0)

    if attribution.shape != (target_size, target_size):
        attribution = np.array(
            Image.fromarray(attribution).resize((target_size, target_size))
        )

    inside_energy = (attribution * mask).sum()
    total_energy = attribution.sum() + 1e-8

    return inside_energy / total_energy


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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EPG on VOC dataset")
    parser.add_argument("--model-config", help="Path to model config (for mmpretrain)")
    parser.add_argument(
        "--model-checkpoint", required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--data-root", required=True, help="Path to VOCdevkit")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--attribution-methods",
        nargs="+",
        default=["gradcam", "ixg"],
        choices=["gradcam", "ixg", "intgrad", "gbp"],
    )
    parser.add_argument("--year", default="2007", choices=["2007", "2012"])
    parser.add_argument("--split", default="val", choices=["train", "val", "trainval"])
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
            Resize((224, 224)),
            ToTensor(),
        ]
    )

    dataset = VOCDetection(
        root=args.data_root,
        year=args.year,
        image_set=args.split,
        download=False,
        transform=transform,
    )

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

    results = {method: [] for method in args.attribution_methods}

    print(f"Evaluating EPG on {len(dataset)} VOC images...")

    for idx, (data, target) in enumerate(loader):
        data = data.to(device)
        data.requires_grad = True

        labels, idx_to_bbox, orig_dims = encode_labels_with_bbox(target)

        if args.model_type == "std":
            data_norm = norm_fn(data)
        else:
            data_norm = torch.cat([data, 1 - data], dim=1)

        for class_idx, bboxes in idx_to_bbox.items():
            for method in args.attribution_methods:
                try:
                    if method == "gradcam":
                        attribution = get_gradcam_attribution(
                            model, data_norm, class_idx
                        )
                    elif method == "ixg":
                        attribution = get_input_x_gradient_attribution(
                            model, data_norm, class_idx
                        )

                    epg = compute_epg_score(attribution, bboxes, orig_dims)
                    results[method].append(epg)
                except Exception as e:
                    print(
                        f"Error computing {method} for image {idx}, class {class_idx}: {e}"
                    )

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} images")

    print("\n=== EPG Results ===")
    for method in args.attribution_methods:
        if results[method]:
            mean_epg = np.mean(results[method])
            print(f"{method}: EPG={mean_epg:.4f}")

    results_file = os.path.join(args.output_dir, "epg_results.npy")
    np.save(results_file, results)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    from PIL import Image

    main()
