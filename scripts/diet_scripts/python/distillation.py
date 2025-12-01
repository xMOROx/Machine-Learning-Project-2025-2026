import os
import argparse
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx, self.data[idx], self.labels[idx]


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


class DatasetWithPreds(torch.utils.data.Dataset):
    def __init__(self, data, labels, model_preds, load_upfront=False):
        self.data = data
        self.labels = labels
        self.load_upfront = load_upfront
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.model_preds = model_preds

        if load_upfront:
            self.data = torch.zeros((len(data), 3, 224, 224))
            for i, path in enumerate(data):
                image = Image.open(path).convert("RGB")
                self.data[i] = self.transform(image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if not self.load_upfront:
            image = self.transform(Image.open(image).convert("RGB"))
        return idx, image, self.labels[idx], self.model_preds[idx]


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


def load_celeba_data(data_path):
    split_file = os.path.join(data_path, "split.csv")
    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []

    with open(split_file, "r") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",")
            file_path = os.path.join(data_path, parts[1])
            hair_label = int(parts[2])
            split = int(parts[4])

            if split == 0:
                train_imgs.append(file_path)
                train_labels.append(hair_label)
            else:
                test_imgs.append(file_path)
                test_labels.append(hair_label)

    return train_imgs, train_labels, test_imgs, test_labels


def get_predictions(model, data, labels, num_classes, device, from_disk=True):
    preds = torch.zeros((len(labels), num_classes))
    if from_disk:
        loader = torch.utils.data.DataLoader(
            DatasetFromDisk(data, labels), batch_size=1024, shuffle=False
        )
    else:
        loader = torch.utils.data.DataLoader(
            Dataset(data, labels), batch_size=1024, shuffle=False
        )

    sm = nn.Softmax(dim=1)
    with torch.no_grad():
        for idx, imgs, _ in loader:
            preds[idx] = sm(model(imgs.to(device))).cpu()

    return preds


def update_mask(mask, data_loader, model, mask_opt, simp_weight, args):
    mask = mask.requires_grad_(True)
    model.eval()

    sm = nn.Softmax(dim=1)
    ups = nn.Upsample(scale_factor=args.ups, mode="bilinear")
    metrics = torch.zeros((7,))
    num_samples = 0

    for idx, batch_x, batch_labels, pred_fb in data_loader:
        batch_x = batch_x.to(args.device)
        batch_mask = ups(mask[idx]).to(args.device)
        pred_fb = pred_fb.to(args.device)

        bg_means = torch.ones((len(idx), 3)) * torch.Tensor([0.527, 0.447, 0.403])
        bg_std = torch.ones((len(idx), 3)) * torch.Tensor([0.229, 0.224, 0.225])
        avg_val = (
            torch.normal(mean=bg_means, std=bg_std)
            .unsqueeze(2)
            .unsqueeze(2)
            .clamp(0, 1)
            .to(args.device)
        )

        pred_fs_d = sm(model(batch_x))
        pred_fs_s = sm(model((batch_mask * batch_x) + (1 - batch_mask) * avg_val))

        t1 = torch.linalg.vector_norm(pred_fb - pred_fs_d, 1)
        t2 = torch.linalg.vector_norm(pred_fs_d - pred_fs_s, 1)
        sim_heur = torch.linalg.vector_norm(batch_mask, 1) / (
            args.im_size * args.im_size
        )
        loss = (simp_weight * sim_heur + t1 + t2) / len(batch_x)

        mask_opt.zero_grad()
        loss.backward()
        mask_opt.step()

        with torch.no_grad():
            mask.copy_(mask.clamp(0, 1))
            t1_acc = (
                (torch.argmax(pred_fb, 1) == torch.argmax(pred_fs_s, 1)).sum().item()
            )
            fs_s_acc = (
                (torch.argmax(pred_fs_s, 1) == batch_labels.to(args.device))
                .sum()
                .item()
            )
            mask_l0 = (
                torch.linalg.vector_norm(batch_mask.flatten(), 0)
                / (batch_mask.shape[2] * batch_mask.shape[3])
            ).item()

        metrics += torch.Tensor(
            [
                loss.item() * len(batch_x),
                sim_heur.sum().item(),
                t1.sum().item(),
                t2.sum().item(),
                t1_acc,
                fs_s_acc,
                mask_l0,
            ]
        )
        num_samples += len(batch_labels)

    metrics /= num_samples
    print(
        f"[Mask] loss: {metrics[0]:.3f}, l1: {metrics[1]:.3f}, t1: {metrics[2]:.3f}, t2: {metrics[3]:.3f}, "
        f"t1_acc: {metrics[4]:.3f}, fs_s_acc: {metrics[5]:.3f}, l0: {metrics[6]:.3f}"
    )
    return metrics


def update_model(mask, data_loader, model, model_opt, args):
    mask = mask.requires_grad_(False)
    model.train()

    sm = nn.Softmax(dim=1)
    ups = nn.Upsample(scale_factor=args.ups, mode="bilinear")
    metrics = torch.zeros((5,))
    num_samples = 0

    for idx, batch_x, batch_labels, pred_fb in data_loader:
        batch_x = batch_x.to(args.device)
        batch_mask = ups(mask[idx]).to(args.device)
        pred_fb = pred_fb.to(args.device)

        bg_means = torch.ones((len(idx), 3)) * torch.Tensor([0.527, 0.447, 0.403])
        bg_std = torch.ones((len(idx), 3)) * torch.Tensor([0.229, 0.224, 0.225])
        avg_val = (
            torch.normal(mean=bg_means, std=bg_std)
            .unsqueeze(2)
            .unsqueeze(2)
            .clamp(0, 1)
            .to(args.device)
        )

        pred_fs_d = sm(model(batch_x))
        pred_fs_s = sm(model((batch_mask * batch_x) + (1 - batch_mask) * avg_val))

        t1 = torch.linalg.vector_norm(pred_fb - pred_fs_d, 1)
        t2 = torch.linalg.vector_norm(pred_fs_d - pred_fs_s, 1)
        loss = (t1 + t2) / len(batch_x)

        model_opt.zero_grad()
        loss.backward()
        model_opt.step()

        with torch.no_grad():
            t1_acc = (
                (torch.argmax(pred_fb, 1) == torch.argmax(pred_fs_s, 1)).sum().item()
            )
            fs_s_acc = (
                (torch.argmax(pred_fs_s, 1) == batch_labels.to(args.device))
                .sum()
                .item()
            )

        metrics += torch.Tensor(
            [
                loss.item() * len(batch_x),
                t1.sum().item(),
                t2.sum().item(),
                t1_acc,
                fs_s_acc,
            ]
        )
        num_samples += len(batch_labels)

    metrics /= num_samples
    print(
        f"[Model] loss: {metrics[0]:.3f}, t1: {metrics[1]:.3f}, t2: {metrics[2]:.3f}, "
        f"t1_acc: {metrics[3]:.3f}, fs_s_acc: {metrics[4]:.3f}"
    )
    return metrics


def evaluate_model(model, mask, train_loader, test_loader, args):
    model.eval()
    sm = nn.Softmax(dim=1)
    ups = nn.Upsample(scale_factor=args.ups, mode="bilinear")

    fs_s_acc, t1_acc, mask_l0, s_count = 0, 0, 0, 0

    with torch.no_grad():
        for idx, batch_x, batch_labels, pred_fb in train_loader:
            batch_x = batch_x.to(args.device)
            batch_mask = ups(mask[idx]).to(args.device)
            pred_fb = pred_fb.to(args.device)

            bg_means = torch.ones((len(idx), 3)) * torch.Tensor([0.527, 0.447, 0.403])
            bg_std = (
                0.1 * torch.ones((len(idx), 3)) * torch.Tensor([0.229, 0.224, 0.225])
            )
            avg_val = (
                torch.normal(mean=bg_means, std=bg_std)
                .unsqueeze(2)
                .unsqueeze(2)
                .clamp(0, 1)
                .to(args.device)
            )
            pred_fs_s = sm(model((batch_mask * batch_x) + (1 - batch_mask) * avg_val))

            t1_acc += (
                (torch.argmax(pred_fb, 1) == torch.argmax(pred_fs_s, 1)).sum().item()
            )
            fs_s_acc += (
                (torch.argmax(pred_fs_s, 1) == batch_labels.to(args.device))
                .sum()
                .item()
            )
            mask_l0 += (
                torch.linalg.vector_norm(batch_mask.flatten(), 0)
                / (batch_mask.shape[2] * batch_mask.shape[3])
            ).item()
            s_count += len(batch_labels)

    fs_t_acc, t2_acc, t_count = 0, 0, 0

    with torch.no_grad():
        for _, test_x, test_labels, fb_t in test_loader:
            fs_t = model(test_x.to(args.device))
            fb_t = fb_t.to(args.device)
            fs_t_acc += (
                (torch.argmax(fs_t, 1) == test_labels.to(args.device)).sum().item()
            )
            t2_acc += (torch.argmax(fb_t, 1) == torch.argmax(fs_t, 1)).sum().item()
            t_count += len(test_labels)

    print(
        f"[EVAL] t1_acc: {t1_acc/s_count:.3f}, fss_acc: {fs_s_acc/s_count:.3f}, "
        f"t2_acc: {t2_acc/t_count:.3f}, fst_acc: {fs_t_acc/t_count:.3f}, mask_l0: {mask_l0/s_count:.3f}"
    )


def distill(mask, model, train_loader, test_loader, mask_opt, model_opt, args):
    num_rounding_steps = args.rounding_steps
    rounding_scheme = [
        0.4 - r * (0.4 / num_rounding_steps) for r in range(num_rounding_steps)
    ]
    simp_weight = [
        1 - r * (0.9 / num_rounding_steps) for r in range(num_rounding_steps)
    ]

    evaluate_model(model, mask, train_loader, test_loader, args)

    for k in range(num_rounding_steps):
        print(f"\n=== Step {k} ===")

        print("Training mask...")
        mask_converged = False
        prev_loss, prev_prev_loss = float("inf"), float("inf")
        while not mask_converged:
            mask_metrics = update_mask(
                mask, train_loader, model, mask_opt, simp_weight[k], args
            )
            mask_loss = mask_metrics[0].item()
            mask_converged = (mask_loss >= 0.995 * prev_prev_loss) and (
                mask_loss <= 1.005 * prev_prev_loss
            )
            prev_prev_loss = prev_loss
            prev_loss = mask_loss

        torch.save(mask, os.path.join(args.output_dir, f"mask_{k}.pt"))
        with torch.no_grad():
            mask.copy_(torch.round(mask + rounding_scheme[k]))

        evaluate_model(model, mask, train_loader, test_loader, args)

        print("Training model...")
        model_converged = False
        prev_loss, prev_prev_loss = float("inf"), float("inf")
        while not model_converged:
            model_metrics = update_model(mask, train_loader, model, model_opt, args)
            model_loss = model_metrics[0].item()
            model_converged = (model_loss < 0.025) or (
                (model_loss >= 0.97 * prev_prev_loss)
                and (model_loss <= 1.005 * prev_prev_loss)
            )
            prev_prev_loss = prev_loss
            prev_loss = model_loss

        evaluate_model(model, mask, train_loader, test_loader, args)
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"fs_{k}.pth"))


def parse_args():
    parser = argparse.ArgumentParser(description="DiET Distillation")
    parser.add_argument("--dataset", choices=["mnist", "xray", "celeba"], required=True)
    parser.add_argument("--data-dir", required=True, help="Path to dataset")
    parser.add_argument("--model-path", required=True, help="Path to baseline model")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--lr", type=float, default=2000, help="Mask learning rate")
    parser.add_argument(
        "--model-lr", type=float, default=0.0001, help="Model learning rate"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ups", type=int, default=4, help="Upsample factor")
    parser.add_argument("--rounding-steps", type=int, default=5)
    parser.add_argument("--noise-class", default="NORMAL", help="Noise class for X-ray")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.im_size = 224

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"DiET Distillation: {args.dataset}")
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
    elif args.dataset == "celeba":
        train_imgs, train_labels, test_imgs, test_labels = load_celeba_data(
            args.data_dir
        )
        num_classes = 3
        from_disk = True

    model = resnet34(weights=None)
    model.fc = nn.Linear(512, num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.to(args.device)
    model.eval()

    print("Getting predictions...")
    train_preds = get_predictions(
        model, train_imgs, train_labels, num_classes, args.device, from_disk
    )
    test_preds = get_predictions(
        model, test_imgs, test_labels, num_classes, args.device, from_disk
    )

    train_loader = torch.utils.data.DataLoader(
        DatasetWithPreds(train_imgs, train_labels, train_preds, from_disk),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        DatasetWithPreds(test_imgs, test_labels, test_preds, from_disk),
        batch_size=8 * args.batch_size,
        shuffle=False,
    )

    mask = torch.ones(
        (len(train_preds), 1, args.im_size // args.ups, args.im_size // args.ups)
    )
    mask = mask.requires_grad_(True)
    mask_opt = torch.optim.SGD([mask], lr=args.lr)
    model_opt = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    distill(mask, model, train_loader, test_loader, mask_opt, model_opt, args)


if __name__ == "__main__":
    main()
