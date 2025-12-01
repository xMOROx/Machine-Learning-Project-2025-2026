import os
import sys
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Hard MNIST dataset from Colorized MNIST"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Path to Colorized MNIST directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Path to output Hard MNIST directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def construct_data(input_pattern, output_dir):
    small_resize = transforms.Resize(56)
    resize = transforms.Resize(112)
    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()

    files = glob.glob(input_pattern)
    png_files = [f for f in files if f.endswith(".png")]

    print(f"Processing {len(png_files)} images...")

    for i, f in enumerate(png_files):
        label = int(os.path.basename(os.path.dirname(f)))
        image = Image.open(f)
        t = to_tensor(resize(image))

        new_tensor = torch.ones((3, 224, 224)).to(t.dtype) * t[:, 0:1, 0:1]

        num_lines = torch.randint(low=0, high=3, size=(2,))
        for j in range(num_lines[0].item()):
            width = random.randint(0, 50)
            if width > 0:
                ind = random.randint(0, 223 - width)
                new_tensor[:, ind : ind + width, :] = torch.rand(3, width, 224)
        for j in range(num_lines[1].item()):
            width = random.randint(0, 50)
            if width > 0:
                ind = random.randint(0, 223 - width)
                new_tensor[:, :, ind : ind + width] = torch.rand(3, 224, width)

        f2 = random.choice(png_files)
        image2 = Image.open(f2)
        t2 = to_tensor(small_resize(image2))
        ind2 = torch.randint(low=0, high=164, size=(2,))
        small_digit_background = torch.where(torch.sum(t2, 0) <= 2, 0, 1)
        t2 = t2 * small_digit_background
        new_tensor[:, ind2[0] : ind2[0] + 56, ind2[1] : ind2[1] + 56] += 0.98 * t2

        ind = torch.randint(low=0, high=112, size=(2,))
        new_tensor[:, ind[0] : ind[0] + 112, ind[1] : ind[1] + 112] = t
        new_tensor = new_tensor.clamp(max=1, min=0)

        new_image = to_image(new_tensor)
        out_file = os.path.join(output_dir, str(label), f"img_{i}.png")
        new_image.save(out_file)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(png_files)} images...")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    for split in ["training", "testing"]:
        input_pattern = os.path.join(args.input_dir, split, "*", "*.png")
        output_split_dir = os.path.join(args.output_dir, split)

        for digit in range(10):
            os.makedirs(os.path.join(output_split_dir, str(digit)), exist_ok=True)

        print(f"Processing {split} data...")
        construct_data(input_pattern, output_split_dir)

    print("Hard MNIST preparation complete!")


if __name__ == "__main__":
    main()
