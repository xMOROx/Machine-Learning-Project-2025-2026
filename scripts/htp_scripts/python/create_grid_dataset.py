import os
import argparse
import random
import cv2
import numpy as np


def create_grid_sample_2x2(ord_images, no_of_classes, base_dir):
    sel_images = []
    sel_classes = []
    attempts = 0
    max_attempts = 1000

    while len(sel_classes) < 4 and attempts < max_attempts:
        attempts += 1
        curr_class = random.randint(0, no_of_classes - 1)
        if (
            curr_class not in sel_classes
            and curr_class in ord_images
            and len(ord_images[curr_class]) > 0
        ):
            sel_classes.append(curr_class)
            curr_idx = random.randint(0, len(ord_images[curr_class]) - 1)
            img = cv2.imread(os.path.join(base_dir, ord_images[curr_class][curr_idx]))
            if img is not None:
                sel_images.append(img)
            else:
                sel_classes.pop()

    if len(sel_images) != 4:
        return None, None

    sel_classes_str = "_".join([str(x) for x in sel_classes])

    row1 = np.concatenate([sel_images[0], sel_images[1]], axis=1)
    row2 = np.concatenate([sel_images[2], sel_images[3]], axis=1)
    final_image = np.concatenate([row1, row2], axis=0)

    return final_image, sel_classes_str


def create_grid_sample_3x3(ord_images, no_of_classes, base_dir):
    sel_images = []
    sel_classes = []
    attempts = 0
    max_attempts = 1000

    while len(sel_classes) < 9 and attempts < max_attempts:
        attempts += 1
        curr_class = random.randint(0, no_of_classes - 1)
        if (
            curr_class not in sel_classes
            and curr_class in ord_images
            and len(ord_images[curr_class]) > 0
        ):
            sel_classes.append(curr_class)
            curr_idx = random.randint(0, len(ord_images[curr_class]) - 1)
            img = cv2.imread(os.path.join(base_dir, ord_images[curr_class][curr_idx]))
            if img is not None:
                sel_images.append(img)
            else:
                sel_classes.pop()

    if len(sel_images) != 9:
        return None, None

    sel_classes_str = "_".join([str(x) for x in sel_classes])

    row1 = np.concatenate([sel_images[0], sel_images[1], sel_images[2]], axis=1)
    row2 = np.concatenate([sel_images[3], sel_images[4], sel_images[5]], axis=1)
    row3 = np.concatenate([sel_images[6], sel_images[7], sel_images[8]], axis=1)
    final_image = np.concatenate([row1, row2, row3], axis=0)

    return final_image, sel_classes_str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create GridPG dataset from confident images"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory with confident images"
    )
    parser.add_argument(
        "--output-dir-2x2", required=True, help="Output directory for 2x2 grids"
    )
    parser.add_argument(
        "--output-dir-3x3", required=True, help="Output directory for 3x3 grids"
    )
    parser.add_argument(
        "--dataset-size", type=int, default=500, help="Number of grid images to create"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    os.makedirs(args.output_dir_2x2, exist_ok=True)
    os.makedirs(args.output_dir_3x3, exist_ok=True)

    raw_images = os.listdir(args.input_dir)
    raw_images = [x for x in raw_images if x.endswith((".png", ".jpg", ".jpeg"))]

    class_to_image_dict = {}
    number_of_classes = -1

    for image in raw_images:
        try:
            image_class = int(image.split("_")[1].split(".")[0])
            number_of_classes = max(number_of_classes, image_class)
            if image_class not in class_to_image_dict:
                class_to_image_dict[image_class] = []
            class_to_image_dict[image_class].append(image)
        except (IndexError, ValueError):
            continue

    print(f"Found {len(raw_images)} images across {len(class_to_image_dict)} classes")
    print(f"Max class index: {number_of_classes}")

    created_2x2 = 0
    created_3x3 = 0

    for idx in range(args.dataset_size):
        img_2x2, name_2x2 = create_grid_sample_2x2(
            class_to_image_dict, number_of_classes + 1, args.input_dir
        )
        if img_2x2 is not None:
            cv2.imwrite(
                os.path.join(args.output_dir_2x2, f"{idx}_{name_2x2}.png"), img_2x2
            )
            created_2x2 += 1

        img_3x3, name_3x3 = create_grid_sample_3x3(
            class_to_image_dict, number_of_classes + 1, args.input_dir
        )
        if img_3x3 is not None:
            cv2.imwrite(
                os.path.join(args.output_dir_3x3, f"{idx}_{name_3x3}.png"), img_3x3
            )
            created_3x3 += 1

        if (idx + 1) % 50 == 0:
            print(f"Created {idx + 1}/{args.dataset_size} grids")

    print(f"\nCreated {created_2x2} 2x2 grids in {args.output_dir_2x2}")
    print(f"Created {created_3x3} 3x3 grids in {args.output_dir_3x3}")


if __name__ == "__main__":
    main()
