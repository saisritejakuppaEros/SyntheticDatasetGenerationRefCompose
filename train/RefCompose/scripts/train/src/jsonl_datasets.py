from pathlib import Path

import os
from PIL import Image
from torchvision import transforms
import random
import torch

Image.MAX_IMAGE_PIXELS = None


def resolve_image_path(image_path: str, root_dir: str | None) -> str:
    """If path is relative and root_dir is set, join root_dir / image_path."""
    if not image_path or not str(image_path).strip():
        return image_path
    if not root_dir or not str(root_dir).strip():
        return image_path
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    return str(Path(root_dir).expanduser() / p)


def multiple_16(num: float):
    return int(round(num / 16) * 16)


def get_random_resolution(min_size=512, max_size=1280, multiple=16):
    resolution = random.randint(min_size // multiple, max_size // multiple) * multiple
    return resolution


def resolve_image_path(image_path: str, root_dir: str | None) -> str:
    """If path is relative and root_dir is set, join; absolute paths unchanged."""
    if not root_dir or not str(root_dir).strip():
        return image_path
    if os.path.isabs(image_path):
        return image_path
    return os.path.join(root_dir, image_path)


def load_image_safely(image_path, size, root_dir=None):
    path = resolve_image_path(image_path, root_dir)
    try:
        image = Image.open(path).convert("RGB")
        return image
    except Exception:
        print("file error: " + path)
        with open("failed_images.txt", "a") as f:
            f.write(f"{path}\n")
        return Image.new("RGB", (size, size), (255, 255, 255))


def _extract_prompts_batch(examples, caption_column, target_column):
    cols = [c.strip() for c in caption_column.split(",") if c.strip()]
    n = len(examples[target_column])
    captions = []
    for i in range(n):
        if len(cols) == 1:
            caption = examples[cols[0]][i]
        else:
            caption = [examples[c][i] for c in cols]
        if isinstance(caption, str):
            if random.random() < 0.1:
                captions.append(" ")
            else:
                captions.append(caption)
        elif isinstance(caption, list):
            if random.random() < 0.1:
                captions.append(" ")
            else:
                captions.append(random.choice(caption))
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    return captions


def make_train_dataset(args, accelerator=None, tokenizer=None):
    from datasets import load_dataset

    if args.train_data_dir is not None:
        print("load_data")
        dataset = load_dataset("json", data_files=args.train_data_dir)

    caption_column = args.caption_column
    target_column = args.target_column
    if args.subject_column is not None:
        subject_columns = args.subject_column.split(",")
    if args.spatial_column is not None:
        spatial_columns = args.spatial_column.split(",")

    size = args.cond_size
    noise_size = get_random_resolution(max_size=args.noise_size)
    subject_cond_train_transforms = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: img.resize(
                    (
                        multiple_16(size * img.size[0] / max(img.size)),
                        multiple_16(size * img.size[1] / max(img.size)),
                    ),
                    resample=Image.BILINEAR,
                )
            ),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomRotation(degrees=20),
            transforms.Lambda(
                lambda img: transforms.Pad(
                    padding=(
                        int((size - img.size[0]) / 2),
                        int((size - img.size[1]) / 2),
                        int((size - img.size[0]) / 2),
                        int((size - img.size[1]) / 2),
                    ),
                    fill=0,
                )(img)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    cond_train_transforms = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def train_transforms(image, noise_size):
        train_transforms_ = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: img.resize(
                        (
                            multiple_16(noise_size * img.size[0] / max(img.size)),
                            multiple_16(noise_size * img.size[1] / max(img.size)),
                        ),
                        resample=Image.BILINEAR,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        return train_transforms_(image)

    def load_and_transform_cond_images(images):
        transformed_images = [cond_train_transforms(image) for image in images]
        return torch.cat(transformed_images, dim=1)

    def load_and_transform_subject_images(images):
        transformed_images = [subject_cond_train_transforms(image) for image in images]
        return torch.cat(transformed_images, dim=1)

    def preprocess_train(examples):
        _examples = {}
        if args.subject_column is not None:
            subject_images = [
                [load_image_safely(examples[column][i], args.cond_size) for column in subject_columns]
                for i in range(len(examples[target_column]))
            ]
            _examples["subject_pixel_values"] = [load_and_transform_subject_images(subject) for subject in subject_images]
        if args.spatial_column is not None:
            spatial_images = [
                [load_image_safely(examples[column][i], args.cond_size) for column in spatial_columns]
                for i in range(len(examples[target_column]))
            ]
            _examples["cond_pixel_values"] = [load_and_transform_cond_images(spatial) for spatial in spatial_images]
        target_images = [load_image_safely(image_path, args.cond_size) for image_path in examples[target_column]]
        _examples["pixel_values"] = [train_transforms(image, noise_size) for image in target_images]
        _examples["prompts"] = _extract_prompts_batch(examples, caption_column, target_column)
        return _examples

    if accelerator is not None:
        with accelerator.main_process_first():
            train_dataset = dataset["train"].with_transform(preprocess_train)
    else:
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    if examples[0].get("cond_pixel_values") is not None:
        cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        cond_pixel_values = None
    if examples[0].get("subject_pixel_values") is not None:
        subject_pixel_values = torch.stack([example["subject_pixel_values"] for example in examples])
        subject_pixel_values = subject_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        subject_pixel_values = None

    target_pixel_values = torch.stack([example["pixel_values"] for example in examples])
    target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()
    prompts = [example["prompts"] for example in examples]

    return {
        "cond_pixel_values": cond_pixel_values,
        "subject_pixel_values": subject_pixel_values,
        "pixel_values": target_pixel_values,
        "prompts": prompts,
    }
