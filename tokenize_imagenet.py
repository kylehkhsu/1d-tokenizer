import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import demo_util
from modeling.titok import TiTok
import pprint
import scipy.io
import ipdb

# Constants and paths
dataset_dir = Path("/svl/data/ILSVRC2012")
train_dir = dataset_dir / "train"
val_dir = dataset_dir / "val"
val_labels_path = dataset_dir / "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
train_synsets_path = dataset_dir / "ILSVRC2012_devkit_t12/data/meta.mat"


def aspect_ratio_preserving_resize_crop(image, size=256, crop_size=256):
    """Resize while preserving aspect ratio and then perform a center crop."""
    width, height = image.size
    if height < width:
        new_height = size
        new_width = int(size * width / height)
    else:
        new_width = size
        new_height = int(size * height / width)
    image = image.resize((new_width, new_height), Image.BILINEAR)

    left = (new_width - crop_size) // 2
    top = (new_height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))

class ImageNetDataset(Dataset):
    """Parent class for ImageNet datasets."""
    def __init__(self, root_dir, labels=None):
        self.root_dir = root_dir
        self.labels = labels if labels is not None else []
        self.image_paths = []
        self.load_data()

    def load_data(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            with Image.open(img_path) as image:
                image = image.convert("RGB")
                image = aspect_ratio_preserving_resize_crop(image)
                image = (
                    torch.from_numpy(np.array(image).astype(np.float32))
                    .permute(2, 0, 1)
                    / 255.0
                )
            return img_path.name, image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None, None


class ImageNetTrainSplit(ImageNetDataset):
    """Dataset for ImageNet train split with nested directory structure."""
    def load_data(self):
        synsets = scipy.io.loadmat(str(train_synsets_path))["synsets"]
        synset_to_label = {}
        for i, synset in enumerate(synsets):
            synset_to_label[synset[0][1][0].item()] = synset["ILSVRC2012_ID"][0][0].item() - 1

        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob("*.JPEG"):
                    self.image_paths.append(img_path)
                    self.labels.append(synset_to_label[class_dir.name])


class ImageNetValSplit(ImageNetDataset):
    """Dataset for ImageNet validation split."""
    def load_data(self):
        self.image_paths = list(self.root_dir.glob("*.JPEG"))
        self.image_paths.sort()
        with open(val_labels_path, "r") as f:
            self.labels = [int(line.strip()) - 1 for line in f.readlines()]

def collate_fn(batch):
    """Custom collate function to filter out failed loads."""
    batch = [b for b in batch if b[1] is not None]
    if len(batch) == 0:
        return None, None, None
    filenames, images, labels = zip(*batch)
    return filenames, torch.stack(images), torch.tensor(labels)

def process_split(tokenizer, dataset_class, split_dir, output_file_prefix, batch_size=64, num_workers=8, device="cuda"):
    """Process a dataset split and save tokens, labels, and file paths."""
    dataset = dataset_class(split_dir)
    print(f"Found {len(dataset)} images in {split_dir.name}")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False
    )

    all_tokens = []
    all_labels = []
    all_paths = []
    for filenames, images, labels in tqdm(dataloader, desc=f"Processing {split_dir.name}"):
        if images is None:
            continue
        images = images.to(device)
        with torch.no_grad():
            encoded_tokens = tokenizer.encode(images)[1]["min_encoding_indices"]
            encoded_tokens = encoded_tokens.squeeze(1).cpu().numpy()
        all_tokens.append(encoded_tokens)
        all_labels.append(labels.numpy())
        all_paths.extend(filenames)

    # Save all data for this split in a single npz file
    all_tokens = np.concatenate(all_tokens, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    np.savez_compressed(
        f"{output_file_prefix}.npz",
        tokens=all_tokens,
        labels=all_labels,
        paths=np.array(all_paths)
    )
    print(f"Saved data for {split_dir.name} to {output_file_prefix}.npz")

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    output_dir = Path("/svl/u/kylehsu/output/1d-tokenizer/ILSVRC2012/titok_l32")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)

    # Load tokenizer
    cache_dir = "/svl/u/kylehsu/.cache/huggingface"
    config = demo_util.get_config("configs/infer/titok_l32.yaml")
    pprint.pprint(config)

    titok_tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet", cache_dir=cache_dir)
    titok_tokenizer.eval()
    titok_tokenizer.requires_grad_(False)
    titok_tokenizer = titok_tokenizer.to(device)
    # Process training and validation datasets
    process_split(titok_tokenizer, ImageNetTrainSplit, train_dir, output_dir / "train", batch_size=256, num_workers=8, device=device)
    # process_split(titok_tokenizer, ImageNetValSplit, val_dir, output_dir / "val", batch_size=256, num_workers=8, device=device)
