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
from huggingface_hub import hf_hub_download
from utils.train_utils import create_pretrained_tokenizer
from tokenize_imagenet import aspect_ratio_preserving_resize_crop, ImageNetTrainSplit, ImageNetValSplit, collate_fn, train_dir, val_dir

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
            encoded_tokens = tokenizer.encode(images)
            encoded_tokens = encoded_tokens.cpu().numpy()
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

    # Constants and paths
    output_dir = Path("/svl/u/kylehsu/output/1d-tokenizer/ILSVRC2012/maskgit-vqgan")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)

    rar_model_size = ["rar_b", "rar_l", "rar_xl", "rar_xxl"][3]

    # download the maskgit-vq tokenizer
    hf_hub_download(repo_id="fun-research/TiTok", filename=f"maskgit-vqgan-imagenet-f16-256.bin", local_dir="./")
    # download the rar generator weight
    # hf_hub_download(repo_id="yucornetto/RAR", filename=f"{rar_model_size}.bin", local_dir="./")

    # load config
    config = demo_util.get_config("configs/training/generator/rar.yaml")
    config.experiment.generator_checkpoint = f"{rar_model_size}.bin"
    config.model.generator.hidden_size = {"rar_b": 768, "rar_l": 1024, "rar_xl": 1280, "rar_xxl": 1408}[rar_model_size]
    config.model.generator.num_hidden_layers = {"rar_b": 24, "rar_l": 24, "rar_xl": 32, "rar_xxl": 40}[rar_model_size]
    config.model.generator.num_attention_heads = 16
    config.model.generator.intermediate_size = {"rar_b": 3072, "rar_l": 4096, "rar_xl": 5120, "rar_xxl": 6144}[
        rar_model_size]

    # Load tokenizer
    cache_dir = "/svl/u/kylehsu/.cache/huggingface"
    tokenizer = create_pretrained_tokenizer(config)
    tokenizer.to(device)
    pprint.pprint(config)

    # Process training and validation datasets
    process_split(tokenizer, ImageNetTrainSplit, train_dir, output_dir / "train", batch_size=256, num_workers=16, device=device)
    # process_split(tokenizer, ImageNetValSplit, val_dir, output_dir / "val", batch_size=32, num_workers=8, device=device)
