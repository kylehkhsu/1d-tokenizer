import demo_util
import numpy as np
import torch
from PIL import Image
import imagenet_classes
import os
from huggingface_hub import hf_hub_download
from modeling.maskgit import ImageBert
from modeling.titok import TiTok
import ipdb
import pprint

cache_dir = "/svl/u/kylehsu/.cache/huggingface"

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)

config = demo_util.get_config("configs/infer/titok_l32.yaml")
pprint.pprint(config)

titok_tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet", cache_dir=cache_dir)
titok_tokenizer.eval()
titok_tokenizer.requires_grad_(False)

device = "cuda"
titok_tokenizer = titok_tokenizer.to(device)

tokens_path = "/svl/u/kylehsu/output/1d-tokenizer/ILSVRC2012/titok_l32/train_tokens.npy"
def tokenize(img_path):
    original_image = Image.open(img_path)
    image = torch.from_numpy(np.array(original_image).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    encoded_tokens = titok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"]

    return encoded_tokens

def decode(tokens):
    image = titok_tokenizer.decode_tokens(tokens)
    image = torch.clamp(image, 0.0, 1.0)
    image = (image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    image = Image.fromarray(image)
    return image

def batch_decode(tokens):
    tokens = torch.from_numpy(tokens).to(device).unsqueeze(1)
    images = titok_tokenizer.decode_tokens(tokens)
    images = torch.clamp(images, 0.0, 1.0)
    images = (images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    images = [Image.fromarray(image) for image in images]
    return images


# tokens = np.load(tokens_path)
# images = batch_decode(tokens[:5])
# for i, image in enumerate(images):
#     image.save(f"./output/sandbox/ILSVRC2012_train_{i}_reconstructed.png")

# tokens = tokenize("assets/ILSVRC2012_val_00008636.png")
# image = decode(tokens)
# image.save("./output/sandbox/ILSVRC2012_val_00008636_reconstructed.png")
ipdb.set_trace()