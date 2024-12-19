import ipdb
import torch
from PIL import Image
import numpy as np
import demo_util
from huggingface_hub import hf_hub_download
from utils.train_utils import create_pretrained_tokenizer
from pathlib import Path

# Choose one from ["rar_b_imagenet", "rar_l_imagenet", "rar_xl_imagenet", "rar_xxl_imagenet"]
rar_model_size = ["rar_b", "rar_l", "rar_xl", "rar_xxl"][3]

# download the maskgit-vq tokenizer
hf_hub_download(repo_id="fun-research/TiTok", filename=f"maskgit-vqgan-imagenet-f16-256.bin", local_dir="./")
# download the rar generator weight
hf_hub_download(repo_id="yucornetto/RAR", filename=f"{rar_model_size}.bin", local_dir="./")

# load config
config = demo_util.get_config("configs/training/generator/rar.yaml")
config.experiment.generator_checkpoint = f"{rar_model_size}.bin"
config.model.generator.hidden_size = {"rar_b": 768, "rar_l": 1024, "rar_xl": 1280, "rar_xxl": 1408}[rar_model_size]
config.model.generator.num_hidden_layers = {"rar_b": 24, "rar_l": 24, "rar_xl": 32, "rar_xxl": 40}[rar_model_size]
config.model.generator.num_attention_heads = 16
config.model.generator.intermediate_size = {"rar_b": 3072, "rar_l": 4096, "rar_xl": 5120, "rar_xxl": 6144}[rar_model_size]


device = "cuda"
# maskgit-vq as tokenizer
tokenizer = create_pretrained_tokenizer(config)
tokenizer.to(device)

# tokenize image
# image_path = Path("/svl/data/ILSVRC2012/val/ILSVRC2012_val_00019999.JPEG")
# # image_path = Path("assets/ILSVRC2012_val_00008636.png")
# img = Image.open(image_path)
# img = img.resize((256, 256))
# img = np.array(img).astype(np.float32) / 255.0
# img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
# img = img.to(device)
# tokens = tokenizer.encode(img)

val = np.load("output/ILSVRC2012/maskgit-vqgan/val.npz")
i = 24659
tokens = torch.from_numpy(val["tokens"][i]).to(device).unsqueeze(0)
image_path = Path(val["paths"][i])
reconstructed_img = tokenizer.decode(tokens)
reconstructed_img = torch.clamp(reconstructed_img, 0.0, 1.0)
reconstructed_img = (reconstructed_img * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
reconstructed_img = Image.fromarray(reconstructed_img)
reconstructed_img.save(f"output/sandbox/rar_{image_path.stem}_reconstructed.png")
ipdb.set_trace()


# generator = demo_util.get_rar_generator(config)
# generator.to(device)
#
# # generate an image
# sample_labels = [torch.randint(0, 999, size=(1,)).item()] # random IN-1k class
# generated_image = demo_util.sample_fn(
#     generator=generator,
#     tokenizer=tokenizer,
#     labels=sample_labels,
#     randomize_temperature=1.0,
#     guidance_scale=4.0,
#     guidance_scale_pow=0.0, # constant cfg
#     device=device
# )
# Image.fromarray(generated_image[0]).save(f"output/sandbox/rar_generated_{sample_labels[0]}.png")