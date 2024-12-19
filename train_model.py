import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoConfig
import wandb
import ipdb
import scipy
from pathlib import Path


# Paths
train_npz_path = "/svl/u/kylehsu/output/1d-tokenizer/ILSVRC2012/titok_l32/train.npz"
val_npz_path = "/svl/u/kylehsu/output/1d-tokenizer/ILSVRC2012/titok_l32/val.npz"

dataset_dir = Path("/svl/data/ILSVRC2012")
train_synsets_path = dataset_dir / "ILSVRC2012_devkit_t12/data/meta.mat"

# Hyperparameters
vocab_size = 4096
sequence_length = 32
batch_size = 1024
# batch_size = 128
max_epochs = 300
learning_rate = 2e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

train = np.load(train_npz_path)
val = np.load(val_npz_path)

# Dataset
class TokenDataset(Dataset):
    """Dataset for causal autoregressive modeling."""
    def __init__(self, tokens, labels, paths):
        self.tokens = tokens
        self.labels = labels
        self.paths = paths
        assert len(self.tokens) == len(self.labels) == len(self.paths)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        sequence = self.tokens[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)  # Sequence excluding the last token
        labels = torch.tensor(sequence[1:], dtype=torch.long)     # Sequence shifted by one
        return {"input_ids": input_ids, "labels": labels}


class ClassFilteredTokenDataset(Dataset):
    def __init__(self, dataset, class_ids):
        self.dataset = dataset
        self.class_ids = class_ids
        self.filtered_indices = [i for i, _ in enumerate(self.dataset.tokens) if self.dataset.labels[i] in self.class_ids]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        return self.dataset[self.filtered_indices[idx]]

train_dataset_all = TokenDataset(train["tokens"], train["labels"], train["paths"])
val_dataset_all = TokenDataset(val["tokens"], val["labels"], val["paths"])

class_ids = list(range(100))
synsets = scipy.io.loadmat(str(train_synsets_path))["synsets"]
class_id_to_description = {synset[0][0].item() - 1: synset[0][2].item() for synset in synsets}
print(f"{len(class_ids)} classes: {'; '.join([class_id_to_description[class_id] for class_id in class_ids])}")

train_dataset = ClassFilteredTokenDataset(train_dataset_all, class_ids)
val_dataset = ClassFilteredTokenDataset(val_dataset_all, class_ids)

print(f"train: {len(train_dataset)}, val: {len(val_dataset)}")

# Prepare dataset
# debug
# num_samples = 2**17
# train_dataset = TokenDataset(train_tokens[:num_samples])
# val_dataset = TokenDataset(val_tokens[:num_samples // 24])

# Transformer model configuration
config = AutoConfig.from_pretrained(
    "gpt2",  # Use GPT-2 architecture
    vocab_size=vocab_size,
    n_positions=sequence_length,
    n_ctx=sequence_length,
    n_embd=512,  # Embedding size (can be adjusted)
    n_layer=24,   # Number of transformer layers
    n_head=16,    # Number of attention heads
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)

# Model
model = AutoModelForCausalLM.from_config(config)


# Initialize wandb before creating the Trainer
wandb.init(
    entity="iris_viscam",
    project="new_enc",  # Set your project name
    name="titok_l32_ac_subset",  # Optional, name for this run
    config={  # Log hyperparameters and config
        "vocab_size": vocab_size,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "class_ids": class_ids,
    },
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./titok_l32_ac",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    max_grad_norm=1.0,
    weight_decay=0.01,
    num_train_epochs=max_epochs,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    report_to="wandb",
    dataloader_num_workers=0,
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("./titok_l32_ac")
print("Training complete and model saved.")
