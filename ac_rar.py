from typing import Iterator, Optional, Tuple, Union

import argparse

import ipdb
import numpy as np
import torch
import tqdm

from pathlib import Path

from gptzip.utils import bits_to_bytes, bytes_to_bits, normalize_pdf_for_arithmetic_coding
from gptzip.helpers import Encoder, Decoder
from gptzip import ArithmeticCoder

from huggingface_hub import hf_hub_download
from utils.train_utils import create_pretrained_tokenizer
import demo_util


class ArithmeticCoderRAR:
    # Helpful links:
    #   > https://github.com/google-deepmind/language_modeling_is_compression
    #   > https://www.cs.cmu.edu/~aarti/Class/10704/Intro_Arith_coding.pdf
    #   > https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html
    #   > https://www.cs.ucf.edu/courses/cap5015/Arithmetic%20Coding%20note%202004.pdf
    # Base 2 means that the coder writes bits.
    ARITHMETIC_CODER_BASE = 2
    # Precision 32 implies 32 bit arithmetic.
    ARITHMETIC_CODER_PRECISION = 32

    def __init__(self, rar):
        self.rar = rar

    @property
    def _rar_device(self) -> torch.device:
        return next(self.rar.parameters()).device


    def encode(
        self,
        data: np.ndarray,
        return_num_padded_bits: bool = False,
    ) -> Union[bytes, tuple[bytes, int]]:
        """Compresses the `data` using arithmetic coding and a pretrained model.

        Args:
            data: The data to be compressed.
            return_num_padded_bits: Whether to return the number of zeros added to the
                encoded bitstream in order to make it byte-decodeable (i.e., divisible by
                8). Usually, this is used when the encoded data has to be decoded again.

        Returns:
            The compressed data.
        """
        sequence_array = torch.tensor(data, dtype=torch.int32).unsqueeze(0).to(self._rar_device)
        condition = torch.tensor([self.rar.none_condition_id]).to(self._rar_device)
        return_labels = False
        orders = None
        is_sampling = False
        logits = self.rar.forward_fn(sequence_array, condition, return_labels, orders, is_sampling)

        if False:   # random pdf baseline
            logits = torch.from_numpy(np.random.randn(1, 257, 1024)).to(self._rar_device)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs[0, :-1, :].cpu().numpy()

        output = list()
        encoder = Encoder(
            base=ArithmeticCoder.ARITHMETIC_CODER_BASE,
            precision=ArithmeticCoder.ARITHMETIC_CODER_PRECISION,
            output_fn=output.append,
        )
        for pdf, symbol in zip(probs, data):
            encoder.encode(normalize_pdf_for_arithmetic_coding(pdf), symbol)
        encoder.terminate()

        compressed_bits = ''.join(map(str, output))
        compressed_bytes, num_padded_bits = bits_to_bytes(compressed_bits)

        if return_num_padded_bits:
            return compressed_bytes, num_padded_bits
        else:
            return compressed_bytes

    def encode_batched(
        self,
        data: np.ndarray,
        return_num_padded_bits: bool = False,
        batch_size: int = 32
    ) -> Union[bytes, Tuple[bytes, int]]:
        """Compresses the `data` using arithmetic coding and a pretrained model in batches.

        Args:
            data: The data to be compressed. Should be a 2D numpy array where each row is a sequence.
            return_num_padded_bits: Whether to return the number of zeros added to the
                encoded bitstream in order to make it byte-decodeable (i.e., divisible by 8).
            batch_size: The number of sequences to process in one batch.

        Returns:
            The compressed data as bytes. Optionally, the number of padded bits.
        """
        all_compressed_bytes = []
        all_num_padded_bits = []

        num_sequences = data.shape[0]
        sequence_length = data.shape[1]
        num_batches = (num_sequences + batch_size - 1) // batch_size

        pbar = tqdm.tqdm(total=num_sequences)
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, num_sequences)
            batch_data = data[batch_start:batch_end]

            # Prepare the batch for input to the model
            batch_tensor = torch.tensor(batch_data, dtype=torch.int32).to(self._rar_device)
            condition = torch.tensor([self.rar.none_condition_id]).to(self._rar_device).expand(batch_tensor.shape[0])
            return_labels = False
            orders = None
            is_sampling = False

            # Forward pass through the RAR model
            logits = self.rar.forward_fn(batch_tensor, condition, return_labels, orders, is_sampling)

            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

            # Encode each sequence in the batch
            for seq_idx in range(batch_tensor.shape[0]):
                output = []
                encoder = Encoder(
                    base=ArithmeticCoder.ARITHMETIC_CODER_BASE,
                    precision=ArithmeticCoder.ARITHMETIC_CODER_PRECISION,
                    output_fn=output.append,
                )

                sequence_probs = probs[seq_idx]
                sequence_data = batch_data[seq_idx]

                for pdf, symbol in zip(sequence_probs[:-1], sequence_data):
                    encoder.encode(normalize_pdf_for_arithmetic_coding(pdf), symbol)

                encoder.terminate()

                compressed_bits = ''.join(map(str, output))
                compressed_bytes, num_padded_bits = bits_to_bytes(compressed_bits)

                all_compressed_bytes.append(compressed_bytes)
                all_num_padded_bits.append(num_padded_bits)
            pbar.update(batch_tensor.shape[0])

        # Combine all compressed bytes and return
        if return_num_padded_bits:
            return all_compressed_bytes, all_num_padded_bits
        else:
            return all_compressed_bytes

    def decode(
        self,
        data: bytes,
        num_padded_bits: int = 0,
        skip_special_tokens: bool = True,
    ) -> bytes:
        """Decompresses the `data` using arithmetic coding and a pretrained model.

        See https://en.wikipedia.org/wiki/Arithmetic_coding for details.

        Args:
            data: The data to be decompressed.
            num_padded_bits: The number of zeros added to the encoded bitstream in order
            to make it byte-decodeable (i.e., divisble by 8).
            skip_special_tokens: Whether to filter out e.g. <eos> in tokens-to-string
                conversion.

        Returns:
            The decompressed data.
        """
        data_iter = iter(bytes_to_bits(data, num_padded_bits=num_padded_bits))

        # The decoder requires a function that reads digits from {0, 1, ..., base - 1}
        # from the compressed input and returns `None` when the input is exhausted.
        def _input_fn(bit_sequence: Iterator[str] = data_iter) -> Optional[int]:
            try:
                return int(next(bit_sequence))
            except StopIteration:
                return None

        decoder = Decoder(
            base=ArithmeticCoder.ARITHMETIC_CODER_BASE,
            precision=ArithmeticCoder.ARITHMETIC_CODER_PRECISION,
            input_fn=_input_fn,
        )
        # We need a dummy token because the language model right-shifts the sequence
        # by onde when computing the conditional probabilities. Concretely, at every
        # step, we need the `pdf` of the next token given all currently decompressed
        # tokens, but without a dummy token, the last `pdf` would be that of the last
        # already decompressed token. The value of the dummy token is irrelevant.
        sequence_array = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.int32)
        # print("3 >> sequence_array.shape", sequence_array.shape)
        probs, past_key_values = self._next_token_probs(
            input_ids=sequence_array[None],
            past_key_values=None
        )
        probs = probs[0, 0]

        idx = 0
        while True:
            # print("idx", idx, "probs.shape", probs.shape, "/ argmax", probs.argmax().item(), "sequence_arr", sequence_array)
            try:
                token = decoder.decode(
                    normalize_pdf_for_arithmetic_coding(probs)
                )
            except StopIteration:
                break
            # print("\t token:", token)
            sequence_array = torch.tensor(
                np.append(sequence_array, token)
                , dtype=torch.int32
            )
            probs, past_key_values = self._next_token_probs(sequence_array[None], past_key_values=past_key_values)
            probs = probs[0, -1]
            idx += 1

        # Remove the dummy token and convert to bytes.
        print(f"Decoded {len(sequence_array)} tokens:", sequence_array)
        return self.tokenizer.decode(sequence_array, skip_special_tokens=skip_special_tokens)


if __name__ == "__main__":
    train_path = Path("output/ILSVRC2012/maskgit-vqgan/train.npz")
    val_path = Path("output/ILSVRC2012/maskgit-vqgan/val.npz")
    train_data = np.load(train_path)
    val_data = np.load(val_path)

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
    config.model.generator.intermediate_size = {"rar_b": 3072, "rar_l": 4096, "rar_xl": 5120, "rar_xxl": 6144}[
        rar_model_size]

    device = "cuda"
    rar = demo_util.get_rar_generator(config)
    rar.to(device)

    coder = ArithmeticCoderRAR(rar=rar)
    codes = []
    train_tokens = train_data["tokens"]
    val_tokens = val_data["tokens"]

    raw_vocab = len(np.unique(train_tokens))

    def compress(tokens):
        num_samples = tokens.shape[0]
        for i, token in tqdm.tqdm(enumerate(tokens)):
            codes.append(coder.encode(token))
        encoded_bits = sum(len(code) * 8 for code in codes)
        raw_sequence_length = tokens.shape[1]
        raw_bits = num_samples * raw_sequence_length * np.log2(raw_vocab)
        compression_ratio = encoded_bits / raw_bits
        print(f"Compression ratio: {compression_ratio:.2f}")

    def compress_batched(tokens):
        num_samples = tokens.shape[0]
        codes = coder.encode_batched(tokens, batch_size=64)
        encoded_bits = sum(len(code) * 8 for code in codes)
        raw_sequence_length = tokens.shape[1]
        raw_bits = num_samples * raw_sequence_length * np.log2(raw_vocab)
        compression_ratio = encoded_bits / raw_bits
        ipdb.set_trace()
        print(f"Compression ratio: {compression_ratio:.2f}")

    with torch.no_grad():

        compress_batched(val_tokens[:64])
