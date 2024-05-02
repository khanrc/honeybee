import re

import torch
from transformers import AutoTokenizer

from pipeline.data_utils.constants import MEDIA_TOKENS, IGNORE_INDEX, HUMAN, AI
from utils.logging import get_logger

# Role tokens
_AI = "\n" + AI
_HUMAN = "\n" + HUMAN

_INFINITE = int(1e12)  # infinite token length for no-truncation

logger = get_logger()


def _pad_trunc(
    x: list[list[int]],
    padding: str,
    padding_side: str,
    pad_value: int,
    max_length: int,
) -> torch.LongTensor:
    """Pad and truncate sequences to the same length

    Args:
        x (list[list[int]])
        padding ("longest" or "max_length")
        padding_side ("left" or "right")
        pad_value (int)
        max_length (int or None): if padding == "max_length", max_length should be given.
    """
    assert padding in ["longest", "max_length"]
    assert padding_side in ["left", "right"]

    lengths = [len(sample) for sample in x]
    if padding == "longest":
        max_length = max(lengths)

    new_x = []
    for sample, length in zip(x, lengths):
        if torch.is_tensor(sample):
            sample = sample.tolist()

        if length >= max_length:
            new_x.append(sample[:max_length])
            continue

        padding_size = max_length - length
        pads = [pad_value] * padding_size
        if padding_side == "right":
            new_x.append(sample + pads)
        else:
            new_x.append(pads + sample)

    return torch.as_tensor(new_x, dtype=torch.long)


class HoneybeeTokenizerMixin:
    def mllm_setup(self, num_visual_tokens: int):
        if self.pad_token is None:
            logger.warning(f"Tokenizer {self.__class__} has no pad_token. Use unk_token instead.")
            self.pad_token = self.unk_token

        self.num_visual_tokens = num_visual_tokens

        # Currently we only support the image modality for media modality.
        self.media_tokens = {k: -int(i + 1) for i, k in enumerate(MEDIA_TOKENS["image"])}
        self.media_lengths = {MEDIA_TOKENS["image"][0]: num_visual_tokens}  # token lengths

    def encode_prompt(self, prompt: str, max_length: int | None, no_eos=False):
        """Tokenize prompt which consists of image-text or text only, with role tokens.
        Role pattern is "AI: " or "Human: ".

        Args:
            prompt
            max_length (int or None): here, max_length is used for truncation.
                If max_length is None, no truncation is applied.
            no_eos: if True, eos token is not added at the end of the prompt.
                Note that eos token is still used for end-of-AI-turn token even no_eos=True.
        """
        max_length = max_length or _INFINITE  # if None, set to infinite for no-truncation

        # output enc_chunk
        enc_chunk = [self.bos_token_id]
        label_chunk = [0]

        if prompt.startswith("[NO_PROMPT] <image> "):
            # Special case: [NO_PROMPT]
            # Here, we prepare data without any prompt.
            # note that prompt is assumed to be [NO_PROMPT] <image> <text>
            # it will be converted to <bos> <image> <text> <eos>

            # bos token
            enc_chunk = [self.bos_token_id]
            label_chunk = [0]

            # prepare whole text data
            # <bos> <image> <text> <eos>
            txt = prompt.replace("[NO_PROMPT] <image> ", "")
            text_chunk = self(txt, add_special_tokens=False)["input_ids"]
            enc_chunk += (
                [self.media_tokens["<image>"]] * self.media_lengths["<image>"]  # media tokens
                + text_chunk  # text tokens
                + [self.eos_token_id]  # eos token
            )
            label_chunk += (
                [0] * self.media_lengths["<image>"]  # labels for media tokens
                + [1] * len(text_chunk)  # labels for text tokens
                + [1]  # labels for eos token
            )

        else:
            # Text-only or Image-Text Data
            # split prompt into chunks by media and role tokens
            pattern = "|".join(map(re.escape, list(self.media_tokens.keys()) + [_AI, _HUMAN]))
            chunk_strs = re.split(f"({pattern})", prompt)
            chunk_strs = [x for x in chunk_strs if len(x) > 0]
            for idx, chunk_str in enumerate(chunk_strs):
                if len(enc_chunk) >= max_length + 1:
                    break

                if chunk_str in self.media_tokens:
                    if len(enc_chunk) + self.media_lengths[chunk_str] > max_length + 1:
                        break

                    enc_chunk += [self.media_tokens[chunk_str]] * self.media_lengths[chunk_str]
                    label_chunk += [0] * self.media_lengths[chunk_str]

                else:
                    label = 1 if (idx > 0 and chunk_strs[idx - 1] == _AI) else 0
                    curr_chunk = self(chunk_str, add_special_tokens=False)["input_ids"]
                    if label == 1:
                        curr_chunk += [self.eos_token_id]
                    enc_chunk += curr_chunk
                    label_chunk += [label] * len(curr_chunk)

        if no_eos and enc_chunk[-1] == self.eos_token_id:
            # the last token can be != eos_token_id; when the prompt is ended with `AI: `.
            # in this case, there is no AI-answer, thus, no eos token is added.
            enc_chunk = enc_chunk[:-1]
            label_chunk = label_chunk[:-1]

        enc_chunk = enc_chunk[: max_length + 1]
        label_chunk = label_chunk[: max_length + 1]
        L = len(enc_chunk)
        assert L == len(label_chunk)

        input_ids = torch.as_tensor(enc_chunk, dtype=torch.long)
        loss_mask = torch.as_tensor(label_chunk, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Label
        labels = input_ids.clone()
        labels[loss_mask != 1] = IGNORE_INDEX

        # The length of input_ids (L) includes <bos> and <eos> tokens.
        # Since max_length does not include <bos> token, L <= max_length + 1
        assert L <= max_length + 1

        return {
            "input_ids": input_ids,  # [L]
            "labels": labels,  # [L]
            "seq_length": L,  # int
            "attention_mask": attention_mask,  # [L]
        }

    def batch_encode_prompt(
        self,
        prompts: list[str],
        padding: str = "longest",
        padding_side: str = "right",
        max_length: int | None = None,
        no_eos=False,
    ) -> dict[str, torch.LongTensor]:
        """Batch encode prompts, pad/truncate to the same length, and collate them.
        Args:
            prompts (list[str])
            padding ("longest" or "max_length")
            padding_side ("left" or "right")
            pad_value (int)
            max_length (int or None): if padding == "max_length", max_length should be given
        """
        batch = [self.encode_prompt(prompt, max_length, no_eos) for prompt in prompts]
        batch = self.batch_collate_pad(batch, padding, padding_side, max_length)

        return batch

    def batch_collate_pad(
        self,
        batch: list,
        padding: str,
        padding_side: str,
        max_length: int | None,
    ) -> dict[str, torch.LongTensor]:
        """Collate batch and pad/truncate to the same length

        Args:
            batch
            padding ("longest" or "max_length")
            padding_side ("left" or "right")
            pad_value (int)
            max_length (int or None): if padding == "max_length", max_length should be given
        """
        if padding == "max_length":
            assert max_length is not None, "max_length should be given if padding == 'max_length'"
        else:
            # if padding == 'longest' and max_length is None, set to infinite for no-truncation
            max_length = max_length or _INFINITE

        input_ids = [sample["input_ids"] for sample in batch]
        labels = [sample["labels"] for sample in batch]
        attention_mask = [sample["attention_mask"] for sample in batch]
        seq_length = [sample["seq_length"] for sample in batch]

        # max_length + 1 for bos_token
        input_ids = _pad_trunc(input_ids, padding, padding_side, self.pad_token_id, max_length+1)
        labels = _pad_trunc(labels, padding, padding_side, IGNORE_INDEX, max_length+1)
        attention_mask = _pad_trunc(attention_mask, padding, padding_side, 0, max_length+1)
        seq_length = torch.as_tensor(seq_length, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "seq_length": seq_length,
        }


#################################################################
# Tokenizer builder
#################################################################
def extend_instance_(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (base_cls, mixin), {})


def build_honeybee_tokenizer(pretrained_tokenizer_name_or_path: str, num_visual_tokens: int):
    """Build Honeybee tokenizer with monkey-patch
    """
    # If use_fast=True, the tokenizer is re-constructed causing long building time (about 5min)
    # Another solution is save-and-load re-constructed fast tokenizer, but we simply use
    # normal version here.
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name_or_path, use_fast=False)

    # monkey patch
    extend_instance_(tokenizer, HoneybeeTokenizerMixin)
    tokenizer.mllm_setup(num_visual_tokens)

    return tokenizer
