import os

import torch
from peft import PeftConfig, PeftModel
from PIL import Image
from transformers import AutoTokenizer
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from pathlib import Path

from honeybee.modeling_honeybee import HoneybeeForConditionalGeneration
from honeybee.processing_honeybee import HoneybeeImageProcessor, HoneybeeProcessor


def load_model(pretrained_ckpt, use_bf16=True, load_in_8bit=False):
    """Model loader.

    Args:
        pretrained_ckpt (string): The path to pre-trained checkpoint.
        use_bf16 (bool, optional): Whether to use bfloat16 to load the model. (Default: True)
        load_in_8bit(bool, optional): Flag to load model in 8it. (Default: False)

    Returns:
        model: Honeybee Model
    """

    # we check whether the model is trained using PEFT
    # by checking existance of 'adapter_config.json' is in pretrained_ckpt folder.
    is_peft = os.path.exists(os.path.join(pretrained_ckpt, "adapter_config.json"))

    if is_peft:
        # when using checkpoints trained using PEFT (by us)
        config = PeftConfig.from_pretrained(pretrained_ckpt)
        if config.base_model_name_or_path == "":
            # when pre-training, there is no definition of base_model_name_or_path
            # but, we saved the base model at <parent_path_of_pretrained_ckpt>/base
            config.base_model_name_or_path = os.path.join(os.path.dirname(pretrained_ckpt), "base")

        base_model = HoneybeeForConditionalGeneration.from_pretrained(
            config.base_model_name_or_path,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.half,
            # avoiding RuntimeError: Expected all tensors to be on the same device
            device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
        )
        model = PeftModel.from_pretrained(
            base_model,
            pretrained_ckpt,
            is_trainable=True,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.half,
        )
    else:
        # when using original mllm checkpoints
        model = HoneybeeForConditionalGeneration.from_pretrained(
            pretrained_ckpt,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.half,
        )
    return model


def get_model(pretrained_ckpt, use_bf16=True, load_in_8bit=False):
    """Model Provider with tokenizer and processor.

    Args:
        pretrained_ckpt (string): The path to pre-trained checkpoint.
        use_bf16 (bool, optional): Whether to use bfloat16 to load the model. (Default: True)
        load_in_8bit(bool, optional): Flag to load model in 8it. (Default: False)

    Returns:
        model: Honeybee Model
        tokenizer: Honeybee (Llama) text tokenizer
        processor: Honeybee processor (including text and image)
    """
    # Load model where base_ckpt is different when the target model is trained by PEFT
    model = load_model(pretrained_ckpt, use_bf16, load_in_8bit)

    image_size = model.config.vision_config.image_size
    num_query_tokens = model.config.num_query_tokens
    num_eos_tokens = getattr(model.config.visual_projector_config, "num_eos_tokens", 1)
    num_visual_tokens = num_query_tokens + num_eos_tokens

    # Build processor
    image_processor = HoneybeeImageProcessor(
        size=image_size,
        crop_size=image_size,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
    )
    # Load tokenizer (LlamaTokenizer)
    tokenizer_ckpt = model.config.lm_config.pretrained_tokenizer_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    processor = HoneybeeProcessor(
        image_processor, tokenizer, num_visual_token=num_visual_tokens
    )

    return model, tokenizer, processor


def do_generate(
    prompts, image_list, model, tokenizer, processor, use_bf16=False, **generate_kwargs
):
    """The interface for generation

    Args:
        prompts (List[str]): The prompt text
        image_list (List[str]): Paths of images
        model (HoneybeeForConditionalGeneration): HoneybeeForConditionalGeneration
        tokenizer (AutoTokenizer): AutoTokenizer
        processor (HoneybeeProcessor): HoneybeeProcessor
        use_bf16 (bool, optional): Whether to use bfloat16. Defaults to False.

    Returns:
        sentence (str): Generated sentence.
    """
    if image_list:
        images = [Image.open(_) for _ in image_list]
    else:
        images = None
    inputs = processor(text=prompts, images=images)
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    return sentence
