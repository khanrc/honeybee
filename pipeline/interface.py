import os

import torch
from peft import PeftConfig, PeftModel

from honeybee import build_honeybee_tokenizer
from honeybee.modeling_honeybee import HoneybeeForConditionalGeneration
from honeybee.processing_honeybee import HoneybeeProcessor
from pipeline.data_utils.processors.default_processor import DefaultProcessor
import utils


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
        # when using checkpoints trained using PEFT
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
        model = HoneybeeForConditionalGeneration.from_pretrained(
            pretrained_ckpt,
            load_in_8bit=load_in_8bit,
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
        tokenizer: Honeybee text tokenizer
        processor: Honeybee processor (including text and image)
    """
    model = load_model(pretrained_ckpt, use_bf16, load_in_8bit)

    # Load processor and tokenizer
    image_processor = DefaultProcessor(
        image_size=model.config.vision_config.image_size,
        image_mean=model.config.vision_config.image_mean,
        image_std=model.config.vision_config.image_std,
    )
    tokenizer = build_honeybee_tokenizer(
        model.config.lm_config.pretrained_tokenizer_name_or_path,
        num_visual_tokens=model.config.num_visual_tokens,
    )
    processor = HoneybeeProcessor(image_processor, tokenizer)

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
        image_list = utils.load_images(image_list)
    else:
        images = None
    inputs = processor(text=prompts, images=images)
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    return sentence
