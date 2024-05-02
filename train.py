import os
import sys
import warnings
from pathlib import Path

import hydra
import torch
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from sconf import Config
from transformers import ProgressCallback, set_seed, enable_full_determinism

from honeybee import HoneybeeConfig, HoneybeeForConditionalGeneration, build_honeybee_tokenizer
from honeybee.processing_honeybee import HoneybeeProcessor
from pipeline.config import set_config, save_config, load_config
from pipeline.custom_trainer import (
    CustomProgressCallback,
    CustomTrainingArguments,
    CustomTrainer,
)
from pipeline.utils import print_trainable_parameters, set_trainable_parameters
from pipeline.data_utils import datasets_provider
from pipeline.data_utils.datasets import PretrainWebdataset
from pipeline.data_utils.multidata_wrapper import MultiDataset
from utils.callbacks import AbstractorTypeConvertCallback
from utils.interpolate_clip_resolution import surgery_clip_pos_emb_
from utils import get_logger, init_logger_
import utils
from tasks import build_task

logger = get_logger()


def log_examples(dataset, tokenizer, n_log=10, log_input_ids=False, log_targets=False):
    """Log examples with masks."""
    for i, ex in enumerate(dataset):
        text_raw = ex["text_raw"]
        task_type = ex.get("task_type", "captioning")
        print(f"[{task_type}]")
        print(text_raw)

        txt = ex["text"]
        L = txt["seq_length"]  # length including eos token
        input_ids = txt["input_ids"]
        assert input_ids[0] == tokenizer.bos_token_id
        if input_ids[L - 1] != tokenizer.eos_token_id:
            print("!!! [Warning] eos token is not found at the last of sequence !!!")
        # image can be null -> visual_tokens_cnt can be 0
        assert (input_ids == -1).sum().item() in [
            0,
            tokenizer.num_visual_tokens,
        ], f"#img_tokens == {(input_ids == -1).sum()}"

        ids = input_ids[1:]  # remove bos token
        L -= 1

        if log_targets:
            prompt = ids[:L].masked_select(txt["loss_mask"][:L].bool())
            prompt_txt = tokenizer.decode(prompt)
            print("---")
            print(f"[Prompt]\n{prompt_txt}")

        if log_input_ids:
            print(f"[input_ids] (L={L+1})\n{input_ids[:L+1]}")  # L+1 to print both bos/eos tokens

        print("=" * 80)
        if (i + 1) == n_log:
            break


def update_model_config_from_ckpt_(config):
    """
    Update `config.model_config` with the one from pre-trained ckpt if available and 
    `config.load_model_config_from_ckpt` is True.
    """
    if not config.get("load_model_config_from_ckpt", False):
        return

    if not config.get("pretrained_ckpt", None):
        return

    # load pre-trained config
    root = Path(config.pretrained_ckpt).parent
    pt_config = load_config(root / "exp_config.yaml")

    # update
    config.model_config = pt_config.model_config

    # log & save
    if utils.is_main_process():
        save_config(config)

        print("=" * 100)
        warnings.warn(  # noqa: B028
            "[auto-model-config] This feature may cause unexpected behavior "
            "when config interpolation is applied to model_config. "
            "Please be careful when using this feature."
        )
        print("[Loaded model configs from checkpoint]")
        print(OmegaConf.to_yaml(config.model_config.asdict()))
        print("[SAVE] The updated config is saved again to `output_dir/exp_config.yaml`.")
        print("=" * 100)


@hydra.main(version_base=None, config_path="configs", config_name="pretrain")
def main(config: DictConfig):
    run_cmd = " ".join(sys.argv)
    utils.print_rank_0(f" >> Run command: `{run_cmd}`")

    config = set_config(config, save=True)

    logger = get_logger()
    init_logger_(logger)  # enable logger

    if utils.is_main_process():
        print("=" * 100)
        print(Config(config).dumps())
        print("=" * 100)

    if config.deterministic:
        enable_full_determinism(config.trainer_kwargs.seed)
    else:
        set_seed(config.trainer_kwargs.seed)

    # update model config from ckpt
    if config.get("pretrained_ckpt", None) and config.get("load_model_config_from_ckpt", False):
        update_model_config_from_ckpt_(config)

    # prepare model
    logger.info("Build model...")
    if config.train.mode in ["debug", "debug-lm"]:
        if config.train.mode == "debug":
            config.model_config.lm_config.pretrained_lm_name_or_path = "gpt2"
            config.model_config.lm_config.pretrained_tokenizer_name_or_path = "gpt2"

        config.train.mode = "debug"

    if config.train.mode in ["pt", "debug"]:
        honeybee_config = HoneybeeConfig(**config.model_config)
        model = HoneybeeForConditionalGeneration(honeybee_config)

        # set dtype of model
        if config.trainer_kwargs.bf16:
            model.to(torch.bfloat16)
        else:
            model.to(torch.float16)

    elif config.train.mode == "sft":
        model = HoneybeeForConditionalGeneration.from_pretrained(
            config.pretrained_ckpt,
            torch_dtype=torch.bfloat16 if config.trainer_kwargs.bf16 else torch.float16,
        )

    else:
        raise ValueError(config.train.mode)

    # pos emb interpolation
    enc_type = config.model_config.vision_config.encoder_type
    image_size = config.image_size
    cur_vm_image_size = model.config.vision_config.image_size
    if image_size != cur_vm_image_size:
        logger.info(f"Interpolate position embedding: {cur_vm_image_size} -> {image_size}")
        if enc_type == "openai.clip":
            assert image_size > cur_vm_image_size, "The image size is bigger than the current CLIP image size"

            surgery_clip_pos_emb_(model.vision_model, image_size)
            model.config.vision_config.image_size = image_size
        else:
            raise ValueError(f"Pos emb interpolation is not supported for {enc_type}.")
    else:
        if enc_type == "hivit":
            logger.info(f"Skip pos emb interpolation: it is always re-initialized for {enc_type}.")

    # save model config to 'output_dir/config.json'
    model.config.to_json_file(os.path.join(config.output_dir, "config.json"))

    # apply lora if requested
    if config.lora_config.use_lora:
        # LoRA
        peft_config = LoraConfig(
            target_modules=r"{}".format(config.lora_config.target_modules),
            inference_mode=config.lora_config.inference_mode,
            r=config.lora_config.lora_r,
            lora_alpha=config.lora_config.lora_alpha,
            lora_dropout=config.lora_config.lora_dropout,
            modules_to_save=config.lora_config.modules_to_save,
        )
        if config.train.mode == "pt":
            # save checkpoint as base model
            model.save_pretrained(os.path.join(config.output_dir, "base"))
        # obtain peft-applied model and set the base_model_name_or_path
        model = get_peft_model(model, peft_config)

    else:
        # Full training
        # if not use lora, manually freeze base model's layers
        # Then, we set trainable parameters
        for param in model.parameters():
            param.requires_grad = False
        # set trainable parameters
        set_trainable_parameters(model, config.train.module_to_update)

    # partially-freeze vision encoder (currently only supported with HiViT)
    if config.get("n_freeze_vision_blocks", None):
        model.vision_model.freeze_blocks(config.n_freeze_vision_blocks)

    if utils.is_main_process():
        for i, (name, param) in enumerate(model.named_parameters()):
            if param.requires_grad and i % 10 == 0:
                print(f"{name} --> {param.requires_grad}")
        print_trainable_parameters(model)

    model.train()

    logger.info("Build tokenizer...")
    hf_model_config = utils.unwrap_peft(model).config
    tokenizer = build_honeybee_tokenizer(
        hf_model_config.lm_config.pretrained_tokenizer_name_or_path,
        num_visual_tokens=hf_model_config.num_visual_tokens,
    )
    logger.info(f" >> Tokenizer is successfully loaded: {tokenizer}")
    if config.data_config.train_cfg.max_length > tokenizer.model_max_length - 1:
        logger.critical(
            f" !!! config max_length ({config.data_config.train_cfg.max_length}) is larger than "
            f"tokenizer.model_max_length ({tokenizer.model_max_length}). It is updated to "
            f"{tokenizer.model_max_length - 1}."
        )
        config.data_config.train_cfg.max_length = tokenizer.model_max_length - 1

    logger.info("Build datasets ...")
    image_mean = hf_model_config.vision_config.image_mean
    image_std = hf_model_config.vision_config.image_std
    proc_kwargs = {
        "image_mean": image_mean,
        "image_std": image_std,
    }
    train_data, _ = datasets_provider(
        config.data_config, tokenizer, split="train", proc_kwargs=proc_kwargs
    )
    valid_data, valid_processors = datasets_provider(
        config.data_config, tokenizer, split="valid", proc_kwargs=proc_kwargs
    )

    # build evaluation tasks
    logger.info("Build eval tasks ...")
    tasks = []
    if config.train.eval_on_task:
        # Use valid processor config
        image_processor = valid_processors["default"]  # use default valid processor (no aug)
        processor = HoneybeeProcessor(
            image_processor,
            tokenizer,
        )
        debug = config.train.mode == "debug"
        for _, task_cfg in config.tasks.items():
            task = build_task(model, tokenizer, processor, task_cfg, debug=debug)
            tasks.append(task)

    # log training dataset information
    if utils.is_main_process():
        n_steps = config.trainer_kwargs.max_steps
        batch_size = config.trainer_kwargs.per_device_train_batch_size
        total_batch_size = utils.get_world_size() * batch_size
        total_examples = n_steps * total_batch_size

        is_multi = isinstance(train_data, MultiDataset)
        names = list(config.data_config.train_dataset.keys())
        sampling_weights = train_data.sampling_weights if is_multi else [1.0]
        datasets = train_data.datasets if is_multi else [train_data]
        assert len(names) == len(sampling_weights) == len(datasets)

        print("=" * 100)
        print("Training datasets:")
        for name, w, dset in zip(names, sampling_weights, datasets):
            N = len(dset)
            if isinstance(dset, (PretrainWebdataset, wds.WebDataset)):
                N *= utils.get_world_size()
            n_examples = int(total_examples * w)
            epoch = n_examples / N
            print(
                f"    [{name:14s}] weight = {w:.3f} | epoch = {epoch:.3f} | "
                f"#data = {N:10d} | #see = {n_examples:6d}"
            )
        print("=" * 100)

    if utils.is_main_process() and config.train.mode == "debug":
        utils.barrier()
        log_examples(train_data, tokenizer)
        utils.barrier()

    logger.info("Start training...")
    trainer = CustomTrainer(
        config=config,
        tokenizer=tokenizer,
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tasks=tasks,
        args=CustomTrainingArguments(**config.trainer_kwargs),
        callbacks=[AbstractorTypeConvertCallback],
    )
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(CustomProgressCallback)

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    # save checkpoint after training
    model.save_pretrained(os.path.join(config.output_dir, "last"), safe_serialization=False)


if __name__ == "__main__":
    main()
