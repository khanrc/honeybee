"""Honeybee configuration"""
import timm
from transformers import AutoConfig, CLIPVisionConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.deformable_detr import DeformableDetrConfig
from transformers.utils import logging
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from pipeline.utils import check_local_file

logger = logging.get_logger(__name__)


class HoneybeeVisionConfig(PretrainedConfig):
    def __init__(
        self,
        pretrained_vision_name_or_path: str = "openai/clip-vit-large-patch14",
        image_size: int = 224,
        image_mean = OPENAI_CLIP_MEAN,
        image_std = OPENAI_CLIP_STD,
        hidden_size: int = None,
        encoder_type: str = "openai.clip",
        **kwargs,
    ):
        assert hidden_size is not None, "hidden_size is required for HoneybeeVisionConfig"
        super().__init__(**kwargs)
        self.pretrained_vision_name_or_path = pretrained_vision_name_or_path
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.hidden_size = hidden_size
        self.encoder_type = encoder_type

    @staticmethod
    def from_exp_config(vision_config: dict):
        """Build MLLVisionConfig from exp config (hydra conifg)
        """
        pretrained_vision_name_or_path = vision_config.get("pretrained_vision_name_or_path")
        if pretrained_vision_name_or_path is None:
            raise ValueError("pretrained_vision_name_or_path is required for vision config.")

        vm_local_files_only, vm_file_name = check_local_file(pretrained_vision_name_or_path)
        encoder_type = vision_config["encoder_type"]
        if encoder_type == "openai.clip":
            v_enc_config = CLIPVisionConfig.from_pretrained(
                vm_file_name,
                local_files_only=vm_local_files_only,
            )
            v_enc_config = v_enc_config.to_dict()
            if "encoder_type" not in v_enc_config:  # for eval on previously trained models
                v_enc_config["encoder_type"] = encoder_type
        else:
            raise NotImplementedError()

        v_enc_config.update(vision_config)
        v_enc_config = HoneybeeVisionConfig(**v_enc_config)

        return v_enc_config


class HoneybeeVisualProjectorConfig(PretrainedConfig):
    def __init__(
        self,
        projector_type: str = "c-abs",
        num_eos_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.num_eos_tokens = num_eos_tokens

    @staticmethod
    def from_exp_config(
        projector_config: dict,
        vision_hidden_size: int,
        lm_hidden_size: int,
    ):
        if projector_config["projector_type"] == "d-abs":
            projector_config = DeformableDetrConfig(**projector_config).to_dict()

        # projector has three inter-module configs:
        # 1) encoder_hidden_size (hidden size of vision model)
        # 2) output_hidden_size (hidden size of LLM)
        # the number of query tokens  (total num_visual_tokens = num_query_tokens + num_eos_tokens)
        inter_module_configs = {
            "encoder_hidden_size": vision_hidden_size,
            "output_hidden_size": lm_hidden_size,
        }

        projector_config = HoneybeeVisualProjectorConfig(
            **(projector_config | inter_module_configs),
        )

        return projector_config


class HoneybeeLanguageConfig(PretrainedConfig):
    def __init__(
        self,
        pretrained_lm_name_or_path: str = "llama-2-7b",
        pretrained_tokenizer_name_or_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained_lm_name_or_path = pretrained_lm_name_or_path
        self.pretrained_tokenizer_name_or_path = (
            pretrained_tokenizer_name_or_path or pretrained_lm_name_or_path
        )


class HoneybeeConfig(PretrainedConfig):
    is_composition = True

    def __init__(
        self,
        vision_config: dict,
        projector_config: dict,
        lm_config: dict,
        **kwargs,
    ):
        """Honeybee model config.

        This init function is called with two different scenario:
        - in PT, explicitly called in train.py, with **hydra exp config**.
        - in FT, implicitly called in from_pretrained, with **hf model config**.

        Thus, we need to address both cases.
        """
        super().__init__(**kwargs)

        # Note) three inter-module configs (vision -> projector or lm -> projector):
        # 1) vision_config.hidden_size -> projector_config.encoder_hidden_size
        # 2) text_config.hidden_size -> projector_config.output_hidden_size
        # the number of query tokens (total num_visual_tokens = num_query_tokens + num_eos_tokens)

        # Vision config
        self.vision_config = HoneybeeVisionConfig.from_exp_config(vision_config)

        # LM config (from exp config)
        self.lm_config = HoneybeeLanguageConfig(**lm_config)
        lm_local_files_only, lm_file_name = check_local_file(
            self.lm_config.pretrained_lm_name_or_path
        )
        self.text_config = AutoConfig.from_pretrained(
            lm_file_name,
            local_files_only=lm_local_files_only,
        )

        # Projector config
        self.projector_config = HoneybeeVisualProjectorConfig.from_exp_config(
            projector_config,
            vision_hidden_size=self.vision_config.hidden_size,
            lm_hidden_size=self.text_config.hidden_size,
        )

    @property
    def num_visual_tokens(self):
        return self.projector_config.num_query_tokens + self.projector_config.num_eos_tokens

    @property
    def hidden_size(self):
        # hidden_size is required for deepspeed auto config
        return self.text_config.hidden_size

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()
        for k, v in output.items():
            if isinstance(v, PretrainedConfig):
                output[k] = v.to_dict()

        return output
    
    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # update old config
        if "hidden_size" in config_dict:
            config_dict.pop("hidden_size")

        if "visual_projector_config" in config_dict:
            config_dict["projector_config"] = config_dict.pop("visual_projector_config")
            config_dict["projector_config"].pop("encoder_hidden_size")
            config_dict["projector_config"]["num_query_tokens"] = config_dict.pop("num_query_tokens")

        return super().from_dict(config_dict, **kwargs)
