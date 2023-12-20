import copy
import os
from typing import Union

from transformers import AutoConfig, CLIPVisionConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.models.deformable_detr import DeformableDetrConfig
from transformers.utils import logging

from pipeline.config import AttrDict
from utils import check_local_file

logger = logging.get_logger(__name__)


class HoneybeeVisualProjectorConfig(PretrainedConfig):
    model_type = "mllm_visual_projector"

    def __init__(
        self,
        projector_type: str = "resampler",
        hidden_size: int = 1024,  #
        num_hidden_layers: int = 6,  #
        num_attention_heads: int = 16,  #
        intermediate_size: int = 4096,  #
        attention_probs_dropout_prob: float = 0.1,  #
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-6,  #
        encoder_hidden_size: int = 1024,  # This will be overwritten by vision_model's hidden_size
        pos_emb=False,
        feature_layer_index=-1,  # vision feature layer index; -1: last layer
        num_eos_tokens=1,
        use_cls=True,
        prenorm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder_hidden_size = encoder_hidden_size

        self.pos_emb = pos_emb
        self.feature_layer_index = feature_layer_index
        self.num_eos_tokens = num_eos_tokens
        self.use_cls = use_cls
        self.prenorm = prenorm

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the visual_projector config dict if we are loading from HoneybeeConfig
        if config_dict.get("model_type") == "mllm":
            config_dict = config_dict["projector_config"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class HoneybeeLanguageConfig(PretrainedConfig):
    model_type = "mllm_lm"

    def __init__(
        self,
        pretrained_lm_name_or_path: str = "llama-2-7b-chat",
        delta_model_name_or_path: str = None,
        pretrained_tokenizer_name_or_path: str = "/data/project_ai_cad_162/cxr-llm/hf_models/hf_llama2/tokenize",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained_lm_name_or_path = pretrained_lm_name_or_path
        self.delta_model_name_or_path = delta_model_name_or_path
        self.pretrained_tokenizer_name_or_path = pretrained_tokenizer_name_or_path

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the visual_projector config dict if we are loading from HoneybeeConfig
        if config_dict.get("model_type") == "mllm":
            config_dict = config_dict["lm_config"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class HoneybeeConfig(PretrainedConfig):
    model_type = "mllm"
    is_composition = True

    def __init__(
        self,
        vision_config: dict | None = None,
        visual_projector_config: dict | None = None,
        num_query_tokens: int = 64,
        lm_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        pretrained_vision_name_or_path = vision_config.get("pretrained_vision_name_or_path")
        if pretrained_vision_name_or_path is None:
            # first check whether it is old checkpoint
            if "pretrained_vision_name_or_path" in kwargs:
                # old checkpoint (before introducing EVA)
                pretrained_vision_name_or_path = kwargs["pretrained_vision_name_or_path"]
                vision_config["pretrained_vision_name_or_path"] = pretrained_vision_name_or_path
                logger.info(
                    "Old checkpoint is detected. Use pretrained_vision_name_or_path from kwargs."
                )
            else:
                # we use CLIP ViT-L/14-224 by default
                pretrained_vision_name_or_path = "openai/clip-vit-large-patch14"
                vision_config["pretrained_vision_name_or_path"] = pretrained_vision_name_or_path
                logger.info(
                    f"We use default pretrained vision model: {pretrained_vision_name_or_path}"
                )

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. ")

        if visual_projector_config is None:
            visual_projector_config = {}
            logger.info("projector_config is None. ")

        if lm_config is None:
            # we use LLAMA-2 7b by default
            lm_config = {}
            logger.info("We use default pretrained lm: LLaMA-2-chat 7B")

        # config for vision tower (CLIP)
        # Required key-value of v_enc_config: hidden_size
        vm_local_files_only, vm_file_name = check_local_file(pretrained_vision_name_or_path)
        encoder_type = vision_config.get("encoder_type", "openai.clip")
        v_enc_config = CLIPVisionConfig.from_pretrained(
            vm_file_name,
            local_files_only=vm_local_files_only,
        )
        v_enc_config = v_enc_config.to_dict()
        if "encoder_type" not in v_enc_config:
            v_enc_config["encoder_type"] = encoder_type

        v_enc_config.update(vision_config)
        v_enc_config = AttrDict.from_nested_dicts(v_enc_config)
        self.vision_config = v_enc_config

        # config for projector
        self.num_query_tokens = num_query_tokens
        if visual_projector_config["projector_type"].startswith("d-abs"):
            self.visual_projector_config = DeformableDetrConfig()
            # overwrite manual arguments
            self.visual_projector_config.update(visual_projector_config)
        else:
            self.visual_projector_config = HoneybeeVisualProjectorConfig(**visual_projector_config)
        # overwrite visual_projector.encoder_hidden_size with vision_model.hidden_size
        self.visual_projector_config.encoder_hidden_size = self.vision_config.hidden_size

        # config for language tower (llama-2)
        self.lm_config = HoneybeeLanguageConfig(**lm_config)
        lm_local_files_only, lm_file_name = check_local_file(
            self.lm_config.pretrained_lm_name_or_path
        )
        self.text_config = AutoConfig.from_pretrained(
            lm_file_name,
            local_files_only=lm_local_files_only,
        )

        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.is_encoder_decoder = self.text_config.is_encoder_decoder

        self.use_decoder_only_language_model = (
            self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        )
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

        for attr, value in self.text_config.attribute_map.items():
            if not hasattr(self, attr):
                setattr(self, attr, value)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config
        output["visual_projector_config"] = self.visual_projector_config.to_dict()
        output["lm_config"] = self.lm_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
