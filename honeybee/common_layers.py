import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from honeybee.configuration_honeybee import HoneybeeConfig


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        output = torch.nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)


class HoneybeePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = HoneybeeConfig
    base_model_prefix = "honeybee"
    supports_gradient_checkpointing = True

    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"language_model.encoder.embed_tokens.weight",
        r"language_model.decoder.embed_tokens.weight",
        r"language_model.lm_head.weight",
    ]
    _no_split_modules = [
        "CLIPEncoderLayer",
        "LlamaDecoderLayer",
        "HoneybeeVisualProjectorLayer",
        "LlamaForCausalLM",
        "Parameter",
    ]
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if (
            isinstance(module, nn.Conv2d)  # noqa: SIM101
            or isinstance(module, nn.Embedding)
            or isinstance(module, nn.Linear)
        ):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            raise ValueError()
