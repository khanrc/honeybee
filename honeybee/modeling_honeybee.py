from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModelForCausalLM
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from honeybee.configuration_honeybee import HoneybeeConfig
from honeybee.visual_encoders import build_encoder
from utils import check_local_file

from .projectors import CAbstractor, DAbstractor

logger = logging.get_logger(__name__)


@dataclass
class HoneybeeForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`HoneybeeForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.

        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


def get_ltor_masks_and_position_ids_from_embeddings(data):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()[:2]

    # Attention mask (lower triangular).
    att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
    ).view(att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size()[:2], dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data[..., 0])

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def apply_delta(base_model, delta_model_name_or_path):
    # Reference: fastchat/model/apply_delta.py from https://github.com/lm-sys/FastChat (vicuna)
    print(f"Loading the delta weights from {delta_model_name_or_path}")
    local_files_only, delta_file_name = check_local_file(delta_model_name_or_path)
    delta, loading_info = AutoModelForCausalLM.from_pretrained(
        delta_file_name,
        local_files_only=local_files_only,
        output_loading_info=True,
    )
    print("[Loading info for delta model] \n", loading_info)
    print("Applying the delta ...")
    for name, param in tqdm(base_model.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]

    return base_model


class HoneybeePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = HoneybeeConfig
    base_model_prefix = "mllm"
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
        factor = self.config.initializer_range
        if (
            isinstance(module, nn.Conv2d)  # noqa: SIM101
            or isinstance(module, nn.Embedding)
            or isinstance(module, nn.Linear)
        ):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.Parameter):
            raise ValueError
            nn.init.trunc_normal_(module.data, mean=0.0, std=factor)

    def _set_gradient_checkpointing(self, module, value=False):
        from transformers.models.clip import modeling_clip

        if isinstance(module, modeling_clip.CLIPEncoder):
            module.gradient_checkpointing = value


HONEYBEE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`HoneybeeConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

HONEYBEE_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`HoneybeeProcessor`]. See [`HoneybeeProcessor.__call__`]
            for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

HONEYBEE_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it. Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

HONEYBEE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`HoneybeeProcessor`]. See [`HoneybeeProcessor.__call__`]
            for details.

        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary of the language model. Input tokens can optionally be
            provided to serve as text prompt, which the language model can continue.

            Indices can be obtained using [`HoneybeeProcessor`]. See [`HoneybeeProcessor.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary of the language model. Only relevant in case an
            encoder-decoder language model (like T5) is used.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are decoder input IDs?](../glossary#decoder-input-ids)

        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            Only relevant in case an encoder-decoder language model (like T5) is used.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def get_media_indices(my_list):
    if isinstance(my_list, torch.Tensor):
        my_list = my_list.cpu().tolist()
    result = []
    for i in range(len(my_list)):
        if i == 0 and my_list[i] < 0:
            result.append(i)
        elif my_list[i] != my_list[i - 1] and my_list[i] < 0:
            result.append(i)
    return result


@add_start_docstrings(
    """
    Honeybee model for generating text given an image and an optional text prompt.
    """,
    HONEYBEE_START_DOCSTRING,
)
class HoneybeeForConditionalGeneration(HoneybeePreTrainedModel):
    config_class = HoneybeeConfig
    main_input_name = "pixel_values"

    def build_projector(self, config: HoneybeeConfig):
        """Build projector (abstractor) and query_tokens (optionally for resampler)"""
        proj_config = config.visual_projector_config
        proj_type = proj_config.projector_type
        num_tokens = self.vision_model.get_num_tokens()
        output_hidden_size = config.text_config.hidden_size  # LM hidden size

        self.abstractor = {
            "c-abs": CAbstractor,
            "d-abs": DAbstractor,
        }[
            proj_type
        ](proj_config, num_tokens, output_hidden_size)

    def __init__(self, config: HoneybeeConfig):
        super().__init__(config)
        self.config = config
        self.encoder_type = config.vision_config.get("encoder_type", "openai.clip")
        self.vision_model = build_encoder(config.vision_config)

        # visual projector
        proj_config = config.visual_projector_config
        self.proj_type = proj_config.projector_type
        self.num_query_tokens = config.num_query_tokens
        self.build_projector(config)
        # deformable attention only supports fp32
        if type(self.abstractor) == DAbstractor:
            self.abstractor.to(torch.float)

        # language model (decoder)
        lm_local_files_only, lm_file_name = check_local_file(
            config.lm_config.pretrained_lm_name_or_path
        )
        language_model = AutoModelForCausalLM.from_pretrained(
            lm_file_name,
            local_files_only=lm_local_files_only,
        )
        if config.lm_config.delta_model_name_or_path is not None:
            apply_delta(language_model, config.lm_config.delta_model_name_or_path)
        self.language_model = language_model

        # Initialize weights and apply final processing
        # Here, weights of abstractor (HoneybeeVisualProjectorModel) is initialized
        self.post_init()
        self.main_input_name = "input_ids"
        from transformers import GenerationConfig

        self.generation_config = GenerationConfig(
            max_length=512,
            do_sample=True,
            top_k=3,
            pad_token_id=0,
            unk_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if (
            len(hf_device_map) > 1
            and "language_model" not in hf_device_map
            and torch.cuda.device_count() > 1
        ):
            # warn users about unexpected behavior when using multi-GPU + Honeybee + `accelerate`.
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    def _get_input_dtype(self):
        dtype = self.vision_model.get_dtype()

        return dtype

    def forward_and_project_vision(self, pixel_values):
        """Forward pixel_values & project (abstract) the visual features to LLM embedding space."""
        assert pixel_values is not None

        # =================================================== #
        # Forward vision model
        # =================================================== #
        v_outputs = self.vision_model(pixel_values, return_dict=True, output_hidden_states=True)
        layer_index = self.config.visual_projector_config.feature_layer_index
        if type(layer_index) == list:  # for multi-scale deformable attn.
            image_embeds = torch.stack(v_outputs.hidden_states, dim=1)[
                :, layer_index
            ]  # [B, num_states, len, dim]
        else:
            image_embeds = v_outputs.hidden_states[layer_index]  # [B, num_patches+1, dim]

        # =================================================== #
        # Forward projector
        # =================================================== #
        if self.proj_type == "d-abs":
            query_features = self.abstractor(image_embeds)["last_hidden_state"]
        else:
            query_features = self.abstractor(image_embeds)

        # query_features: [B, L, dim]
        return query_features

    @add_start_docstrings_to_model_forward(HONEYBEE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=HoneybeeForConditionalGenerationModelOutput, config_class=HoneybeeConfig
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        num_images,
        non_padding_mask: Optional[torch.LongTensor] = None,
        non_media_mask: Optional[torch.LongTensor] = None,
        prompt_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        is_null_image=None,
    ) -> Union[Tuple, HoneybeeForConditionalGenerationModelOutput]:
        r"""
        Returns:
        """
        if pixel_values is not None:
            pixel_values = pixel_values.to(self._get_input_dtype())
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # get text embedding
        text_tokens_ = input_ids.clone()
        batch_size = input_ids.shape[0]

        media_token_indices = [
            # [:-1] since we would not use the last token for embedding
            get_media_indices(text_tokens_[i][:-1])
            for i in range(batch_size)
        ]
        text_tokens_[text_tokens_ < 0] = 1  # Not used
        text_embeds = self.get_input_embeddings()(text_tokens_)  # Temporally Embedding
        if hasattr(self.language_model, "transformer") and hasattr(
            self.language_model.transformer, "word_embeddings_layernorm"
        ):
            text_embeds = self.language_model.transformer.word_embeddings_layernorm(text_embeds)

        if pixel_values is not None:
            query_features = self.forward_and_project_vision(pixel_values)
            img_seq_length = query_features.shape[1]  # [B, L, lm_dim]
        num_images_per_sample = num_images.long().cpu().tolist()

        text_chunk_embeds = []
        img_idx = 0
        # sanity check (-1 is image token)
        n_vision_tokens = (input_ids == -1).sum(1)
        assert (
            (n_vision_tokens == num_images * img_seq_length).all().item()
        ), f"Expected #img_tokens={n_vision_tokens}, but got {num_images * img_seq_length}"

        for b in range(batch_size):
            start = 0
            result = []
            if len(media_token_indices[b]) > 0:
                for i, pos in enumerate(media_token_indices[b]):
                    if pos > start:
                        result.append(text_embeds[b, start:pos])  # add tokens before visual tokens
                    result.append(query_features[img_idx + i])  # add visual tokens
                    start = pos + img_seq_length
            if start < text_embeds.shape[1]:
                result.append(text_embeds[b, start:])  # add instruction & response

            img_idx += num_images_per_sample[b]
            text_chunk_embeds.append(torch.cat(result, dim=0))

        # Actual Input Embeddings
        input_embeds = torch.stack(text_chunk_embeds, dim=0)

        # Create causal mask and position ids
        _, loss_mask, position_ids = get_ltor_masks_and_position_ids_from_embeddings(input_embeds)

        # Calculate the loss_mask
        non_padding_mask = non_padding_mask.long()
        non_media_mask = non_media_mask.long()
        prompt_mask = prompt_mask.long()
        loss_mask = loss_mask[:, :-1]

        loss_mask = loss_mask * non_padding_mask * non_media_mask * prompt_mask
        labels[:, 1:][loss_mask != 1] = -100

        # Forward into GPT
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            output_attentions=self.config.output_attentions,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        isdecoder=True,
        is_null_image=None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if pixel_values is not None:
            pixel_values = pixel_values.to(self._get_input_dtype())

        if input_ids is None:
            return self.language_model.generate(attention_mask=attention_mask, **generate_kwargs)

        if attention_mask is None:
            attention_mask = input_ids.new_ones(*input_ids.shape)

        batch_size = input_ids.size(0)
        media_token_indices = [get_media_indices(input_ids[i]) for i in range(batch_size)]
        num_images_per_sample = [len(x) for x in media_token_indices]
        # pre-compute n_vision_tokens for sanity check
        n_vision_tokens = (input_ids == -1).sum(1)  # -1 is image token id
        input_ids = input_ids.clone()  # prevent inplace modify
        input_ids[input_ids < 0] = 0  # Not used

        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()
        batch_size = input_ids.shape[0]
        # get text embedding
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if hasattr(self.language_model, "transformer") and hasattr(
            self.language_model.transformer, "word_embeddings_layernorm"
        ):
            inputs_embeds = self.language_model.transformer.word_embeddings_layernorm(inputs_embeds)

        # get visual embedding
        if pixel_values is not None:
            pixel_values = pixel_values.to(input_ids.device)
            with torch.no_grad():
                image_embeds = self.forward_and_project_vision(pixel_values)
                img_seq_length = image_embeds.shape[1]  # [B, L, lm_dim]

            # sanity check
            num_images = torch.as_tensor(num_images_per_sample, device=n_vision_tokens.device)
            assert (
                (n_vision_tokens == num_images * img_seq_length).all().item()
            ), f"Expected #img_tokens={n_vision_tokens}, but got {num_images * img_seq_length}"

            # ===================
            # Get actual input embeddings
            # ===================
            text_chunk_embeds = []
            text_chunk_attns = []
            img_idx = 0
            for b in range(batch_size):
                start = 0
                result = []
                result_attn = []
                for i, pos in enumerate(media_token_indices[b]):
                    if pos > start:
                        result.append(inputs_embeds[b, start:pos])
                        result_attn.append(attention_mask[b, start:pos])
                    result.append(image_embeds[img_idx + i])
                    img_embed_attn_mask = torch.ones(
                        image_embeds[img_idx + i].shape[0], device=inputs_embeds.device
                    )
                    # For demo, is_null_image is None, so, we donot need masking with it.
                    if is_null_image is not None:
                        # Masking the null image in the attention mask during training,
                        img_embed_attn_mask = img_embed_attn_mask * ~is_null_image[img_idx + i]
                    result_attn.append(img_embed_attn_mask)
                    start = pos + img_seq_length
                if start < inputs_embeds.shape[1]:
                    result.append(inputs_embeds[b, start:])
                    result_attn.append(attention_mask[b, start:])

                img_idx += num_images_per_sample[b]
                text_chunk_embeds.append(torch.cat(result, dim=0))
                text_chunk_attns.append(torch.cat(result_attn, dim=0))
            inputs_embeds = torch.stack(text_chunk_embeds, dim=0)
            attention_mask = torch.stack(text_chunk_attns, dim=0)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=None,
        past_key_values=None,
        attention_mask=None,
        **model_kwargs,
    ):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model,
        # the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "is_decoder": True,
        }
