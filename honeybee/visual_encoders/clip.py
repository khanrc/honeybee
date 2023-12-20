from transformers import CLIPVisionModel
from .visual_encoder_mixin import VisualEncoderMixin


class CustomCLIP(CLIPVisionModel, VisualEncoderMixin):
    def get_dtype(self):
        return self.vision_model.embeddings.class_embedding.data.dtype

    def get_num_tokens(self):
        return self.vision_model.embeddings.num_positions  # this includes cls token
