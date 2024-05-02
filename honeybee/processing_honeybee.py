import torch
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding


class HoneybeeProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = "HoneybeeTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__()
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, texts=None, images=None, return_tensors="pt"):
        if all(img is None for img in images):  # for the texts only case
            images = None

        if texts is None and images is None:
            raise ValueError("You have to specify either texts or images. Both cannot be none.")

        if texts is not None:
            # Return keys: ['input_ids', 'attention_mask']
            encoding = self.tokenizer.batch_encode_prompt(prompts=texts, padding_side="left", no_eos=True)

        if images is not None:
            images = [image for image in images if image is not None]  # filter out none images
            image_features = torch.stack([self.image_processor(image) for image in images])

        if texts is not None and images is not None:
            encoding["pixel_values"] = image_features
            return BatchEncoding(data=encoding)
        elif texts is not None:
            return BatchEncoding(data=encoding)
        else:
            return BatchEncoding(data=dict(pixel_values=image_features), tensor_type=return_tensors)
