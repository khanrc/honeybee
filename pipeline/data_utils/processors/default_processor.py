from torchvision import transforms
from PIL import Image

from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from .builder import PROCESSORS


@PROCESSORS.register_module()
class DefaultProcessor:
    def __init__(self, image_size=224, image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD):
        self.image_size = image_size

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ])

    def __call__(self, image):
        if image:
            image_input = self.image_transform(image)
        else:
            image_input = None

        return image_input
