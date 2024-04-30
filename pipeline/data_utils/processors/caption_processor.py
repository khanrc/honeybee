from torchvision import transforms as T
from PIL import Image

from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from pipeline.data_utils.randaugment import RandomAugment
from .default_processor import DefaultProcessor
from .builder import PROCESSORS


@PROCESSORS.register_module()
class CaptionProcessor(DefaultProcessor):
    def __init__(self, image_size=224, min_scale=0.5, aug=None, image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD):
        self.image_size = image_size
        self.min_scale = min_scale

        if aug == "rand":
            # randaugment
            transforms = [
                T.RandomResizedCrop(image_size,scale=(min_scale, 1.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            ]
        elif aug == "default":
            # scale + flip
            transforms = [
                T.RandomResizedCrop(image_size,scale=(min_scale, 1.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(),
            ]
        elif aug == "scale":
            transforms = [
                T.RandomResizedCrop(image_size,scale=(min_scale, 1.0), interpolation=Image.BICUBIC),
            ]
        elif aug in [None, "none"]:
            # same as default processor
            transforms = [
                T.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            ]
        else:
            raise ValueError(f"Unknown aug type: {aug}")

        transforms += [
            T.ToTensor(),
            T.Normalize(image_mean, image_std),
        ]
        self.image_transform = T.Compose(transforms)
        self.text_transform = None
