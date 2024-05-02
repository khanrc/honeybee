import os
import random
import glob
from typing import List, Union

import torch
import webdataset as wds
from torch.utils.data import IterableDataset

# currently, only support webdataset format, i.e., .tar extetion.
_FORMAT_EXT = {
    "webdataset": "tar",
}

def make_url_list(root, data_names, format="webdataset"):
    urls = []
    for name in data_names:
        data_path = os.path.join(root, name)
        assert os.path.exists(data_path), f"Data path {data_path} does not exist."
        _urls = sorted(glob.glob(os.path.join(data_path, f"*.{_FORMAT_EXT[format]}"), recursive=True))
        urls.extend(_urls)
    return urls


class PretrainWebdataset(IterableDataset):
    """
    Webstyle dataset for captioning datasets. Note that the webstyle dataset only support training
    and cannot be used for evaluation.
    """
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        root: str,
        num_pairs: Union[int, List[int]],
        data_names: Union[str, List[str]] = ["blip_cc_sbu", "blip_laion"],
        data_format: str = "webdataset",
        **ignore_kwargs,
    ):
        total_gpus = int(os.environ.get("WORLD_SIZE", 1))

        if not isinstance(data_names, list):
            data_names = [data_names]
        if not isinstance(num_pairs, list):
            num_pairs = [num_pairs]

        assert len(data_names) == len(num_pairs), "data_names and num_pairs should have the same length."

        super().__init__()

        self.tokenizer = tokenizer
        self.templatizer = None
        # NOTE pt_webdataset is assumed as captioning.
        self.processors = processors["captioning"]
        self.max_length = max_length

        # Currently we only support the image modality for media modality.
        self.media_tokens = tokenizer.media_tokens
        self.media_lengths = tokenizer.num_visual_tokens  # visual(image) token length

        # Make url list from the root and data_names.
        self.urls = make_url_list(root, data_names, format=data_format)
        assert len(self.urls) > 0 and isinstance(self.urls[0], str)

        self.n_samples = 0
        for n_pairs in num_pairs:
            self.n_samples += int(n_pairs // total_gpus)

        print(f"Total pre-training items {data_names}: {self.n_samples * total_gpus}")

    def set_templatizer(self, templatizer):
        self.templatizer = templatizer

    def __len__(self):
        return self.n_samples

    def identity(self, x):
        return x

    def process_image(self, images):
        if not isinstance(images, list):
            images = [images]
        images = [self.processors(image=image) for image in images]
        images = torch.stack(images, dim=0)  # [1, 3, 224, 224]
        return images

    def process_text(self, text):
        if self.templatizer is not None:
            examples = [{"caption": text}]
            text = self.templatizer(examples)

        return text

    def custom_return(self, data):
        img, txt = data

        if self.templatizer is not None:
            text_input = self.tokenizer.encode_prompt(
                txt, self.max_length
            )
            return {
                "image": img,
                "text": text_input,
                "text_raw": txt,
            }
        else:
            raise ValueError("Templatizer is not set.")

    def setup(self):
        rng = random.Random(42)  # seed=42 for shuffle rng
        urls = self.urls.copy()
        # Convert urls into ResampledShards instead of using resampled=True in WebDataset,
        # for deterministic resampling (option of deterministic=True).
        urls = wds.ResampledShards(urls, deterministic=True)

        ds = (
            wds.WebDataset(urls, handler=wds.warn_and_continue)
            .shuffle(1000, rng=rng)
            .decode("pil", handler=wds.warn_and_continue)
            .to_tuple("jpg", "txt", handler=wds.warn_and_continue)
            .map_tuple(self.process_image, self.process_text)
            .map(self.custom_return)
            .with_epoch(self.__len__() * 1000000)  # to generate an infinite stream of samples
            .with_length(self.__len__())
        )

        return ds
