import os
import random
from typing import List, Union

import braceexpand
import torch
import webdataset as wds
from torch.utils.data import IterableDataset


def expand_urls(urls):
    def decode(urls):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            result.extend(braceexpand.braceexpand(url))
        return result

    if isinstance(urls, str):
        return decode(urls)
    elif isinstance(urls, tuple):
        results = []
        for urls_ in urls:
            results += decode(urls_)
        return results
    else:
        return list(urls)


class PretrainWebdataset(IterableDataset):
    """
    Webstyle dataset for captioning datasets. Note that the webstyle dataset only support training
    and cannot be used for evaluation.
    """

    names = {
        "blip_cc_sbu",
        "blip_laion",
        "coyo100m",
    }
    # CHECK 요 데이터 아래 포맷으로 받을 수 있나>
    urls_info = {
        "blip_cc_sbu": "./data/pretrain/BLIP/cc_sbu/cc_sbu_dataset/{00000..01255}.tar",
        "blip_laion": "./data/pretrain/BLIP/laion/laion_dataset/{00000..10488}.tar",
        "coyo100m": "./data/pretrain/coyo/v1.0.0/webdataset/100m/{000000..008191}.tar",
    }

    n_samples_info = {
        "blip_cc_sbu": 10986670,  # 11M
        "blip_laion": 87795279,  # 87M
        "coyo100m": 100000000,
    }

    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        data_names: Union[str, List[str]] = ["blip_cc_sbu", "blip_laion"],
        **ignore_kwargs,
    ):
        total_gpus = int(os.environ.get("WORLD_SIZE", 1))

        if not isinstance(data_names, list):
            data_names = [data_names]

        for name in data_names:
            if name not in self.names:
                raise NotImplementedError(
                    f"Dataset type {name} is not supported. (Supported types: {self.names})"
                )

        super().__init__()

        self.tokenizer = tokenizer
        self.templatizer = None
        # NOTE pt_webdataset is assumed as captioning.
        self.processors = processors["captioning"]
        self.max_length = max_length

        # Currently we only support the image modality for media modality.
        self.media_tokens = tokenizer.media_tokens
        self.media_lengths = tokenizer.num_visual_tokens  # visual(image) token length

        urls = tuple(self.urls_info[name] for name in data_names)
        self.urls = expand_urls(urls)
        assert isinstance(self.urls[0], str)

        self.n_samples = 0
        for name in data_names:
            self.n_samples += int(self.n_samples_info[name] // total_gpus)

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