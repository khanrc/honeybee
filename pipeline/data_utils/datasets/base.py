import time
import traceback

import torch
from torch.utils.data.dataset import Dataset
import utils


class BaseDataset(Dataset):
    """Base dataset class

    Data loading process:
        (offline) init -> load_data -> (finalize_data)
        (online) __getitem__ -> process_data -> preprocess_data ->
            image_processor -> build_text_from_data -> tokenizer
    """
    def __init__(self, tokenizer, processors, max_length, **kwargs):
        super().__init__()

        self.tokenizer = tokenizer
        self.processors = processors
        self.max_length = max_length
        self.templatizer = None

        self.cluster_shuffle = kwargs.pop("cluster_shuffle", False)

        if kwargs and utils.is_main_process():
            print("=" * 80)
            print("Dataset ignore kwargs: {}".format(kwargs))
            print("=" * 80)

    def set_templatizer(self, templatizer):
        self.templatizer = templatizer

    def __len__(self):
        return len(self.dataset)

    def load_data(self):
        """Load data files and parse data samples with dataset-specific parsing logics

        The result instruction text should follow the shared format example:
            'system message'
            Human: 'prompt'
            Human: <image>
            AI: 'answer'

        Required keys in result dictionary:
            'image': pull path of an image file
            'task_type': used for selecting a processor
            NOTE templatizer parse 'examples' into 'text'; only one or the other is required.
            'text': parsed instruction text like above example
            'examples': a list of examples for template-based instruction generation

        Return:
            Parsed data list
        """
        raise NotImplementedError()

    def preprocess_data(self, data):
        """ perform pre-processing for the given data if required
        Args:
            data: datapoint given from self.dataset
        """
        return data

    def build_text_from_data(self, data):
        return data["text"]

    def process_data(self, data, processor):
        data = self.preprocess_data(data)

        # Process Image if exists
        if "image" in data and len(data["image"]) > 0:
            image_urls = data["image"]
            images = utils.load_images(image_urls)
            images = [processor(image=image) for image in images]
            images = torch.stack(images, dim=0)
        else:
            images = None

        # Process Text
        text = self.build_text_from_data(data)
        text_input = self.tokenizer.encode_prompt(
            text, self.max_length
        )

        return {
            "image": images,
            "text_raw": text,
            "text": text_input,
            "task_type": data["task_type"],
        }

    def __getitem__(self, index):
        data = self.dataset[index]
        task_type = data.get("task_type", "dummy_default").split("_")[-1]  # Get processor type
        while True:
            try:
                data = self.process_data(data, self.processors[task_type])

            except Exception as e:
                traceback.print_exc()
                print(e)
                time.sleep(0.1)
                index = 0 if index == (len(self) - 1) else index + 1
                data = self.dataset[index]
                task_type = data.get("task_type", "dummy_default").split("_")[-1]
                continue
            break

        return data
