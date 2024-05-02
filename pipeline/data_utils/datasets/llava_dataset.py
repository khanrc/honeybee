import os
from typing import List, Union

import utils
from pipeline.data_utils.constants import SYSTEM_MESSAGE, ROLE_PATTERNS, MEDIA_TOKENS
from .base import BaseDataset
from .common import (
    chunking_by_keyword, remove_special_token_from_text, load_json_files
)


class LLaVAInstructDataset(BaseDataset):
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        input_files: Union[str, List[str]],
        image_root_path: str,
        image_tokens: List[str] = [
            "<image>\n",
            "\n<image>",
        ],  # special tokens for an image in this dataset
        as_instruct: bool = True,  # default to instruction tuning
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)

        self.image_tokens = image_tokens
        self.as_instruct = as_instruct
        raw_data_lst = load_json_files(input_files)
        self.dataset = self.load_data(raw_data_lst, image_root_path, SYSTEM_MESSAGE)

        utils.print_rank_0(f"Load {self.__class__.__name__}. len: {len(self.dataset)}")

    def get_dataset_name(self):
        return "llava"

    def load_data(self, raw_data, image_root_path, system_message):
        """Data is a list where each item is similar to following
        {
            "id": "005116462",
            "image": "00511/005116462.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nRender a clear and concise summary of the photo."
                },
                {
                    "from": "gpt",
                    "value": "$ 10 - cute cheap printed mini dress - khaki multicolor striped floral print peasant short sleeve tunic"
                }
            ]
        },
        """

        parsed_data = []
        for item in raw_data:
            temp_dict = {
                "image": os.path.join(image_root_path, item["image"]),
                "task_type": f"{self.get_dataset_name()}_inst",
            }

            if self.as_instruct:
                """Prepare data with instruction"""
                temp_text = system_message
                convs = item["conversations"]
                for conv in convs:
                    role = conv["from"]
                    temp_text += ROLE_PATTERNS[role]  # e.g., '\nHuman: '

                    txts = chunking_by_keyword(
                        conv["value"], keyword_patterns=self.image_tokens
                    )
                    for i, txt in enumerate(txts):
                        if txt in self.image_tokens:
                            # Convert the data-specific image token to the shared image token
                            txt = MEDIA_TOKENS["image"][0]
                            if i != 0:
                                txt = ROLE_PATTERNS["human"] + txt
                            else:
                                txt = txt + ROLE_PATTERNS["human"]
                        temp_text += txt

            else:
                """Prepare data as simple image-text pair"""
                if "conversatons" in item:
                    # there is a typo in original annotation
                    txt = item["conversatons"][1]["value"]
                else:
                    txt = item["conversations"][1]["value"]

                # prompt: [NO_PROMPT] <image> <text> where [NO_PROMPT] is an indicator
                # it will be converted to [<bos> <image> <text> <eos>]
                temp_text = (
                    "[NO_PROMPT] "
                    + MEDIA_TOKENS["image"][0]
                    + " "
                    + remove_special_token_from_text(txt, patterns=self.image_tokens)
                )

            temp_dict["text"] = temp_text
            parsed_data.append(temp_dict)

        return parsed_data
