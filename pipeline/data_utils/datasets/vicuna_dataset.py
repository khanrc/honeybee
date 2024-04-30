from typing import List, Union

from pipeline.data_utils.constants import SYSTEM_MESSAGE, ROLE_PATTERNS
from .base import BaseDataset
from .common import remove_special_token_from_text, load_json_files


class VicunaInstructDataset(BaseDataset):
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        input_files: Union[str, List[str]],
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)

        raw_data_lst = load_json_files(input_files)
        self.dataset = self.load_data(raw_data_lst, SYSTEM_MESSAGE)

        print(f"Load Vicuna instruction dataset. len: {len(self.dataset)}")

    def load_data(self, raw_data, system_message):
        parsed_data = []

        for item in raw_data:
            temp_dict = {
                "task_type": "vicuna_inst",
            }

            # Add system message
            temp_text = system_message

            # Add prompt and answers
            convs = item["conversations"]
            for i, conv in enumerate(convs):
                role = conv["from"]
                if i == 0 and role != "human":
                    continue
                temp_text += ROLE_PATTERNS[role]
                temp_text += conv["value"]

            temp_text = remove_special_token_from_text(temp_text)
            temp_dict["text"] = temp_text
            temp_dict["id"] = item["id"]
            parsed_data.append(temp_dict)

        return parsed_data
