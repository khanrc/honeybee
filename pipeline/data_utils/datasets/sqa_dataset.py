import os
import re
from typing import List, Union

from pipeline.data_utils.datasets.common import load_json_files
from pipeline.data_utils.datasets.base_task import BaseTaskDataset, optionize


def parse_question(question):
    """<question>\nContext: <context>\nOptions: (A) <option1> (B) <option2> ..."""
    pattern = r"(?P<question>.+?)\nContext: (?P<context>.*?)\nOptions: (?P<options>.+)"

    # re.DOTALL allows regex pattern `.` to match newlines
    match = re.search(pattern, question, re.DOTALL)
    assert match, f"Question {question} cannot be parsed."

    # remove <image>
    option_str = match.group("options").replace("<image>", "").strip()
    options = re.split(r"\([A-Z]\)", option_str)
    options = [option.strip() for option in options if option.strip()]

    parsed = {
        "question": match.group("question"),
        "context": match.group("context"),
        "options": options,
    }
    return parsed


def parse_answer(answer):
    """Parse an answer string into three groups corresponding to LECTURE, SOLUTION, and ###\nANSWER

    Args:
        answer (String): Sentences consisted of lecture, soution, and answer as one sequence.

    Returns:
        Dict: Parsed sentence dict.
    """
    _ANSWER_PATTERN_DICT = {
        "LECTURE:": "lecture",
        "SOLUTION:": "solution",
        "###\nANSWER:": "answer",
    }

    pattern = re.compile("|".join(_ANSWER_PATTERN_DICT.keys()))
    parts = pattern.split(answer)
    keys = pattern.findall(answer)

    parsed_dict = {value: None for value in _ANSWER_PATTERN_DICT.values()}
    for key, part in zip(keys, parts[1:]):
        parsed_dict[_ANSWER_PATTERN_DICT[key]] = part.lstrip()

    answer_index = ord(parsed_dict["answer"].rstrip(".")) - ord("A")
    parsed_dict["answer_index"] = answer_index

    return parsed_dict


class SQAInstructDataset(BaseTaskDataset):
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        option,
        input_files: Union[str, List[str]],
        image_root_path: str,
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)
        self.option = option

        raw_data_lst = load_json_files(input_files)
        self.dataset = self.load_data(raw_data_lst, image_root_path)

        print(f"Load SQA instruction train split dataset. len: {len(self.dataset)}")

    def load_data(self, raw_data, image_root_path):
        parsed_data = []
        for item in raw_data:
            convs = item["conversations"]
            image = os.path.join(image_root_path, item["image"]) if "image" in item else None
            # question, context, options
            parsed_question = parse_question(convs[0]["value"])
            # lecture, solution, answer
            parsed_answer = parse_answer(convs[1]["value"])
            parsed_data.append(
                (
                    image,
                    {
                        "question": parsed_question["question"],
                        "context": parsed_question["context"],
                        "options": parsed_question["options"],
                        "lecture": parsed_answer["lecture"],
                        "solution": parsed_answer["solution"],
                        "answer_index": parsed_answer["answer_index"],
                    },
                )
            )

        parsed_data = self.finalize_data(parsed_data, task_type="sqa_vqa")

        return parsed_data

    def process_example_online(self, ex):
        option, answer = optionize(ex["options"], ex["answer_index"], **self.option)
        ex = {
            "question": ex["question"],
            "context": ex["context"],
            "option": option,
            "answer": answer,
            "lecture": ex["lecture"],
            "solution": ex["solution"],
        }

        return ex
