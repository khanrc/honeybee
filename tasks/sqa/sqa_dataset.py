"""Modified from the LLaVA implementation:
    https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_science.py
"""
import json
import os
import re

from PIL import Image

from pipeline.data_utils.utils import optionize
from tasks.base_dataset import Example, TaskDataset
import utils

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


class SQADataset(TaskDataset):
    def __init__(
        self, root, processor, split="test"
    ):
        self.root = root
        self.split = split
        self.image_folder = os.path.join(root, "images", split)
        annotation_path = os.path.join(root, f"llava_{split}_QCM-LEPA.json")

        with open(annotation_path, "r") as f:
            self.data = json.load(f)

        self.processor = processor

        utils.print_rank_0(f"ScienceQA total {split} split dataset size = {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        question_id = item["id"]
        # NOTE we also can parse question (to question, context, and options) and templatize it
        question = item["conversations"][0]["value"]  # question + context
        parsed_question = parse_question(question)

        parsed_answer = parse_answer(item["conversations"][1]["value"])

        if "image" in item:
            imgpath = os.path.join(self.image_folder, item["image"])
            image = Image.open(imgpath)
            image_prompt = "Human: <image>"
        else:
            image = None
            image_prompt = None
            imgpath = None

        option, answer = optionize(parsed_question["options"], parsed_answer["answer_index"])

        question = parsed_question["question"]
        context = parsed_question["context"]
        option = option
        prompt = f"Answer with the option's letter from the given choices directly. {question}\nContext: {context}\nThere are several options:\n{option}\n"
        prompt = self.build_prompt(prompt, image_prompt)

        data = {
            "prompt": prompt,
            "question": question,
            "question_id": question_id,
            "image_path": str(imgpath),
        }

        data.update(parsed_answer)
        ex = Example(index, image, prompt, data)

        return ex
