"""Modified from the LLaVA implementation:
    https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_science.py
"""
import json
import os

from pipeline.data_utils.datasets.base_task import optionize
from pipeline.data_utils.datasets.sqa_dataset import parse_answer, parse_question
from tasks.base_dataset import Example, TaskDataset
import utils


class SQADataset(TaskDataset):
    def __init__(
        self, root, processor, template_name: str, split="test",
    ):
        self.root = root
        self.split = split
        self.image_folder = os.path.join(root, "images", split)
        annotation_path = os.path.join(root, f"llava_{split}_QCM-LEPA.json")

        with open(annotation_path, "r") as f:
            self.data = json.load(f)

        self.processor = processor
        self.set_templatizer("eval-sqa", template_name)

        utils.print_rank_0(f"ScienceQA total {split} split dataset size = {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        question_id = item["id"]
        question = item["conversations"][0]["value"]  # question + context
        parsed_question = parse_question(question)

        parsed_answer = parse_answer(item["conversations"][1]["value"])

        if "image" in item:
            imgpath = os.path.join(self.image_folder, item["image"])
            image = utils.load_image(imgpath)
            image_prompt = "Human: <image>"
        else:
            image = None
            image_prompt = None
            imgpath = None

        option, answer = optionize(parsed_question["options"], parsed_answer["answer_index"])
        parsed_answer["answer"] = answer
        parsed_data = {
            "question": parsed_question["question"],
            "context": parsed_question["context"],
            "option": option,
            "lecture": parsed_answer["lecture"],
            "solution": parsed_answer["solution"],
            "answer": answer,
        }
        prompt = self.build_prompt(parsed_data, image_prompt)

        data = {
            "prompt": prompt,
            "question": question,
            "question_id": question_id,
            "image_path": str(imgpath),
        }

        data.update(parsed_answer)
        ex = Example(index, image, prompt, data)

        return ex
