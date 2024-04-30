"""Modified from the LLaVA implementation:
    https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_science.py
"""
import json
import os

from tasks.base_dataset import TaskDataset, Example
import utils


class OWLDataset(TaskDataset):
    def __init__(self, root, processor):
        self.root = root
        self.image_folder = os.path.join(root, "images")
        annotation_path = os.path.join(root, "questions.jsonl")

        with open(annotation_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line.strip("\n")) for line in f.readlines()]

        self.processor = processor
        print(f"Owl-Eval total dataset size = {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        question_id = item["question_id"]
        question = item["question"]

        imgpath = os.path.join(self.image_folder, item["image"])
        image = utils.load_image(imgpath)
        image_prompt = "Human: <image>"

        prompt = self.build_prompt(question, image_prompt)

        data = {
            "prompt": prompt,
            "question": question,
            "question_id": question_id,
            "image_path": str(item["image"]),
        }
        ex = Example(index, image, prompt, data)

        return ex
