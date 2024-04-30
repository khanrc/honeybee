from pathlib import Path

import utils
from tasks.base_dataset import Example, TaskDataset


class LlavaBenchDataset(TaskDataset):
    """LLaVA-Bench in the Wild
    """
    def __init__(self, root, processor):
        self.root = Path(root)
        self.data = utils.load(self.root / "questions.jsonl")
        self.processor = processor
        print(f"LLaVA-Bench-W total dataset size = {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dic = self.data[index]
        image_path = self.root / "images" / dic["image"]
        image = utils.load_image(image_path)
        question = dic["text"]
        category = dic["category"]
        qid = dic["question_id"]

        prompt = self.build_prompt(question)

        data = {
            "index": index,
            "prompt": prompt,
            "question": question,
            "category": category,
            "question_id": qid,
        }
        ex = Example(index, image, prompt, data)

        return ex
