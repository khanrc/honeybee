import os
from pathlib import Path
from PIL import Image
from tasks.base_dataset import TaskDataset, Example
from pipeline.data_utils.utils import optionize
import utils


class SEEDDataset(TaskDataset):
    """SEED-Bench dataset"""
    def __init__(self, root, processor, target="image"):
        assert target in ["image", "video", "all"]
        assert target == "image", "video is not supported yet"
        root = Path(root)
        self.root = root
        self.image_root = self.root / "SEED-Bench-image"

        js = utils.load(root / "SEED-Bench.json")
        if target == "all":
            self.data = js["questions"]
        elif target == "image":
            self.data = [q for q in js["questions"] if q["question_type_id"] <= 9]
        elif target == "video":
            self.data = [q for q in js["questions"] if q["question_type_id"] > 9]
        else:
            raise ValueError(target)

        self.question_types: dict = js["question_type"]
        self.qid2type = {qid: qtype for qtype, qid in self.question_types.items()}

        self.processor = processor
        utils.print_rank_0(f"SEED-Bench total dataset size = {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dic = self.data[index]
        data_id = dic["data_id"]
        image_path = self.image_root / data_id
        image = Image.open(image_path)
        question = dic["question"]
        indices = "abcd"
        choices = [dic[f"choice_{idx}"] for idx in indices]
        answer = dic["answer"]
        answer_index = indices.index(answer.lower())
        option_str, answer_str = optionize(choices, answer_index)

        prompt = f"Answer with the option's letter from the given choices directly. {question}\nThere are several options:\n{option_str}\n"
        prompt = self.build_prompt(prompt)

        data = {
            "question": question,
            "answer": answer_str,
            "prompt": prompt,
            "choices": choices,
            "image_path": str(image_path),
            "question_id": dic["question_id"],
            "question_type_id": dic["question_type_id"],
            "question_type": self.qid2type[dic["question_type_id"]],
        }
        ex = Example(index, image, prompt, data)
        return ex
