import os

from .base_task import BaseTaskDataset, optionize
import utils


def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


class AOKVQADataset(BaseTaskDataset):
    """A-OKVQA dataset
    """
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        option,
        root: str = "./data/A-OKVQA",
        image_root: str = "./data/opensets_coco/images/",
        split="train",
        nturn=1,  # number of QA pairs per image
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)

        assert split in ["train", "val", "test"]
        self.split = split
        self.nturn = nturn
        self.option = option

        self.dataset = self.load_data(root, split, image_root)

        print(f"Load A-OKVQA {split} dataset. len: {len(self.dataset)}")

    def load_data(self, root, split, image_root):
        ann_path = os.path.join(root, f"aokvqa_v1p0_{split}.json")
        js = utils.load(ann_path)

        data = []
        for dic in js:
            assert split == dic['split']
            image_id = dic['image_id']
            question = dic['question']
            choices = dic['choices']
            answer_idx = dic['correct_choice_idx']
            rationales = dic['rationales']
            direct_answers = dic['direct_answers']

            image_path = get_coco_path(split, image_id, image_root)
            data.append((
                str(image_path),
                {
                    "question": question,
                    "choices": choices,
                    "answer_idx": answer_idx,
                    "rationales": rationales,
                    "direct_answers": direct_answers,
                }
            ))

        data = self.finalize_data(data, task_type="aokvqa_vqa", nturn=self.nturn)

        return data

    def process_example_online(self, ex):
        option, answer = optionize(ex["choices"], ex["answer_idx"], **self.option)
        rationale = " ".join(ex["rationales"])
        ex = {
            "question": ex["question"],
            "option": option,
            "answer": answer,
            "rationale": rationale,
        }

        return ex
