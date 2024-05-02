from pathlib import Path

from .base_task import BaseTaskDataset
import utils


class GQADataset(BaseTaskDataset):
    """GQA dataset
    """
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        dedup,
        root: str = "./data/GQA",
        split="train",
        balanced=True,
        nturn=1,  # number of QA pairs per image
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)

        assert split in ["train", "val", "test"]
        self.split = split
        self.nturn = nturn
        self.dedup = dedup

        self.dataset = self.load_data(root, split, balanced)

        print(f"Load GQA {split} dataset. len: {len(self.dataset)}")

    def load_data(self, root, split, balanced):
        if not balanced:
            raise NotImplementedError("GQA only supports balanced annotations (1M) for now.")

        root = Path(root)
        if split == "train" and balanced:
            # `minimal` version has only (imageId, question, answer, fullAnswer) informations,
            # enabling fast loading of annotation.
            ann_path = root / "annotations" / "train_balanced_questions_minimal.json"
        else:
            ann_path = root / "annotations" / f"{split}_balanced_questions.json"
        js = utils.load(ann_path)

        data = []
        for dic in js.values():
            img_id = dic["imageId"]
            img_path = root / "images" / f"{img_id}.jpg"

            # example)
            # Q: Is the sky dark?
            # A: yes
            # FullA: Yes, the sky is dark.
            question = dic["question"]
            answer = dic["answer"]
            full_answer = dic["fullAnswer"]

            data.append((
                str(img_path),
                {
                    "question": question,
                    "answer": answer,
                    "full_answer": full_answer,
                }
            ))

        data = self.finalize_data(data, task_type="gqa_vqa", nturn=self.nturn)

        return data
