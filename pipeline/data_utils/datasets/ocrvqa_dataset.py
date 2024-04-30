import os
from pathlib import Path

from .base_task import BaseTaskDataset
import utils


class OCRVQATrainDataset(BaseTaskDataset):
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        input_path: str,
        image_root_path: str,
        dedup,
        split="train",
        nturn=1,  # number of QA pairs per image
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)

        splits = ["train", "val", "test"]
        split_idx = splits.index(split) + 1
        self.split = split
        self.split_index = split_idx
        self.nturn = nturn
        self.dedup = dedup

        raw_data = utils.load(input_path)
        self.dataset = self.load_data(raw_data, image_root_path)

        print(f"Load OCRVQA {split} dataset. len: {len(self.dataset)}")

    def load_data(self, raw_data, image_root_path):
        # flatten dataset & filter by split
        image_root = Path(image_root_path)
        data = []
        for key, dic in raw_data.items():
            if dic["split"] != self.split_index:
                continue

            img_url = dic["imageURL"]
            ext = os.path.splitext(img_url)[1]
            img_path = image_root / f"{key}{ext}"

            for q, a in zip(dic["questions"], dic["answers"]):
                data.append((img_path, {"question": q, "answer": a}))

        # build prompt data
        data = self.finalize_data(data, task_type="ocrvqa_vqa", nturn=self.nturn)

        return data
