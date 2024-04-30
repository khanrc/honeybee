import os

import utils
from pipeline.data_utils.datasets.base_task import BaseTaskDataset


class VSRDataset(BaseTaskDataset):
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        input_path: str,
        image_root_path: str,
        split="train",
        nturn=1,  # number of QA pairs per image
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)

        splits = ["train", "val", "test"]
        assert split in splits
        self.split = split
        self.nturn = nturn

        raw_data = utils.load(input_path)
        self.dataset = self.load_data(raw_data, image_root_path)

        print(f"Load VSR {split} dataset. len: {len(self.dataset)}")

    def load_data(self, raw_data, image_root_path):
        # flatten dataset
        data = []
        for dic in raw_data:
            img_url = "/".join(
                dic["image_link"].split("/")[-2:]
            )  # split and img name, e.g., train2017/000000296471.jpg"
            img_path = os.path.join(image_root_path, img_url)
            question = dic["caption"]
            question_interro = dic["caption"].split("is")
            question_interro = [str_.lower().replace(".", "?").strip() for str_ in question_interro]
            question_interro = "Is " + " ".join(question_interro)

            answer = "yes" if dic["label"] == 1 else "no"

            data.append(
                (
                    str(img_path),
                    {"question": question, "question_interro": question_interro, "answer": answer},
                )
            )

        # build prompt data
        data = self.finalize_data(data, task_type="vsr_vqa", nturn=self.nturn)
        return data
