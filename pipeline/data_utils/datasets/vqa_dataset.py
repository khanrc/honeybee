from pathlib import Path

from .base_task import BaseTaskDataset
import utils


def build_vqa_dataset(root, split="train"):
    assert split in ["train", "val"]
    root = Path(root)
    q_path = root / f"v2_OpenEnded_mscoco_{split}2014_questions.json"
    a_path = root / f"v2_mscoco_{split}2014_annotations.json"

    qjs = utils.load(q_path)
    ajs = utils.load(a_path)

    questions = qjs["questions"]
    annotations = ajs["annotations"]
    assert len(questions) == len(annotations)
    assert qjs["data_subtype"] == ajs["data_subtype"]

    data_subtype = qjs["data_subtype"]

    data = []
    for q, a in zip(questions, annotations):
        assert q["question_id"] == a["question_id"]
        assert q["image_id"] == a["image_id"]

        image_id = q["image_id"]
        img_fn = 'COCO_' + data_subtype + '_'+ str(image_id).zfill(12) + '.jpg'
        img_path = root / "images" / img_fn

        data.append((
            str(img_path),
            {
                "question": q["question"],
                "answer": a["multiple_choice_answer"],
            },
        ))

    return data


class VQADataset(BaseTaskDataset):
    """VQAv2 dataset"""
    def __init__(
        self,
        tokenizer,
        processors,
        max_length,
        dedup,
        root: str = "./data/VQAv2",
        split="train",
        nturn=1,  # number of QA pairs per image
        **kwargs,
    ):
        super().__init__(tokenizer, processors, max_length, **kwargs)

        assert split in ["train", "val"]
        self.split = split
        self.nturn = nturn
        self.dedup = dedup

        self.dataset = self.load_data(root, split)

        print(f"Load VQAv2 {split} dataset. len: {len(self.dataset)}")

    def load_data(self, root, split):
        raw_data = build_vqa_dataset(root, split)
        data = self.finalize_data(raw_data, task_type="vqa_vqa", nturn=self.nturn)

        return data
