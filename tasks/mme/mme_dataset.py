from pathlib import Path
from PIL import Image
from tasks.base_dataset import TaskDataset, Example
import utils


EVAL_TYPE_DICT = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}


def load_subset(dir_path):
    root = Path(dir_path)
    dset_name = root.name

    imgpaths = list(root.glob("**/*.jpg")) + list(root.glob("**/*.png"))
    imgpaths = sorted(imgpaths)

    def get_txtpath(imgpath):
        txtpath = imgpath.with_suffix(".txt")
        txtname = txtpath.name
        if txtpath.exists():
            return txtpath

        if imgpath.parent.name == "images":
            return imgpath.parent.parent / "questions_answers_YN" / txtname

        raise ValueError(f"Cannot find txt path from image path `{imgpath}`")

    data = []
    for imgpath in imgpaths:
        txtpath = get_txtpath(imgpath)
        with txtpath.open(encoding="utf-8") as f:
            for line in f:
                q, a = line.strip().split("\t")
                data.append((dset_name, imgpath, q, a))

    return data


class MMEDataset(TaskDataset):
    def __init__(self, root, processor):
        root = Path(root)
        data = []
        for subset in EVAL_TYPE_DICT["Perception"] + EVAL_TYPE_DICT["Cognition"]:
            data += load_subset(root / subset)
        utils.print_rank_0(f"MME total dataset size = {len(data)}")
        assert len(data) == 2374

        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dset_name, imgpath, question, answer = self.data[index]

        prompt = f"Answer the question using a single word or phrase. {question}"
        prompt = self.build_prompt(prompt)

        imgid = imgpath.name
        image = Image.open(imgpath)

        data = {
            "question": question,
            "answer": answer,
            "image_path": str(imgpath),
            "image_id": imgid,
            "dataset_name": dset_name,
        }
        ex = Example(index, image, prompt, data)

        return ex
