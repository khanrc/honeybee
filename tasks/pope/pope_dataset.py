import os
from pathlib import Path
from tasks.base_dataset import TaskDataset, Example
import utils


class POPEDataset(TaskDataset):
    def __init__(self, root, processor, template_name):
        root = Path(root)

        categories = [
            "adversarial",
            "popular",
            "random",
        ]

        self.data = []
        for category in categories:
            path = root / f"coco_pope_{category}.json"
            data = utils.load(path, mode="jsonl")
            for dic in data:
                dic["category"] = category
                self.data.append(dic)

        self.processor = processor
        self.set_templatizer("eval-vqa", template_name)
        utils.print_rank_0(f"POPE total dataset size = {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Example)
        # {'question_id': 1,
        # 'image': 'COCO_val2014_000000310196.jpg',
        # 'text': 'Is there a snowboard in the image?',
        # 'label': 'yes',
        # 'category': 'adversarial'}
        dic = self.data[index]
        imgpath = dic["image"]
        imgpath = os.path.join(utils.COCO_ROOT, "val2014", imgpath)
        image = utils.load_image(imgpath)

        question = dic["text"]
        prompt = self.build_prompt(question)
        answer = dic["label"]

        data = {
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "image_path": str(imgpath),
            "category": dic["category"],
        }
        ex = Example(index, image, prompt, data)
        return ex
