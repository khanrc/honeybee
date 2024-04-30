"""MMMU evaluation, following official llava-1.5 evaluation script
Ref: https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/run_llava.py
"""
import utils
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from tasks.base_dataset import TaskDataset, Example

from pipeline.data_utils.templates import Templatizer
from .mmmu_utils.data_utils import process_single_sample, CAT_SHORT2LONG

logger = utils.get_logger()
logger.disabled = False


class MMMUDataset(TaskDataset):
    def _save_cache(self, root, data, split):
        # save cache
        logger.info("Saving fast-cache by splitting image/metadata ...")
        image_root = root / "images"
        image_root.mkdir(exist_ok=True, parents=True)

        # sanity check: check all of data["id"] is unique
        ids = [ex["id"] for ex in data]
        assert len(ids) == len(set(ids))

        metadata = []
        for ex in tqdm(data):
            ex_id = ex["id"]
            for i in range(1, 8):
                image_key = f"image_{i}"
                image = ex[image_key]
                if image is None:
                    break

                image_id = f"{ex_id}__{image_key}.{image.format.lower()}"
                image.save(image_root / image_id)
                ex[image_key] = image_id
            metadata.append(ex)

        utils.dump(metadata, root / f"{split}.json")
        logger.info(f" -- MMMU {split} set is successfully cached.")

    def load_dataset(self, split: str, root):
        # check image/metadata splitted cache exists
        cache_path = root / f"{split}.json"
        if cache_path.exists():
            data = utils.load(cache_path)  # splitted cache
            return data

        # Load MMMU from Huggingface dataset
        sub_dataset_list = []
        N = len(CAT_SHORT2LONG)
        for i, subject in enumerate(CAT_SHORT2LONG.values()):
            sub_dataset = load_dataset("MMMU/MMMU", subject, split=split)
            sub_dataset_list.append(sub_dataset)
            logger.info(f"[{i+1}/{N}] Loaded {subject} dataset (size = {len(sub_dataset)})")

        # merge all dataset
        data = concatenate_datasets(sub_dataset_list)

        # save cache
        self._save_cache(root, data, split)

        return data

    def __init__(self, root, split, processor, template_name):
        root = Path(root)
        self.root = root
        self.split = split
        self.processor = processor

        logger.info("Load MMMU datasets ...")
        self.data = self.load_dataset(split, root)
        logger.info(f"Total MMMU dataset size = {len(self.data)}")

        self.templatizer_mc = Templatizer.from_names(template_name, "mmb")
        self.templatizer_short = Templatizer.from_names(template_name, "eval-vqa")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Example:
        sample = self.data[index]
        sample = process_single_sample(sample)

        question = sample["question"]  # we can remove <image 1>
        question_type = sample["question_type"]
        options = eval(sample["options"])
        answer = sample["answer"]

        image = sample["image"]
        if isinstance(image, str):
            image = utils.load_image(self.root / "images" / image)
        else:
            image = image.convert("RGB")

        all_choices = []
        if question_type == "multiple-choice":
            option_str = "\nThere are several options:\n"
            for i, option in enumerate(options):
                index_chr = chr(ord("A") + i)
                option_str += f"{index_chr}. {option}\n"
                all_choices.append(index_chr)

            question += option_str
            prompt = self.templatizer_mc([{"question": question}])
        else:
            prompt = self.templatizer_short([{"question": question}])

        data = {
            "question": question,
            "answer": answer,
            "id": sample["id"],
            "prompt": prompt,
            "question_type": question_type,
            "all_choices": all_choices,  # for random selection
            "num_option_images": sample["num_option_images"],
        }

        ex = Example(index, image, prompt, data)
        return ex
