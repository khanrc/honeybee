"""Modified from the LLaVA implementation:
    https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_science.py
"""

import os

import pandas as pd

from tasks.base_dataset import Example, TaskDataset
from utils.misc import decode_base64_to_image
import utils


class MMBDataset(TaskDataset):
    def __init__(self, root, processor, template_name: str, split="dev", sys_prompt="There are several options:", template_pattern="mmb"):
        self.root = root
        self.split = split
        self.sys_prompt = sys_prompt
        annotation_path = os.path.join(root, f"mmbench_{split}_20230712.tsv")

        self.df = pd.read_csv(annotation_path, sep="\t")
        self.processor = processor
        self.set_templatizer(template_pattern, template_name)

        utils.print_rank_0(f"MMBench total {split} dataset size = {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return "N/A"

    def __getitem__(self, idx):
        # Step 1: parse available information.
        index = self.df.iloc[idx]["index"]
        image = decode_base64_to_image(self.df.iloc[idx]["image"])
        question = self.df.iloc[idx]["question"]
        answer = (
            self.df.iloc[idx]["answer"] if "answer" in self.df.columns else "None"
        )  # dummy for test split
        category = self.df.iloc[idx]["category"]
        l2_category = self.df.iloc[idx]["l2-category"]

        option_candidate = ["A", "B", "C", "D", "E"]
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = f"{self.sys_prompt}\n"
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"

        hint = self.load_from_df(idx, "hint")

        # Step2: construct prompt for model input
        prompt = question
        prompt += f"\nContext: {hint}"
        prompt += f"\n{options_prompt}"
        prompt = self.build_prompt(prompt)

        data = {
            "prompt": prompt,
            "question": question,
            "hint": hint,
            "index": int(index),
            "answer": answer,
            "category": category,
            "l2_category": l2_category,
            "options_dict": options,
        }

        ex = Example(idx, image, prompt, data)

        return ex
