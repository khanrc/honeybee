import base64
import io
import json
import os
import pickle

import pandas as pd
from PIL import Image
from transformers import logging


def dump(obj, path, mode=None, make_dir=True):
    path = str(path)
    if mode is None:
        mode = path.split(".")[-1]

    if make_dir:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    if mode == "txt":
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(obj))
    elif mode == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    elif mode in ["pkl", "pickle"]:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    elif mode in ["csv", "tsv"]:
        df = pd.DataFrame(obj)
        df.to_csv(path, sep="\t", index=False)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def load(path, mode=None):
    path = str(path)
    if mode is None:
        mode = path.split(".")[-1]

    if mode == "txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif mode == "json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif mode == "jsonl":
        with open(path, encoding="utf-8") as f_:
            lines = f_.readlines()
            lines = [x.strip() for x in lines]
            if lines[-1] == "":
                lines = lines[:-1]
            return [json.loads(x) for x in lines]
    elif mode in ["pkl", "pickle"]:
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


class transformers_log_level:
    """https://github.com/huggingface/transformers/issues/5421#issuecomment-1317784733
    Temporary set log level for transformers
    """

    orig_log_level: int
    log_level: int

    def __init__(self, log_level: int):
        self.log_level = log_level
        self.orig_log_level = logging.get_verbosity()

    def __enter__(self):
        logging.set_verbosity(self.log_level)

    def __exit__(self, type, value, trace_back):
        logging.set_verbosity(self.orig_log_level)


def get_num_params(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def check_local_file(model_name_or_path):
    """Check local file in "TRANSFORMERS_CACHE" directory
    """
    if "TRANSFORMERS_CACHE" not in os.environ:
        return False, model_name_or_path

    file_name = os.path.join(
        os.environ["TRANSFORMERS_CACHE"], f"models--{model_name_or_path.replace('/', '--')}"
    )
    local_files_only = os.path.exists(file_name)
    file_name = file_name if local_files_only else model_name_or_path
    return local_files_only, file_name
