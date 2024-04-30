import base64
import io
import json
import os
import pickle

import pandas as pd
from PIL import Image, ImageFile
from transformers import logging


ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

COCO_ROOT = "./data/opensets_coco/images/"


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
    elif mode =="jsonl":
        assert isinstance(obj, list), "obj must be a list for jsonl mode"
        lines = [json.dumps(x, ensure_ascii=False) for x in obj]
        with open(path, "w", encoding="utf8") as f:
            f.write("\n".join(lines))
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


def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")


def load_images(image_paths: list[str] | str) -> list[Image.Image]:
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    images = [load_image(image_path) for image_path in image_paths]
    return images


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = load_image(io.BytesIO(image_data))
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
