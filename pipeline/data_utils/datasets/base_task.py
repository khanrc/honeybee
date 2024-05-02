import random
from collections import defaultdict
from .base import BaseDataset
from typing import Union


def idx2option(idx: int, style="upper", deco="dot"):
    """
    idx: [0, N-1]
    style: upper, lower, num
    deco: None, paren, dot, rparen
    """
    idx = {
        "upper": chr(ord("A") + idx),
        "lower": chr(ord("a") + idx),
        "num": f"{idx + 1}",
    }[style]

    idx = {
        None: "{idx}",
        "paren": "({idx})",
        "dot": "{idx}.",
        "rparen": "{idx})",
    }[deco].format(idx=idx)

    return idx

# option augmentation
# - option shuffle / index
# - optionize boolean
# - N/A aug...?
def optionize(
    options: list[str],
    answer_idx: int,
    shuffle=False,
    aug_idx_style=False,
    include_answer_str=False,
    sep="\n"
) -> (str, str):
    """Convert options (list of str) to option string.
    This process also includes:
    - option shuffling
    - index augmentation

    Args:
        options (list[str])
        answer_idx (int)
        shuffle (bool): shuffle options
        aug_idx_style (bool): randomly choose index style
            Aug examples: (1) / 1. / (A) / A.
        include_answer_str (bool): include answer string
            False: A
            True: A. {answer}

    Return:
        (option_str, answer_str)
    """
    if isinstance(options, str):
        # already optionized
        return options

    answer = options[answer_idx]
    if shuffle:
        random.shuffle(options)
        answer_idx = options.index(answer)

    if not aug_idx_style:
        style = "upper"
        deco = "dot"
    else:
        style = random.choice(["upper", "lower", "num"])
        deco = random.choice(["paren", "dot", "rparen"])

    indices = [idx2option(i, style=style, deco=deco) for i in range(len(options))]
    answer_str = idx2option(answer_idx, style=style, deco=None)
    if include_answer_str:
        answer_str = f"{answer_str}. {answer}"

    options_with_index = [
        f"{idx} {option}"
        for idx, option in zip(indices, options)
    ]
    option_str = sep.join(options_with_index)
    return option_str, answer_str


def dedup(cluster: list[dict], key="answer", whitelist=None, online=False) -> list[dict | list]:
    """Deduplicate cluster by answer.
    if online=True, we just return list of data (will be deduplicated in build_text_from_data function)
    """
    if len(cluster) == 1:
        return cluster

    if whitelist == "all":
        return cluster

    dup = defaultdict(list)
    for dic in cluster:
        dedup_value = dic[key]
        dup[dedup_value].append(dic)
    dup = dict(dup)

    ret = []
    for dedup_value, dics in dup.items():
        if len(dics) == 1 or (whitelist and dedup_value in whitelist):
            ret += dics
        else:
            if not online:
                ret.append(dics[0])
            else:
                # if online, we return dics (list of dict).
                # at the __getitem__() of dataset loader, random sampling will be performed.
                ret.append(dics)

    return ret


class BaseTaskDataset(BaseDataset):
    """Base dataset for task datasets (VQAs, ...)
    """
    def finalize_data(self, raw_data: Union[list, dict], task_type="sft", nturn=1) -> list:
        """Convert raw_data to required form, supporting multi-turn format.

        Args:
            raw_data (list or dict): raw_data can be list or dict (clustered)
                list example: [
                    (image_path, {"question": question, "answer": answer, ...}),
                    ...
                ]
                dict example (clustered): {
                    image_path: [
                        {"question": question, "answer": answer, ...},
                        ...
                    ],
                    ...
                }
            nturn: the number of examples per image
                (1: single-turn; -1: multi-turn using all examples)

        Return:
            data (list)
        """
        if isinstance(raw_data, list):
            clusters = defaultdict(list)
            for image_path, ex in raw_data:
                clusters[image_path].append(ex)
        elif isinstance(raw_data, dict):
            clusters = raw_data
        else:
            raise TypeError(f"raw_data must be list or dict, but got {type(raw_data)}")

        # NOTE dedup is only applied for multi-turn
        if nturn > 1 and hasattr(self, "dedup") and self.dedup is not False:
            # deduplicate clusters
            assert isinstance(self.dedup, dict), "dedup config must be dict"
            dedup_key = self.dedup.key
            whitelist = self.dedup.whitelist
            online = self.dedup.get("online", True)
            if whitelist != "all":
                clusters = {
                    image_path: dedup(cluster, key=dedup_key, whitelist=whitelist, online=online)
                    for image_path, cluster in clusters.items()
                }

        if self.cluster_shuffle:
            # cluster shuffle
            for image_path, cluster in clusters.items():
                random.shuffle(cluster)

        data = []
        for image_path, examples in clusters.items():
            N = len(examples)
            step = nturn if nturn > 0 else N
            for i in range(0, N, step):
                item = {
                    "examples": examples[i:i+step],
                    "task_type": task_type,
                }
                if image_path is not None:
                    item["image"] = str(image_path)
                data.append(item)

        return data

    def process_example_online(self, example) -> dict:
        return example

    def build_text_from_data(self, data):
        examples = [
            self.process_example_online(
                example if not isinstance(example, list) else random.choice(example)
            )
            for example in data["examples"]
        ]

        if "image" in data:
            text = self.templatizer(examples)
        else:
            # text-only case
            text = self.templatizer(examples, image_prompt=None)
        return text

    def preprocess_bbox(self, x, y, w, h, W, H, bbox_format="normalize", bbox_coord_style=3):
        if bbox_format == "normalize":
            # normalized format from pink
            x1 = x / W
            y1 = y / H
            x2 = (x + w) / W
            y2 = (y + h) / H
        else:
            # return without preprocessing
            x1, y1, x2, y2 = x, y, x + w, y + h

        def format_coord(x):
            if bbox_coord_style == 2:
                return f"{x:.2f}"
            elif bbox_coord_style == 3:
                return f"{x:.3f}"
            else:
                raise ValueError(f"Invalid bbox_coord_style: {bbox_coord_style}")

        # formatting bbox
        bbox = [format_coord(v) for v in [x1, y1, x2, y2]]
        bbox_str = ",".join(bbox)
        bbox_str = f"[{bbox_str}]"
        return bbox_str

