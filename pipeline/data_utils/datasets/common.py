import re
import json
from typing import Union
from pipeline.data_utils.constants import MEDIA_TOKENS


def load_json_files(input_files: Union[list[str], str], key_pattern=None):
    raw_data_lst = []

    if isinstance(input_files, str):
        input_files = [input_files]

    for input_file in input_files:
        with open(input_file, "r") as f:
            data_temp = json.load(f)
        if key_pattern is not None:
            data_temp = data_temp[key_pattern]
        raw_data_lst.extend(data_temp)

    return raw_data_lst


def chunking_by_keyword(txt, keyword_patterns=["<image>\n", "\n<image>"]):
    pattern = "|".join(map(re.escape, keyword_patterns))
    chunk_strs = re.split(f"({pattern})", txt)
    chunk_strs = [x for x in chunk_strs if len(x) > 0]

    return chunk_strs


def remove_special_token_from_text(txt, patterns=None):
    if patterns is not None:
        for pattern in patterns:
            if pattern in txt:
                txt = txt.replace(pattern, "")

    # if a special media token in the conversation, replace it to a non-special token.
    for v in MEDIA_TOKENS.values():
        for v_ in v:
            txt = txt.replace(v_, "".join([c for c in v_ if c not in ["<", ">"]]))

    return txt
