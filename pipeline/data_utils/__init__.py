from utils import print_rank_0

from .datasets import load_dataset
from .multidata_wrapper import MultiDataset
from .processors.builder import build_processors


def datasets_provider(data_config, tokenizer, split="train", proc_kwargs=None):
    print_rank_0(f"> building {split} datasets for MLLM ...")
    proc_kwargs = proc_kwargs or {}

    # initialize processors
    processors = build_processors(data_config["processors"][split], **proc_kwargs)

    # load datasets
    data_cfgs = data_config.get(f"{split}_dataset", None)
    if data_cfgs is not None:
        common_cfgs = data_config[f"{split}_cfg"]
        cluster_shuffle = data_config.get("cluster_shuffle", False)

        # if data_config.template is not set or data_config.template.name is None,
        # datasets do not use templatizer.
        template_cfg = data_config.get("template", {"name": None})
        template_name = template_cfg["name"]
        datasets_lst = [
            load_dataset(
                dset_name, tokenizer, processors, common_cfgs["max_length"], dt["classname"],
                template_name=template_name, cluster_shuffle=cluster_shuffle, **dt["data_cfg"],
            )
            for dset_name, dt in data_cfgs.items()
        ]

        if len(datasets_lst) > 1 or data_config.get("force_multidataset", False):
            print_rank_0(f"> Wrapping Multidataset ... (#dataset={len(datasets_lst)})")
            # wrap with Multidataset class
            dataset = MultiDataset(datasets_lst, **common_cfgs)
        else:
            print_rank_0("> Single dataset ...")
            dataset = datasets_lst[0]
        print_rank_0("> finished creating MLLM datasets ...")
    else:
        dataset = None

    return dataset, processors
