from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.dist import is_dist_avail_and_initialized

from .mmb import MMBDataset, MMBTask
from .mme import MMEDataset, MMETask
from .owl_eval import OWLDataset, OWLTask
from .pope import POPEDataset, POPETask
from .seed import SEEDDataset, SEEDTask
from .sqa import SQADataset, SQATask
from .llavabench import LlavaBenchDataset, LlavaBenchTask
from .mmmu import MMMUDataset, MMMUTask
from .mm_vet import MMVetDataset, MMVetTask

TASK_LIST = [
    MMETask,
    SQATask,
    MMBTask,
    OWLTask,
    POPETask,
    SEEDTask,
    LlavaBenchTask,
    MMMUTask,
    MMVetTask
]
TASK_DICT = {c.__name__: c for c in TASK_LIST}

DATASET_LIST = [
    MMEDataset,
    SQADataset,
    MMBDataset,
    OWLDataset,
    POPEDataset,
    SEEDDataset,
    LlavaBenchDataset,
    MMMUDataset,
    MMVetDataset,
]
DATASET_DICT = {c.__name__: c for c in DATASET_LIST}


def build_dataset(name, processor, **kwargs):
    dataset_class = DATASET_DICT[name]
    dataset = dataset_class(processor=processor, **kwargs)
    return dataset


def build_dataloader(dataset, batch_size, num_workers):
    collate_fn = dataset.collate_fn
    sampler = (
        DistributedSampler(dataset, shuffle=False) if is_dist_avail_and_initialized() else None
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader


def build_task(model, tokenizer, processor, task_config, debug=False):
    # build dataset
    dataset = build_dataset(processor=processor, **task_config.dataset)
    loader = build_dataloader(dataset, **task_config.dataloader)

    # build task
    task_name = task_config.name
    task_class = TASK_DICT[task_name]
    task = task_class(model, tokenizer, processor, loader, task_config.gen_kwargs, debug=debug)

    return task
