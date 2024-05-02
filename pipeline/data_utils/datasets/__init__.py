from ..templates import Templatizer
from .aokvqa_dataset import AOKVQADataset
from .gqa_dataset import GQADataset
from .llava_dataset import LLaVAInstructDataset
from .ocrvqa_dataset import OCRVQATrainDataset
from .pt_webdataset import PretrainWebdataset
from .refexploc_dataset import RefExpLocDataset
from .sqa_dataset import SQAInstructDataset
from .vgloc_dataset import VGLocDataset
from .vicuna_dataset import VicunaInstructDataset
from .vqa_dataset import VQADataset
from .vsr_dataset import VSRDataset

DATASET_CLASS_LIST = [
    LLaVAInstructDataset,
    PretrainWebdataset,
    SQAInstructDataset,
    VicunaInstructDataset,
    OCRVQATrainDataset,
    VQADataset,
    AOKVQADataset,
    GQADataset,
    VGLocDataset,
    VSRDataset,
    RefExpLocDataset,
]
DATASET_CLASS_DICT = {c.__name__: c for c in DATASET_CLASS_LIST}


def load_dataset(
    dset_name, tokenizer, processors, max_length, class_name: str, template_name: str, **kwargs
):
    dataset_class = DATASET_CLASS_DICT[class_name]
    dset = dataset_class(tokenizer, processors, max_length, **kwargs)

    if template_name is not None:
        templatizer = Templatizer.from_names(template_name, dset_name)
        dset.set_templatizer(templatizer)

    if isinstance(dset, PretrainWebdataset):
        dset = dset.setup()

    return dset
