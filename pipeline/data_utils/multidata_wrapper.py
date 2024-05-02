import math
from typing import Iterable, Union

import numpy as np
from torch.utils.data import Dataset, IterableDataset
import utils


class MultiDataset(IterableDataset):
    def __init__(
        self,
        datasets: Iterable[Dataset],
        sampling_weights: Union[str, list] = "uniform",
        force_one_per_dataset: bool = False,
        batch_per_device: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()
        total_gpus = utils.get_world_size()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"

        self.ds_len_lst = [len(d) for d in self.datasets]
        self.len = sum(self.ds_len_lst) // total_gpus
        self.n_datasets = len(self.datasets)
        self.sampling_weights = self._parse_sampling_weights(sampling_weights)
        self.force_one_per_dataset = force_one_per_dataset
        self.batch_per_device = batch_per_device

        print(f"Total datasets len: {self.len * total_gpus}")
        print(f"   >>> force_one_per_dataset: {self.force_one_per_dataset}")

    def _parse_sampling_weights(self, sampling_weights):
        if sampling_weights == "uniform":
            sampling_weights = [1 / self.n_datasets] * self.n_datasets
        elif sampling_weights == "length_ratio":
            sampling_weights = self.ds_len_lst
            sampling_weights_np = np.array(sampling_weights)
            sampling_weights = sampling_weights_np / sampling_weights_np.sum()
        elif isinstance(sampling_weights, list):
            assert len(self.datasets) == len(
                sampling_weights
            ), "lengths of datasets and sampling weights should be equal."
            # normalize
            sampling_weights = [w / math.fsum(sampling_weights) for w in sampling_weights]
        else:
            raise Exception(
                "The sampling_weights should be one of ['uniform', 'length_ratio', list of weights]"
            )

        return np.array(sampling_weights)

    def __len__(self):
        return self.len

    def __iter__(self):
        datasets = [
            iter(dset) if isinstance(dset, IterableDataset) else dset
            for dset in self.datasets
        ]
        bidx = 0
        for _ in range(self.len):
            if self.force_one_per_dataset and bidx < self.n_datasets:
                # if set to force_one_per_dataset, for n_datasets = 4 and batch_size = 16,
                # for 0 <= bidx < 4, we perform round-robin selection
                # for 4 <= bidx < 16, we perform weight-based sampling
                dset_idx = bidx
            else:
                dset_idx = np.random.choice(self.n_datasets, p=self.sampling_weights)

            dset = datasets[dset_idx]
            if isinstance(dset, Iterable):
                data = next(dset)
            else:
                N = len(dset)
                di = np.random.randint(N)
                data = dset[di]

            if self.force_one_per_dataset:
                bidx = (bidx + 1) % self.batch_per_device
            yield data
