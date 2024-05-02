import os.path as osp
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional
import time

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer, PrinterCallback, ProgressCallback
from transformers.training_args import TrainingArguments
from tqdm import tqdm

import utils
from pipeline.utils import get_cosine_schedule_with_warmup, seed_worker
from pipeline.collate import batchify


class CustomPrinterCallback(PrinterCallback):
    """Printer callback for BC run"""
    def on_train_begin(self, args, state, control, **kwargs):
        """
        Event called at the beginning of training.
        """
        super().on_train_begin(args, state, control, **kwargs)
        self._start_time = time.time()
        self.logger = utils.get_logger()

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            gstep = state.global_step
            maxstep = state.max_steps

            # Compute ETA and make tqdm-like ETA string
            elapsed = time.time() - self._start_time
            tqdm_meter = tqdm.format_meter(gstep, maxstep, elapsed, ncols=0)

            self.logger.info(f"{tqdm_meter} | {logs}")


class CustomProgressCallback(ProgressCallback):
    """Progress callback for local run"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            gstep = state.global_step
            log_str = f"[Step {gstep}] {logs}"
            self.training_bar.write(log_str)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    min_lr: float = 0.0


class CustomTrainer(Trainer):
    def __init__(
        self,
        config,
        tokenizer,
        tasks,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.can_return_loss = True
        self.truncation_on_batchify = config.data_config.truncation_on_batchify
        self.max_length = config.data_config.train_cfg.max_length

        # a flag whether evaluation on benchmark (metric) or validation data (loss)
        self.eval_on_task = config.train.eval_on_task
        self.tasks = tasks

        self.config = config
        self.tokenizer = tokenizer

    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        sampler = DistributedSampler(dataset) if not isinstance(dataset, IterableDataset) else None
        collate_fn = partial(
            batchify,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            use_trunc=self.truncation_on_batchify,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
        )

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        sampler = DistributedSampler(dataset, shuffle=False)
        collate_fn = partial(
            batchify,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            use_trunc=self.truncation_on_batchify,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if self.model.proj_type == "d-abs":
            # Cast the type of ddetr abstractor to float32 because ddetr does not support fp16 or bf16.
            self.model.abstractor.to(torch.float)

        output_metric = None
        if self.eval_dataset is not None:
            # if given evaluation dataset for computing validation loss,
            # we call ORIGINAL evaluate() function
            output_metric = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if self.eval_on_task:
            for task in self.tasks:
                # if evaluate on benchmark, we use task class
                task_name = task.get_name()
                scores, results = task.evaluate(progbar=True)

                if self.is_world_process_zero():
                    print(scores.dumps(), flush=True)
                    self.log({f"{task_name}/" + k: v for k, v in scores.get_summary().items()})

                    # dump results
                    gstep = self.state.global_step
                    output_dir = osp.join(self.config.output_dir, f"eval_results/step={gstep}")
                    utils.dump(scores, osp.join(output_dir, f"{task_name}_scores.txt"))
                    utils.dump(
                        results,
                        osp.join(output_dir, f"{task_name}_prediction_results_all.json"),
                    )

            # for safe setting of training mode
            self.model.train()

        return output_metric

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either
        before this method is called or passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L1081-L1097
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_type == "cosine":
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    lr=self.args.learning_rate,
                    min_lr=self.args.min_lr,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
                self._created_lr_scheduler = True
            else:
                assert (
                    self.args.min_lr == 0.0
                ), "min_lr > 0. is only supported for cosine scheduler."
                return super(num_training_steps, optimizer)

        return self.lr_scheduler
