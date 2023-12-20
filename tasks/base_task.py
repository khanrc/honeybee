import torch
import torch.distributed as dist
from tqdm import tqdm

import utils


class TaskScore:
    def __init__(self, scores):
        self.scores = scores

    def get_summary(self, max_level=1):
        raise NotImplementedError()

    def dumps(self):
        raise NotImplementedError()

    def __repr__(self):
        return self.dumps()


class Task:
    default_gen_kwargs = {}

    def __init__(self, model, tokenizer, processor, data_loader, gen_kwargs=None):
        self.model = utils.unwrap_ddp(model)
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_loader = data_loader

        if gen_kwargs is not None:
            self.default_gen_kwargs = {
                **self.default_gen_kwargs,
                **gen_kwargs,
            }

    def compute_score(self, results: dict) -> TaskScore:
        """Compute score from preds and format results if required

        Args:
            results (dict)

        Return:
            scores (TaskScore): scores for log
        """
        raise NotImplementedError()

    def reformat_results(self, results: dict) -> dict:
        """Optionally re-format results for dump

        If required, you can re-format results dictionary here.
        """
        return results

    def dump_submission_file(self, result_dir: str, results: dict):
        """Optionally parse the results into a specific format of a submission file.

        If required, you can make a submission file here.
        """

    @torch.no_grad()
    def eval_loop(self, bf16=True, progbar=False, **gen_kwargs):
        """Evaluation loop

        Args:
            bf16 (bool): whether to use bf16
            progbar (bool): whether to use progress bar (tqdm)
            gen_kwargs (dict): kwargs for model.generate
        """
        # override default gen_kwargs
        self.model.eval()
        gen_kwargs = {**self.default_gen_kwargs, **gen_kwargs}

        results = {}
        iterator = self.data_loader
        if progbar and utils.is_main_process():
            iterator = tqdm(iterator, desc=self.get_name())
        utils.barrier()

        for i, batch in enumerate(iterator):
            inputs = batch.inputs
            if bf16:
                inputs = {
                    k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()
                }
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            gens = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
            preds = self.tokenizer.batch_decode(gens, skip_special_tokens=True)

            for id, pred, data in zip(batch.ids, preds, batch.data):
                results[id] = {
                    "pred": pred,
                    **data,
                }

        return results

    def evaluate(self, bf16=True, progbar=False, **gen_kwargs):
        """Evaluate model on dataset

        Args:
            bf16: whether to use bf16. Default: True
            progbar: whether to use progress bar (tqdm). Default: False
            printf: print function. Default: None (no print)
        """
        each_result = self.eval_loop(bf16=bf16, progbar=progbar, **gen_kwargs)

        # gather results across multiple gpus
        gathered_results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_results, each_result)

        results = {}
        scores = None
        if utils.is_main_process():
            for gathered_result in gathered_results:
                results.update(gathered_result)
            # sort by key
            results = dict(sorted(results.items()))

            scores = self.compute_score(results)
            results = self.reformat_results(results)

        return scores, results

    @classmethod
    def get_name(cls):
        """Get task name (= class name)"""
        return cls.__name__
