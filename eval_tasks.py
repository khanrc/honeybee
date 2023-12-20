import argparse
import os

import torch
from sconf import Config
from torch.distributed import destroy_process_group, init_process_group

import utils
from pipeline.interface import get_model
from tasks import build_task
from utils.logging import get_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt_path",
    type=str,
    help="Path to the trained checkpoint.",
)
parser.add_argument(
    "--result_dir",
    type=str,
    default="eval_results/",
    help="Path to the result files.",
)
parser.add_argument("--config", nargs="+", required=True, help="Task configs.")
parser.add_argument(
    "--load_results",
    action="store_true",
    help="Load saved results without model inference. Only for the results without re-formatted.",
)
parser.add_argument(
    "--dump_submission_file",
    action="store_true",
    help="Dump a submission file with a specific format to evaluate on a evaluation server.",
)
parser.add_argument(
    "--batch_size", "-B",
    type=int,
    default=None,
    help="Per-device batch size for evaluation. (default: use the value in the config)",
)

logger = get_logger()


def dist_setup():
    # Expected to use torchrun
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def init(ckpt_path, load_results=False):
    if load_results:
        logger.info("Skip init model in load_results mode.")
        return None, None, None

    logger.info("Init (load model, tokenizer, processor) ...")

    # create model
    model, tokenizer, processor = get_model(ckpt_path)
    model.cuda()

    logger.info(" -- Init done.")
    return model, tokenizer, processor


def eval_single(
    model,
    tokenizer,
    processor,
    config_path,
    result_dir,
    load_results=False,
    dump_submission_file=False,
):
    task_config = Config(config_path)
    task_config = next(iter(task_config.values()))  # get first child
    if args.batch_size is not None:
        task_config.dataloader.batch_size = args.batch_size

    if utils.is_main_process():
        print("=" * 80)
        print(Config(task_config).dumps())
        print("=" * 80)

    task_name = task_config.name
    task = build_task(model, tokenizer, processor, task_config)

    if not load_results:
        scores, results = task.evaluate(progbar=True)
    else:
        result_path = os.path.join(result_dir, f"{task_name}_prediction_results_all.json")
        results = utils.load(result_path)
        scores = task.compute_score(results)

    summary = {}
    if utils.is_main_process():
        print(scores)

        # write scores & results
        score_path = os.path.join(result_dir, f"{task_name}_scores.txt")
        utils.dump(scores, score_path)

        if not load_results:
            result_path = os.path.join(result_dir, f"{task_name}_prediction_results_all.json")
            utils.dump(results, result_path)

        if dump_submission_file:
            task.dump_submission_file(result_dir, results)

        # reformat summary
        summary = scores.get_summary(max_level=2)
        name = task.get_name()
        summary = {f"{name}/{k}": v for k, v in summary.items()}

    return summary


def eval(model, tokenizer, processor, args):
    summaries = {}
    for config in args.config:
        utils.barrier()
        if utils.is_main_process():
            print(f"Evaluate {config} ...")

        cur_summary = eval_single(
            model,
            tokenizer,
            processor,
            config,
            args.result_dir,
            args.load_results,
            args.dump_submission_file,
        )

        # integrate summaries
        summaries.update(cur_summary)

    if utils.is_main_process():
        print("=" * 80)
        print("Summary:")
        for k, v in summaries.items():
            print(f"{k}: {v:.2f}")


if __name__ == "__main__":
    args = parser.parse_args()
    if utils.is_main_process():
        print(args)

    dist_setup()
    model, tokenizer, processor = init(args.ckpt_path, args.load_results)
    eval(model, tokenizer, processor, args)
    destroy_process_group()
