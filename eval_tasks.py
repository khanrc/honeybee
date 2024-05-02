import argparse
import os
from pathlib import Path

import torch
from torch.distributed import destroy_process_group, init_process_group
from omegaconf import OmegaConf
from hydra import initialize, compose

import utils
from pipeline.interface import get_model
from pipeline.config import AttrDict
from tasks import build_task
from utils.logging import get_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="checkpoints/7B-C-Abs-M144/last",
    help="Path to the trained checkpoint.",
)
parser.add_argument(
    "--result_dir",
    type=str,
    default="eval_results/",
    help="Path to the result files.",
)
parser.add_argument("--config", nargs="+", required=True, help="Task config names.")
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
parser.add_argument("--template", type=str, default="auto", help="Template name for the evaluation.")
parser.add_argument(
    "--batch_size", "-B",
    type=int,
    default=None,
    help="Per-device batch size for evaluation. (default: use the value in the config)",
)

logger = get_logger()


def load_exp_config_with_tasks(ckpt_path: str, task_config_names: list[str]) -> AttrDict:
    # load task configs
    with initialize(version_base=None, config_path="configs/tasks", job_name="eval_tasks"):
        task_cfg = {}
        for cfg_name in task_config_names:
            cfg = compose(config_name=cfg_name)
            task_cfg |= OmegaConf.to_container(cfg)

    # override tasks and resolve with exp config
    exp_config_path = Path(ckpt_path).parent / "exp_config.yaml"
    if exp_config_path.exists():
        exp_cfg = OmegaConf.load(exp_config_path)
        exp_cfg.tasks = OmegaConf.create(task_cfg)
        exp_cfg = AttrDict.from_omegaconf(exp_cfg)  # resolve & to AttrDict
    else:
        # if exp_config does not exist, create config with task configs, without resolving.
        exp_cfg = AttrDict.from_nested_dicts({"tasks": task_cfg})
        logger.warning(f"Exp config does not exist: {exp_config_path}; task configs are not resolved.")

    return exp_cfg


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

    # DDP is not necessary for evaluation
    model.cuda()

    logger.info(" -- Init done.")
    return model, tokenizer, processor


def eval_single(
    model,
    tokenizer,
    processor,
    task_config,
    template_name,
    result_dir,
    load_results=False,
    dump_submission_file=False,
):
    if args.batch_size is not None:
        task_config.dataloader.batch_size = args.batch_size

    if template_name != "auto" and "template_name" in task_config.dataset:
        task_config.dataset.template_name = template_name

    if utils.is_main_process():
        print("=" * 80)
        print(task_config.dumps())
        print("=" * 80)

    task = build_task(model, tokenizer, processor, task_config)
    task_name = task.get_name()

    if not load_results:
        scores, results = task.evaluate(progbar=True)
    else:
        result_path = os.path.join(result_dir, f"{task_name}_prediction_results_all.json")
        results = utils.load(result_path)
        if utils.is_main_process():
            scores = task.compute_score(results)
        else:
            scores = None

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
            print(f" -- Dump submission file to `{result_dir}`.")

        # reformat summary
        summary = scores.get_summary(max_level=2)
        name = task.get_name()
        summary = {f"{name}/{k}": v for k, v in summary.items()}

    return summary


def eval(model, tokenizer, processor, args):
    exp_cfg = load_exp_config_with_tasks(args.ckpt_path, args.config)
    assert len(exp_cfg.tasks) == len(args.config)

    summaries = {}
    for cfg_name, task_cfg in zip(args.config, exp_cfg.tasks.values()):
        utils.barrier()
        if utils.is_main_process():
            print(f"Evaluate {cfg_name} ...")

        cur_summary = eval_single(
            model,
            tokenizer,
            processor,
            task_cfg,
            args.template,
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
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.template and args.template.lower() in ["none", "null"]:
        args.template = None

    if utils.is_main_process():
        print(args)

    dist_setup()
    model, tokenizer, processor = init(args.ckpt_path, args.load_results)
    eval(model, tokenizer, processor, args)
    destroy_process_group()
