CKPT=checkpoints/7B-C-Abs-M144/last
RESULTSDIR=eval_results/
GPU=auto  # use all available GPUs

torchrun --nproc_per_node=$GPU --standalone eval_tasks.py \
    --ckpt_path ${CKPT} \
    --result_dir ${RESULTSDIR} \
    --config \
        configs/tasks/mme.yaml \
        configs/tasks/mmb.yaml \
        configs/tasks/seed.yaml \
        configs/tasks/sqa.yaml
