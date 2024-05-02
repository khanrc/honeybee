CKPT=$1 # e.g., checkpoints/7B-C-Abs-M256/last
RESULTSDIR=eval_results/
GPU=auto  # use all available GPUs

torchrun --nproc_per_node=$GPU --standalone eval_tasks.py \
    --ckpt_path ${CKPT} \
    --result_dir ${RESULTSDIR} \
    --template honeybee_default \
    --config \
        mmb.yaml \
        mme.yaml \
        seed.yaml \
        llavabench.yaml \
        sqa.yaml \
        mm_vet.yaml \
        mmmu.yaml \
        pope.yaml \
