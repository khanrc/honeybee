#!/bin/bash

config_name="finetune"  # default

while getopts "p:c:" opt; do
  case "$opt" in
    p)
      ckpt="$OPTARG"
      ;;
    c)
      config_name="$OPTARG"
      ;;
    \?)
      echo "Usage: $0 [-p pretrained_ckpt] [-c config_name] ..."; exit 1;
      ;;
  esac
done

if [ ! -e "$ckpt" ]; then
  echo "Pretrained checkpoint not found: $ckpt"
  exit 1
fi

shift $((OPTIND-1))

ft_output_dir="output/ft/$1"
mkdir -p ${ft_output_dir}

echo "==================================================================================="
echo "Using config: $config_name"
echo "Using pretrained checkpoint: $ckpt"
echo "Output directory: $ft_output_dir"
echo "==================================================================================="

deepspeed train.py \
  --config-name=${config_name} output_dir="${ft_output_dir}" pretrained_ckpt="${ckpt}" "${@:2}" \
  2>&1 | tee "${ft_output_dir}/train.log"

