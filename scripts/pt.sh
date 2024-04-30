#!/bin/bash
config_name="pretrain"  # default

while getopts "c:" opt; do
  case "$opt" in
    c)
      config_name="$OPTARG"
      ;;
    \?)
      echo "Usage: $0 [-c config_name] ..."; exit 1;
      ;;
  esac
done

shift $((OPTIND-1))

# Pre-training
pt_output_dir="output/pt/${1}"
mkdir -p ${pt_output_dir}

echo "==================================================================================="
echo "Using config: $config_name"
echo "Output directory: $pt_output_dir"
echo "==================================================================================="

deepspeed ./train.py --config-name=${config_name} "${@:2}" output_dir=${pt_output_dir} \
  2>&1 | tee ${pt_output_dir}/train.log

