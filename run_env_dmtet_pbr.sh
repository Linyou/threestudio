export http_proxy=http://192.168.48.17:18000
export https_proxy=http://192.168.48.17:18000
export HF_HOME="/mnt/pfs/share/pretrained_model/.cache/huggingface"
export OMP_NUM_THREADS=2
python launch.py --config configs/sd_dmtet_pbr_finetuneb_control.yaml --train --gpu 0 \
system.prompt_processor.prompt="" \
system.geometry_convert_from=""
