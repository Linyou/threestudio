export http_proxy=http://192.168.48.17:18000
export https_proxy=http://192.168.48.17:18000
export HF_HOME="/mnt/pfs/share/pretrained_model/.cache/huggingface"
export OMP_NUM_THREADS=2
python launch.py --config configs/mvdream-sd21-vast-rgb-normal-v2.yaml --train --gpu 0 \
system.prompt_processor.prompt="a green Hulk dressed in spiderman suit"
