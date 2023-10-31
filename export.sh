export http_proxy=http://192.168.48.17:18000
export https_proxy=http://192.168.48.17:18000
export HF_HOME="/mnt/pfs/share/pretrained_model/.cache/huggingface"
export OMP_NUM_THREADS=2
python launch.py --config configs/mvdream-sd21-vast-rgb-normal-dmtet-export.yaml --export --gpu 0 \
system.prompt_processor.prompt="a green Hulk dressed in spiderman suit" \
system.geometry_convert_from="/mnt/pfs/users/liuxuebo/project/threestudio_unidream2/outputs/rgb_normal_label_normal_40_2/mvdream/a_green_Hulk_dressed_in_spiderman_suit@20231031-200712/ckpts/epoch=0-step=5000.ckpt"
