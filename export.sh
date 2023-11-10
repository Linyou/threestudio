export http_proxy=http://192.168.48.17:18000
export https_proxy=http://192.168.48.17:18000
export HF_HOME="/mnt/pfs/share/pretrained_model/.cache/huggingface"
export OMP_NUM_THREADS=2
# python launch.py --config configs/mvdream-sd21-vast-rgb-normal-dmtet-export.yaml --export --gpu 0 \
# system.prompt_processor.prompt="a green Hulk dressed in spiderman suit" \
# system.geometry_convert_from="/mnt/pfs/users/liuxuebo/project/threestudio_unidream2/outputs/rgb_normal_label_normal_40_2/mvdream/a_green_Hulk_dressed_in_spiderman_suit@20231031-200712/ckpts/epoch=0-step=5000.ckpt"

python launch.py --config configs/sd_dmtet_pbr.yaml --export --gpu 0 \
system.prompt_processor.prompt="A bulldog wearing a black pirate hat" \
system.prompt_processor_II.prompt="A bulldog wearing a black pirate hat" \
system.exporter.texture_size=2048 \
system.exporter.save_normal=true \
system.exporter.texture_format=png \
system.geometry_convert_from="/mnt/pfs/users/linyoutian/worktree/trt_pbr/outputs/stage_III_dmtet_pbr/rgb/006-sd_01_05_train_studio_light_tv10_r100_m100_clip10_gui20_guiii20_r02_2000/A_bulldog_wearing_a_black_pirate_hat_stageIII/ckpts/last.ckpt"
