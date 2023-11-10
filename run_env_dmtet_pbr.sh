export http_proxy=http://192.168.48.17:18000
export https_proxy=http://192.168.48.17:18000
export HF_HOME="/mnt/pfs/share/pretrained_model/.cache/huggingface"
export OMP_NUM_THREADS=2
prompt="A bulldog wearing a black pirate hat"
basecolor_prefix="2d image of "
basecolor_sixfix=", albedo, base color"
rgb_sixfix=" with PBR textures, studio lighting"
python launch.py --config configs/sd_dmtet_pbr.yaml --train --gpu 0 \
system.prompt_processor.prompt="$prompt $rgb_sixfix" \
system.prompt_processor_II.prompt="$basecolor_prefix $prompt $basecolor_sixfix" \
system.geometry_convert_from="/mnt/pfs/users/linyoutian/threestudio/outputs/stage_II_dmtet/rgb_normal/v3/A_bulldog_wearing_a_black_pirate_hat_stageII/ckpts/last.ckpt" \