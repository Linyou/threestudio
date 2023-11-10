# 加速版Unidream

镜像使用： `ccr-23gxup9u-vpc.cnc.bj.baidubce.com/model/mvdream_speedup:v1`

分两阶段，第一阶段nerf，第二阶段dmtet

加入第三阶段PBR，使用了两个SD，所以要输入第二个prompt `system.prompt_processor_II.prompt`
`export.sh` 也根据PBR的需求做了修改

1. 第一阶段：`sh run_env.sh`
2. 第二阶段：`sh run_env_dmtet.sh`
3. 第三阶段：`sh run_env_dmtet_pbr.sh`
4. 导出模型：`sh export