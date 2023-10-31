from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

import time
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start_time = time.time()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"Function {func.__name__} took {elapsed_time:.4f} microsecond to execute.")
        return result
    return wrapper

@threestudio.register("mvdream-system-vast")
class MVDream(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        refinement: bool = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    @timing_decorator
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch, step=self.true_global_step)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
       
    def training_step(self, batch, batch_idx):
        out = self(batch)
        prompt_utils = self.prompt_processor()
        if "comp_rgb" in out:
            rgb = out["comp_rgb"]
        else:
            rgb = None
        if "comp_normal" in out:
            normal = out["comp_normal"]
        else:
            normal = None
        guidance_out = self.guidance(
            rgb, normal, prompt_utils, **batch, rgb_as_latents=False, step=self.true_global_step
        )

        loss = 0.0

        for name, value in guidance_out.items():
            # self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if not self.cfg.refinement:
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    pass
                    # raise ValueError(
                    #     "Normal is required for orientation loss, no normal is found in the output."
                    # )
                else:
                    loss_orient = (
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum() / (out["opacity"] > 0).sum()
                    self.log("train/loss_orient", loss_orient)
                    loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            if self.C(self.cfg.loss.lambda_sparsity) > 0:
                loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
                self.log("train/loss_sparsity", loss_sparsity)
                loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
                self.log("train/loss_opaque", loss_opaque)
                loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if self.C(self.cfg.loss.lambda_z_variance) > 0:
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

            if hasattr(self.cfg.loss, "lambda_eikonal") and self.C(self.cfg.loss.lambda_eikonal) > 0:
                loss_eikonal = (
                    (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                ).mean()
                self.log("train/loss_eikonal", loss_eikonal)
                loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)
        else:
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                # import pdb;pdb.set_trace()
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else [
                    {
                        "type": "rgb",
                        "img": out["comp_color"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ) + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_albedo"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ] if "comp_albedo" in out else []
            ) + (
                [
                    {
                        "type": "grayscale",
                        "img": out["comp_roughness"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out["comp_specular_metallic"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ] if "comp_roughness" in out else [] 
            ) + (
                [
                    {
                        "type": "grayscale",
                        "img": out["comp_visiblity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ] if "comp_visiblity" in out else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else [
                    {
                        "type": "rgb",
                        "img": out["comp_color"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ) + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_albedo"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ] if "comp_albedo" in out else []
            ) + (
                [
                    {
                        "type": "grayscale",
                        "img": out["comp_roughness"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out["comp_specular_metallic"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ] if "comp_roughness" in out else [] 
            ) + (
                [
                    {
                        "type": "grayscale",
                        "img": out["comp_visiblity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ] if "comp_visiblity" in out else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
