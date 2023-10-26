from dataclasses import dataclass, field

import torch
import random
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("unidream-system")
class UniDream(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'only-geometry', 'geometry', 'uni']
        stage: str = "coarse"
        latent_steps: int = 1000
        pass

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance_normal = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance_normal)


    def forward(self, batch: Dict[str, Any], uni_choice) -> Dict[str, Any]:
        if "geometry" in self.cfg.stage or uni_choice == 'normal': 
            render_out = self.renderer(**batch, render_rgb=False)
        else:
            render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        
        if self.cfg.stage == 'only-geometry':
            self.geometry.initialize_shape()

    def training_step(self, batch, batch_idx):
        # import pdb
        # pdb.set_trace()
        if self.cfg.stage == "uni":
            if random.random() < 0.8:
                uni_choice = 'rgb'
            else:
                uni_choice = 'normal'
            out = self(batch, uni_choice)
        else:
            out = self(batch)
        prompt_utils = self.prompt_processor()
        if self.cfg.stage == "coarse":
            guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(
                guidance_inp, prompt_utils, **batch, rgb_as_latents=False
            )
        elif self.cfg.stage == "geometry":
            guidance_inp = out["comp_normal"]
            guidance_out = self.guidance_normal(
                guidance_inp, prompt_utils, **batch, rgb_as_latents=False
            )
        elif self.cfg.stage == "only-geometry":
            if self.true_global_step < self.cfg.latent_steps:
                guidance_inp = torch.cat(
                    [out["comp_normal"] * 2.0 - 1.0, out["opacity"]], dim=-1
                )
                guidance_out = self.guidance_normal(
                    guidance_inp, prompt_utils, **batch, rgb_as_latents=True
                )
            else:
                guidance_inp = out["comp_normal"]
                guidance_out = self.guidance_normal(
                    guidance_inp, prompt_utils, **batch, rgb_as_latents=False
                )
        elif self.cfg.stage == "uni":
            if uni_choice == 'rgb':
                guidance_inp = out["comp_rgb"]
                guidance_out = self.guidance(
                    guidance_inp, prompt_utils, **batch, rgb_as_latents=False
                )
            elif uni_choice == 'normal':
                guidance_inp = out["comp_normal"]
                guidance_out = self.guidance_normal(
                    guidance_inp, prompt_utils, **batch, rgb_as_latents=False
                )
        loss = 0.0

        for name, value in guidance_out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        
        if self.cfg.stage == "coarse":
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        elif "geometry" in self.cfg.stage or self.cfg.stage == "uni":
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )
            # if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
            #     loss_laplacian_smoothness = out["mesh"].laplacian()
            #     self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
            #     loss += loss_laplacian_smoothness * self.C(
            #         self.cfg.loss.lambda_laplacian_smoothness
            #     )
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch, 'rgb')
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + 
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch, 'rgb')
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ],
            + 
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
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
