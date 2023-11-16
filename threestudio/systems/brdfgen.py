from dataclasses import dataclass, field, asdict
from omegaconf import OmegaConf

import imageio

import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from einops import rearrange
import cv2

import os
import random
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch.nn.functional as F

from threestudio.utils.misc import cleanup, get_device, load_module_weights

# TODO: Update this example when TensorBoard is released with
# https://github.com/tensorflow/tensorboard/pull/4585
# which supports fenced codeblocks in Markdown.
import json
def pretty_json(hp):
  json_hp = json.dumps(hp, indent=2)
  return "".join("\t" + line for line in json_hp.splitlines(True))

def gray_to_rgb(gray):
    return gray.expand(-1, 3, -1, -1)

def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    # return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)
    return f

@threestudio.register("brdfgen-system")
class BRDFGen(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        latent_steps: int = 1000
        texture: bool = False
        test_reset_light: bool = False
        rgb_as_latents: bool = False
        use_albedo_brighness_loss: bool = False
        use_guidance_base: bool = False
        guidance_base_type: str = ""
        guidance_base: dict = field(default_factory=dict)
        render_relight: bool = False
        refinement: bool = False
        test_light_rotation: bool = False
        test_light_rotate_half: bool = False
        test_light_rotation_all: bool = False
        use_ks_smooth_loss: bool = False
        use_albedo_bg: bool = False
        
        detach_albedo: bool = False
        
        guidance_type_II: str = ""
        guidance_II: dict = field(default_factory=dict)
        
        prompt_processor_type_II: str = ""
        prompt_processor_II: dict = field(default_factory=dict)
        
        use_azimuth_offset: bool = False
        azimuth_offset: int = 90
        
        render_visiblity: bool = True
        init_rm_with_value: bool = False
        init_r_value: float = 0.6
        init_m_value: float = 0.2
        
        reuse_prev_geometry: bool = False
        
        clip_prop: float = 1
        guidance_alter: bool = False
        
        guidance_eval: bool = True
        
    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        if (
            self.cfg.geometry_convert_from  # from_coarse must be specified
            and not self.cfg.weights  # not initialized from coarse when weights are specified
            and not self.resumed  # not initialized from coarse when resumed from checkpoints
        ):
            threestudio.info("Initializing geometry from a given checkpoint ...")
            from threestudio.utils.config import load_config, parse_structured

            prev_cfg = load_config(
                os.path.join(
                    os.path.dirname(self.cfg.geometry_convert_from),
                    "../configs/parsed.yaml",
                )
            )  # TODO: hard-coded relative path
            prev_system_cfg: BaseLift3DSystem.Config = parse_structured(
                self.Config, prev_cfg.system
            )
            prev_geometry_cfg = prev_system_cfg.geometry
            prev_geometry_cfg.update(self.cfg.geometry_convert_override)
            prev_geometry = threestudio.find(prev_system_cfg.geometry_type)(
                prev_geometry_cfg
            )
            state_dict, epoch, global_step = load_module_weights(
                self.cfg.geometry_convert_from,
                module_name="geometry",
                map_location="cpu",
            )
            prev_geometry.load_state_dict(state_dict, strict=False)
            # restore step-dependent states
            prev_geometry.do_update_step(epoch, global_step, on_load_weights=True)
            # convert from coarse stage geometry
            prev_geometry = prev_geometry.to(get_device())
            self.geometry = threestudio.find(self.cfg.geometry_type).create_from(
                prev_geometry,
                self.cfg.geometry,
                copy_net=self.cfg.geometry_convert_inherit_texture,
            )
            if not self.cfg.reuse_prev_geometry:
                del prev_geometry
            else:
                self.prev_geometry = prev_geometry
                
            cleanup()
        else:
            self.geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)

        self.material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        self.background = threestudio.find(self.cfg.background_type)(
            self.cfg.background
        )
        self.renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )
        
        if self.cfg.reuse_prev_geometry and self.cfg.geometry_convert_from and not self.cfg.weights and not self.resumed:
            self.prev_meterials = threestudio.find(prev_system_cfg.material_type)(prev_system_cfg.material)
                
            self.pre_renderer = threestudio.find(prev_system_cfg.renderer_type)(
                prev_system_cfg.renderer,
                geometry=self.prev_geometry,
                material=self.prev_meterials,
                background=self.background,
            )
        

        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        if self.cfg.use_guidance_base:
            self.guidance_base = threestudio.find(self.cfg.guidance_base_type)(
                self.cfg.guidance_base
            )
            
        self.prompt_utils = self.prompt_processor()
            
        if self.cfg.guidance_type_II != "":
            self.prompt_processor_II = threestudio.find(self.cfg.prompt_processor_type_II)(
                self.cfg.prompt_processor_II
            )
            self.guidance_II = threestudio.find(self.cfg.guidance_type_II)(self.cfg.guidance_II)

            self.prompt_utils_II = self.prompt_processor_II()
        
            self.background_II = threestudio.find(self.cfg.background_type)(
                self.cfg.background
            )
        
        self.relight_batch_ids = [0, 10, 60]
        
        if self.cfg.test_light_rotation_all:
            self.material.read_all_lights()
        if self.cfg.reuse_prev_geometry:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                "lambdalabs/sd-image-variations-diffusers", 
                subfolder="feature_extractor", 
            )
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "lambdalabs/sd-image-variations-diffusers", 
                subfolder="image_encoder",
            )
        
    def clip_image_loss(self, batch, current_rgb):
        
        with torch.no_grad():
            pre_render_out = self.pre_renderer(
                **batch, 
                render_rgb=self.cfg.texture, 
                step=self.true_global_step, 
                render_depth=False,
                positions_jitter=False,
                detach_albedo=False,
            )
            
        pre_rgb = pre_render_out["comp_color"] if "comp_color" in pre_render_out else pre_render_out["comp_rgb"]
        
        batch_images = torch.cat([pre_rgb, current_rgb])
        
        clip_image_mean = torch.as_tensor(self.feature_extractor.image_mean)[:,None,None].to(pre_rgb.device, dtype=torch.float32)
        clip_image_std = torch.as_tensor(self.feature_extractor.image_std)[:,None,None].to(pre_rgb.device, dtype=torch.float32)
        
        
        batch_images = rearrange(batch_images, 'b h w c -> b c h w')
        batch_images_in_proc = TF.resize(batch_images, (self.feature_extractor.crop_size['height'], self.feature_extractor.crop_size['width']), interpolation=InterpolationMode.BICUBIC)
        # do the normalization in float32 to preserve precision
        imgs_in_proc = ((batch_images_in_proc.float() - clip_image_mean) / clip_image_std)
        image_embeddings = self.image_encoder(imgs_in_proc).image_embeds.unsqueeze(1)
        
        pre_embedding, current_embedding = image_embeddings.chunk(2)
        
        return F.mse_loss(pre_embedding, current_embedding)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_kwargs = {}
        if self.cfg.use_albedo_bg:
            render_kwargs['background_II'] = self.background_II
            
            
        render_kwargs["render_visiblity"] = self.cfg.render_visiblity
        render_out = self.renderer(
            **batch, 
            render_rgb=self.cfg.texture, 
            step=self.true_global_step, 
            render_depth=False,
            positions_jitter=self.cfg.use_ks_smooth_loss,
            detach_albedo=self.cfg.detach_albedo,
            **render_kwargs,
        )
        
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if not self.cfg.texture:
            # initialize SDF
            # FIXME: what if using other geometry types?
            self.geometry.initialize_shape()

    def training_step(self, batch, batch_idx):
        loss = 0.0

        out = self(batch)
        prompt_utils = self.prompt_utils
        
        guidance_eval = self.true_global_step == 0 if not self.cfg.init_rm_with_value else self.true_global_step == 100

        guidance_inp = out["comp_color"]
        cond_inp = out["comp_normal"]
        
        if self.true_global_step < 100 and self.cfg.init_rm_with_value:
            
            loss = F.mse_loss(out["raw_roughness"], torch.zeros_like(out["raw_roughness"]) + self.cfg.init_r_value)
            loss += F.mse_loss(out["raw_metallic"], torch.zeros_like(out["raw_metallic"]) + self.cfg.init_m_value)
            
            return {"loss": loss}
        
        if "comp_normal" in out:
            if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                loss_normal_consistency = out["mesh"].normal_consistency()
                self.log("train/loss_normal_consistency", loss_normal_consistency)
                loss += loss_normal_consistency * self.C(
                    self.cfg.loss.lambda_normal_consistency
                )
            
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )

        if self.cfg.guidance_alter:
            # run_main_guidance = self.true_global_step % 2 == 0
            # run_dual_guidance = self.true_global_step % 2 == 1
            if random.random()<0.5:
                run_main_guidance = False
                run_dual_guidance = True
            else:
                run_main_guidance = True
                run_dual_guidance = False
        else:
            run_main_guidance = True
            run_dual_guidance = True

        # main guidance
        if run_main_guidance:
            if isinstance(
                self.guidance,
                threestudio.models.guidance.controlnet_guidance.ControlNetGuidance,
            ):
                guidance_out = self.guidance(
                    guidance_inp, cond_inp, prompt_utils, **batch, rgb_as_latents=False
                )
            elif isinstance(
                self.guidance,
                threestudio.models.guidance.stable_diffusion_guidance.StableDiffusionGuidance,
            ):
                guidance_out = self.guidance(
                    guidance_inp, 
                    prompt_utils,  
                    **batch, 
                    rgb_as_latents=self.cfg.rgb_as_latents,
                    guidance_eval=guidance_eval,
                )
                if guidance_eval:
                    self.guidance_evaluation_save(
                        out["comp_color"].detach()[: guidance_out["eval"]["bs"]],
                        guidance_out["eval"]
                    )
                    guidance_out.pop("eval")
            elif isinstance(
                self.guidance,
                threestudio.models.guidance.stable_diffusion_guidance.StableDiffusionGuidance,
            ) or isinstance(
                self.guidance,
                threestudio.models.guidance.stable_diffusion_guidance_trt.StableDiffusionGuidanceTRT,
            ):
                guidance_out = self.guidance(
                    guidance_inp, 
                    prompt_utils,  
                    **batch, 
                    rgb_as_latents=self.cfg.rgb_as_latents,
                    guidance_eval=guidance_eval,
                )
                if guidance_eval:
                    self.guidance_evaluation_save(
                        out["comp_color"].detach()[: guidance_out["eval"]["bs"]],
                        guidance_out["eval"]
                    )
                    guidance_out.pop("eval")
            elif isinstance(
                self.guidance,
                threestudio.models.guidance.stable_diffusion_unified_guidance_trt.StableDiffusionUnifiedGuidanceTRT,
            ) or isinstance(
                self.guidance,
                threestudio.models.guidance.stable_diffusion_unified_guidance.StableDiffusionUnifiedGuidance
            ):
                controlnet_conditions = {
                    "rgb": out['comp_color']
                }
                if "comp_normal" in out:
                    controlnet_conditions["normal"] = out["comp_normal"]
                guidance_out = self.guidance(
                    out["comp_color"], prompt_utils, **batch, rgb_as_latents=False,
                    controlnet_conditions=controlnet_conditions
                )
            elif isinstance(
                self.guidance,
                threestudio.models.guidance.brdf_guidance.BRDFGenGuidance
            ):
                guidance_out = self.guidance(
                    out["comp_color"], 
                    out["comp_albedo"],
                    out["comp_roughness"],
                    out["comp_specular_metallic"],
                    out["comp_normal"],
                    prompt_utils,  
                    **batch, 
                )
                
            for name, value in guidance_out.items():
                # self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
            

        # dual guidance
        if self.cfg.guidance_type_II != "" and run_dual_guidance:
            prompt_utils_II = self.prompt_utils_II
            # prompt_utils_II = self.prompt_utils
            guidance_out_II = self.guidance_II(
                out["comp_albedo"], 
                prompt_utils_II,  
                **batch, 
                rgb_as_latents=self.cfg.rgb_as_latents,
                guidance_eval=guidance_eval,
            )
            if guidance_eval:
                self.guidance_evaluation_save(
                    out["comp_albedo"].detach()[: guidance_out_II["eval"]["bs"]],
                    guidance_out_II["eval"],
                    name='train_II'
                )
                guidance_out_II.pop("eval")
                
            for name, value in guidance_out_II.items():
                self.log(f"train/{name}_II", value)
                if name.startswith("loss_"):
                    loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")+"_II"])
            
                
        if self.cfg.use_albedo_brighness_loss:
            loss += self.C(self.cfg.loss['lambda_brighness']) * out["raw_albedo"].mean()
            
        if self.cfg.use_ks_smooth_loss:
            kd_ks_smooth_loss = 0
            if "kd_grad" in out:
                kd_ks_smooth_loss += self.C(self.cfg.loss['lambda_kd_loss'])*(out['kd_grad']**2).mean()
                
            if "roughness_grad" in out:
                kd_ks_smooth_loss += self.C(self.cfg.loss['lambda_kr_loss'])*(out['roughness_grad']**2).mean()
                
            if "metallic_grad" in out:
                kd_ks_smooth_loss += self.C(self.cfg.loss['lambda_km_loss'])*(out['metallic_grad']**2).mean()
                
            loss += kd_ks_smooth_loss
            
        
            
        if self.material.cfg.environment_train and 'lambda_light_regularizer_tv' in self.cfg.loss:
            if self.C(self.cfg.loss['lambda_light_regularizer_tv']) > 0 and random.random()<0.5:
                loss += self.C(self.cfg.loss['lambda_light_regularizer_tv']) * self.material.light_regularizer_tv()
            
        if self.material.cfg.environment_train and 'lambda_light_regularizer_white' in self.cfg.loss:
            if self.C(self.cfg.loss['lambda_light_regularizer_white']) > 0:
                loss += self.C(self.cfg.loss['lambda_light_regularizer_white']) * self.material.light_regularizer_white()

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        
        if self.material.cfg.use_auto_gamma:
            self.log("train_params/gamma", self.material.auto_gamma.gamma.data)
        
        if 'lambda_metallic_loss' in self.cfg.loss:
            if self.C(self.cfg.loss['lambda_metallic_loss']) > 0:
                o = out['raw_metallic']
                loss += self.C(self.cfg.loss['lambda_metallic_loss']) * (-o*torch.log(o)).mean()
            
        if 'lambda_clipimage_loss' in self.cfg.loss:
            if self.C(self.cfg.loss['lambda_clipimage_loss']) > 0:
                if random.random()<self.cfg.clip_prop:
                    loss += self.clip_image_loss(batch, out["comp_color"])

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        self.cfg.render_visiblity = False
        out = self(batch)
        self.cfg.render_visiblity = True
        
        if self.cfg.texture:
            if self.true_global_step % 500 == 0:
                probe = self.renderer.material.get_lightning().get_env_map()
                probe_save_path = self.get_save_path(f"probe/it{self.true_global_step}.hdr")
                imageio.imwrite(probe_save_path, probe.detach().cpu().numpy())
            
            
        # tonemap = cv2.createTonemapDrago(2.2)
        # scale = 5
        # ldr = scale * tonemap.process(out["comp_color"][0].detach().cpu().numpy())
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": _rgb_to_srgb(out["comp_color"][0]),
                        # "img": ldr,
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if self.cfg.texture
                else []
            ) + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_render"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ] if "comp_render" in out else []
            )+ (
                [
                    {
                        "type": "rgb",
                        "img": (out["comp_shading_normal"][0]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
            ) + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_albedo"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_roughness"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_specular_metallic"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
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
            log_to_logger=True,
            align=1024,
        )

    def on_validation_epoch_end(self):
        pass
    
    def loop_all_lights_noraml(self, batch, batch_idx):
        # self.renderer.material.read_all_lights()
        self.material.test_switch_lights = True
        lightning_width = self.material.light.base_image.shape[1]
        delta = lightning_width/self.dataset.n_views
        self.material.test_switch_lights_delta = int(batch_idx*delta)
        for i in range(len(self.material.lights)):
            self.material.lights_step_count = i
            self.material.rotate_all_lightning(int(lightning_width/2))
            out = self(batch)
            self.save_image_grid(
                f"it{self.true_global_step}-test-lights-rotate-{i}/batch_{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": _rgb_to_srgb(out["comp_color"][0]),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                )
                + [
                    {
                        "type": "rgb",
                        "img": _rgb_to_srgb(out["comp_diffuse"][0]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": _rgb_to_srgb(out["comp_specular"][0]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": _rgb_to_srgb(out["comp_diffuse_light"][0]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": _rgb_to_srgb(out["comp_specular_light"][0]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },

                ],
                name="test_step_relight",
                step=self.true_global_step,
                log_to_logger=False,
                align=1024,
            )
        self.renderer.material.test_switch_lights = False

    def test_step(self, batch, batch_idx):
        self.cfg.render_visiblity = False
        if self.cfg.test_reset_light:
            self.renderer.material.reset_light()
            
        if self.cfg.test_light_rotate_half:
            light_upd_delta = int(self.material.light.base_image.shape[1]/2)
            self.material.rotate_lightning(light_upd_delta)
            
        self.renderer.material.cfg.light_rotation = False
        
        # if self.cfg.render_relight:
        #     if batch_idx in self.relight_batch_ids:
        #         self.loop_all_lights(batch, batch_idx)
                
        if self.cfg.test_light_rotation_all:
            self.loop_all_lights_noraml(batch, batch_idx)
                
        if self.cfg.test_light_rotation:
            light_upd_delta = int(self.material.light.base_image.shape[1]*batch_idx/self.dataset.n_views)
            self.material.rotate_lightning(light_upd_delta)
                
        out = self(batch)
    
        
        if self.cfg.texture and batch_idx == 0:
            probe = self.renderer.material.light.get_env_map()
            probe_save_path = self.get_save_path(f"probe/probe.hdr")
            imageio.imwrite(probe_save_path, probe.detach().cpu().numpy())
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": _rgb_to_srgb(out["comp_color"][0]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if self.cfg.texture
                else []
            ) + (
                [
                    {
                        "type": "rgb",
                        "img": (out["comp_shading_normal"][0]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ] 
            ) + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_albedo"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_roughness"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_specular_metallic"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
            ),
            name="test_step",
            step=self.true_global_step,
            log_to_logger=True,
            align=1024,
        )

    def on_test_epoch_end(self):
        if self.cfg.test_light_rotation_all:
            for i in range(len(self.renderer.material.lights)):
                self.save_img_sequence(
                    f"it{self.true_global_step}-test-lights-rotate-{i}",
                    f"it{self.true_global_step}-test-lights-rotate-{i}",
                    "(\d+)\.png",
                    save_format="mp4",
                    fps=30,
                    name=f"test-lights-rotate-{i}",
                    step=self.true_global_step,
                    log_to_logger=True,
                )
        
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
            log_to_logger=True,
        )
        
    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out, name='train'):
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-{name}.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.true_global_step,
            texts=guidance_eval_out["texts"],
        )
