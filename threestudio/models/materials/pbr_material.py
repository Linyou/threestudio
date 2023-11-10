import random
from dataclasses import dataclass, field

import os
import glob
import envlight
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.utils.ops import get_activation
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.typing import *

import imageio
imageio.plugins.freeimage.download()

EPS = torch.finfo(torch.float32).eps

def linear_to_srgb(
    linear: torch.Tensor,
    eps: Optional[float] = None
) -> torch.Tensor:
  """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
  if eps is None:
    eps = torch.tensor([EPS, ], device=linear.device)
  srgb0 = 323 / 25 * linear
  srgb1 = (211 * torch.maximum(linear, eps)**(5 / 12) - 11) / 200
  return torch.where(linear <= 0.0031308, srgb0, srgb1)

#----------------------------------------------------------------------------
# Vector operations
#----------------------------------------------------------------------------

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

class AutoGamma(torch.nn.Module):
    def __init__(self, gamma=1.0, trainable=True):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.tensor(gamma, requires_grad=trainable))
        
    def forward(self, x):
        return x.pow(1.0 / (F.relu(self.gamma)+1e-10))

@threestudio.register("pbr-material")
class PBRMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        material_activation: str = "sigmoid"
        environment_texture: str = "load/lights/mud_road_puresky_1k.hdr"
        environment_scale: float = 2.0
        min_metallic: float = 0.0
        max_metallic: float = 0.9
        min_roughness: float = 0.08
        max_roughness: float = 0.9
        use_bump: bool = True
        use_preload_environment: bool = False
        environment_train: bool = False
        albedo_train: bool = True
        roughness_train: bool = True
        metallic_train: bool = True
        use_preset_env_color: bool = False
        color_preset: float = 0.5
        brdf_autoencoder: bool = False
        switch_lights: bool = False
        switch_lights_interval: int = 1000
        specular_model: bool = False
        nerual_bsdf: bool = False
        light_rotation: bool = False
        use_visiblity: bool = False
        use_auto_gamma: bool = False
        fix_gamma: bool = False
        test_environment_scale: float = 1.0
        light_rotation_rand: float = 10
        single_channel_light: bool = False
        albedo_gamma: bool = False
        
        reset_light_path: str = "load/lights/mud_road_puresky_1k.hdr"
        
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 6,
                "n_hidden_layers": 1,
            }
        )

    cfg: Config

    def configure(self) -> None:
        self.requires_normal = True
        self.requires_tangent = self.cfg.use_bump

        self.light = envlight.EnvLight(
            path=self.cfg.environment_texture if self.cfg.use_preload_environment else None, 
            color_preset=self.cfg.color_preset if self.cfg.use_preset_env_color else None,
            scale=self.cfg.environment_scale, 
            trainable=self.cfg.environment_train,
        )
        if self.cfg.use_auto_gamma:
            if self.cfg.fix_gamma:
                self.auto_gamma = AutoGamma(2.2, False)
            else:
                self.auto_gamma = AutoGamma(trainable=True)
        
        if self.cfg.brdf_autoencoder:
            self.brdf_encoder = get_mlp(
                n_input_dims=9,
                n_output_dims=3,
                config=self.cfg.mlp_network_config,
            )
            self.brdf_decoder = get_mlp(
                n_input_dims=3,
                n_output_dims=9,
                config=self.cfg.mlp_network_config,
            )
        else:
            self.brdf_encoder = None
            self.brdf_decoder = None
            
        if self.cfg.switch_lights:
            self.lights = []
            env_root = "/mnt/pfs/users/linyoutian/env_map"
            env_paths = glob.glob(os.path.join(env_root, '*.hdr'))
            for env_path in env_paths:
                self.lights.append(
                    envlight.EnvLight(
                        env_path, scale=self.cfg.environment_scale, 
                        # trainable=True,
                    )
                )
            self.lights_step_count = 0

        FG_LUT = torch.from_numpy(
            np.fromfile("load/lights/bsdf_256_256.bin", dtype=np.float32).reshape(
                1, 256, 256, 2
            )
        )
        self.register_buffer("FG_LUT", FG_LUT)
        
        if self.cfg.specular_model:
            self.albedo_dim = 3
            self.visibility_dim = 1
            self.specular_metallic_dim = 3
            self.roughness_dim = 1
            self.perturb_normal_dim = 3
        else:
            self.albedo_dim = 3
            self.visibility_dim = 1
            self.specular_metallic_dim = 1
            self.roughness_dim = 1
            self.perturb_normal_dim = 3
            
        self.test_switch_lights = False
        self.test_switch_lights_delta = 0

    def forward(
        self,
        features: Float[Tensor, "*B Nf"],
        viewdirs: Float[Tensor, "*B 3"],
        shading_normal: Float[Tensor, "B ... 3"],
        tangent: Optional[Float[Tensor, "B ... 3"]] = None,
        azimuth: Float[Tensor, "*B"] = None,
        detach_albedo: bool = False,
        render_visiblity: bool = False,
        **kwargs,
    ) -> Float[Tensor, "*B 3"]:
        prefix_shape = features.shape[:-1]
        
        if "dir_features" in kwargs:
            dir_features = kwargs["dir_features"]
            neural_resiual = get_activation(self.cfg.material_activation)(dir_features)
            (
                # neural_diffuse,
                # neural_specular,
                neural_diffuse_light,
                neural_specular_light,
            ) = torch.split(
                neural_resiual,
                [3, 3], dim=-1
            )
        
        if self.cfg.brdf_autoencoder:
            material: Float[Tensor, "*B Nf"] = self.brdf_encoder(features)
            material = self.brdf_decoder(material)
            
        material = get_activation(self.cfg.material_activation)(features)
        # else:
        #     material: Float[Tensor, "*B Nf"] = [get_activation(self.cfg.material_activation)(
        #         f
        #     ) for f in features]
        
        (
            original_albedo, 
            # neural_diffuse,
            # neural_specular,
            visibility,
            specular_metallic, 
            roughness, 
            perturb_normal
        ) = torch.split(
            material, 
            [
                self.albedo_dim, 
                # 3, 3,
                self.visibility_dim,
                self.specular_metallic_dim,
                self.roughness_dim,
                self.perturb_normal_dim,
            ], 
            dim=-1
        )
    
        if detach_albedo:
            albedo = original_albedo.detach()
        else:
            albedo = original_albedo
            
        if not self.cfg.albedo_train:
            albedo = torch.full_like(albedo, 0.5)
        if not self.cfg.metallic_train:
            specular_metallic = torch.full_like(specular_metallic, 0.0)
        if not self.cfg.roughness_train:
            roughness = torch.full_like(roughness, 0.5)
            
            
        if not self.cfg.specular_model:
            specular_metallic = (
                specular_metallic * (self.cfg.max_metallic - self.cfg.min_metallic)
                + self.cfg.min_metallic
            )
            roughness = (
                roughness * (self.cfg.max_roughness - self.cfg.min_roughness)
                + self.cfg.min_roughness
            )

        if self.cfg.use_bump:
            assert tangent is not None
            # perturb_normal is a delta to the initialization [0, 0, 1]
            perturb_normal = (perturb_normal * 2 - 1) + torch.tensor(
                [0, 0, 1], dtype=perturb_normal.dtype, device=perturb_normal.device
            )
            perturb_normal = F.normalize(perturb_normal.clamp(-1, 1), dim=-1)

            # apply normal perturbation in tangent space
            bitangent = F.normalize(torch.cross(tangent, shading_normal), dim=-1)
            shading_normal = (
                tangent * perturb_normal[..., 0:1]
                - bitangent * perturb_normal[..., 1:2]
                + shading_normal * perturb_normal[..., 2:3]
            )
            shading_normal = F.normalize(shading_normal, dim=-1)

        v = -viewdirs
        n_dot_v = (shading_normal * v).sum(-1, keepdim=True)
        reflective = n_dot_v * shading_normal * 2 - v
        reflective = safe_normalize(reflective)

        if not self.cfg.specular_model:
            diffuse_albedo = (1 - specular_metallic) * albedo
            F0 = (1 - specular_metallic) * 0.04 + specular_metallic * albedo
        else:
            diffuse_albedo = albedo
            F0 = specular_metallic

        if not self.cfg.nerual_bsdf:
            fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1)
            fg = dr.texture(
                self.FG_LUT,
                fg_uv.reshape(1, -1, 1, 2).contiguous(),
                filter_mode="linear",
                boundary_mode="clamp",
            ).reshape(*prefix_shape, 2)
                
            specular_albedo = F0 * fg[:, 0:1] + fg[:, 1:2]
        else:
            # use mlp 
            specular_albedo = F0
            
        if self.cfg.light_rotation:
            # self.light.update_light(1)
            # self.light.build_mips()
            roll_pixels = int((azimuth[0].item() / 180.) * 2048 + 800)
            self.light.update_light(roll_pixels, rand=self.cfg.light_rotation_rand if self.training else 0)
            self.light.build_mips()
        
        if self.cfg.switch_lights and self.training:
            if kwargs.get("step", 0) % self.cfg.switch_lights_interval == 0:
                self.lights_step_count += 1

            light_index = self.lights_step_count % len(self.lights)
            
            diffuse_light, specular_light = self.lights[light_index](
                shading_normal[..., [0, 2, 1]], 
                reflective[..., [0, 2, 1]], 
                roughness
            )
        elif self.test_switch_lights:
            delta = self.test_switch_lights_delta
            light_index = self.lights_step_count
            # import pdb; pdb.set_trace()
            self.lights[light_index].update_light(delta)
            self.lights[light_index].build_mips()
            # print(f"running light: {light_index} and delta: {delta}")
            diffuse_light, specular_light = self.lights[light_index](
                shading_normal[..., [0, 2, 1]], 
                reflective[..., [0, 2, 1]], 
                roughness
            )
        else:
            diffuse_light, specular_light = self.light(
                shading_normal[..., [0, 2, 1]], 
                reflective[..., [0, 2, 1]], 
                roughness
            )
        if self.cfg.single_channel_light:
            diffuse_light, specular_light = (
                torch.mean(diffuse_light, dim=-1, keepdim=True), 
                torch.mean(specular_light, dim=-1, keepdim=True)
            )
        diffuse_color = (diffuse_albedo) * (diffuse_light)
        specular_color = (specular_albedo) * (specular_light)
        # specular_color = specular_albedo * specular_light

        color = diffuse_color + specular_color
        color = color.clamp(0.0, 1.0)
        
        if self.cfg.use_auto_gamma:
            color = self.auto_gamma(color)
            if self.cfg.albedo_gamma:
                original_albedo = self.auto_gamma(original_albedo)
        
        # color = color.pow(1.0 / (F.relu(self.gamma)+1e-10))
        # to srgb
        # color = torch.clamp(linear_to_srgb(color), 0.0, 1.0)
        results_dict = {}
        if self.cfg.use_visiblity:
            render = color
            # visibility = visibility * 0.5
            if render_visiblity:
                color = render * (1.0 - visibility)
            results_dict.update({
                'render': render,
                'visiblity': visibility,
            })
        # else:
        #     color = self.auto_gamma(color)
        
        results_dict.update({
            # Modulate by hemisphere visibility
            'color': color,
            'albedo': original_albedo,
            # 'neural_diffuse': neural_diffuse,
            # 'neural_specular': neural_specular,
            # 'neural_diffuse_light': neural_diffuse_light,
            # 'neural_specular_light': neural_specular_light,
            'specular_metallic': specular_metallic,
            'roughness': roughness,
            'diffuse': diffuse_color,
            'specular': specular_color,
            'diffuse_light': diffuse_light,
            'specular_light': specular_light,
            'shading_normal': (shading_normal + 1.0)*0.5,
        })

        return results_dict
    
    def light_regularizer_tv(self):
        term1 = torch.pow(self.light.base[:,1:,:,:]-self.light.base[:,:-1,:,:], 2).sum()
        term2 = torch.pow(self.light.base[:,:,1:,:]-self.light.base[:,:,:-1,:], 2).sum()
        return (term1+term2)/self.light.max_res
    
    def light_regularizer_white(self):
        white = (self.light.base[..., 0:1] + self.light.base[..., 1:2] + self.light.base[..., 2:3]) / 3
        return torch.mean(torch.abs(self.light.base - white))
    
    def reset_light(self):
        self.light = envlight.EnvLight(
            path=self.cfg.reset_light_path, 
            scale=self.cfg.test_environment_scale,
            trainable=True,
        )
        
    def rotate_lightning(self, delta=1):
        self.light.update_light(delta)
        self.light.build_mips()
        
    def rotate_all_lightning(self, delta=1):
        light_index = self.lights_step_count
        self.lights[light_index].update_light(delta)
        self.lights[light_index].build_mips()

    def get_lightning(self):
        if self.cfg.switch_lights:
            return self.lights[self.lights_step_count % len(self.lights)]
        else:
            return self.light

    def read_all_lights(self):
        self.lights = []
        env_root = "/mnt/pfs/users/linyoutian/threestudio/load/lights/hmap"
        env_paths = glob.glob(os.path.join(env_root, '*.hdr'))
        for env_path in env_paths:
            self.lights.append(
                envlight.EnvLight(
                    env_path, 
                    scale=self.cfg.test_environment_scale, 
                    # trainable=True,
                )
            )

    def export(self, features: Float[Tensor, "*N Nf"], **kwargs) -> Dict[str, Any]:
        material: Float[Tensor, "*N Nf"] = get_activation(self.cfg.material_activation)(
            features
        )
        albedo = material[..., :3]
        metallic = (
            material[..., 4] * (self.cfg.max_metallic - self.cfg.min_metallic)
            + self.cfg.min_metallic
        )
        roughness = (
            material[..., 5] * (self.cfg.max_roughness - self.cfg.min_roughness)
            + self.cfg.min_roughness
        )
        # import pdb; pdb.set_trace()

        out = {
            "albedo": albedo,
            "metallic": metallic,
            "roughness": roughness,
        }

        if self.cfg.use_bump:
            perturb_normal = (material[..., 5:8] * 2 - 1) + torch.tensor(
                [0, 0, 1], dtype=material.dtype, device=material.device
            )
            perturb_normal = F.normalize(perturb_normal.clamp(-1, 1), dim=-1)
            perturb_normal = (perturb_normal + 1) / 2
            out.update(
                {
                    "bump": perturb_normal,
                }
            )

        return out


