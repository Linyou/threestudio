import random
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DPMSolverSinglestepScheduler,
    StableDiffusionPipeline,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.networks import ToDTypeWrapper
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, enable_gradient, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *

from extern.mvdream_rn.models.unet import UNet3DConditionModel
from extern.mvdream_rn.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from einops import rearrange
from diffusers import AutoencoderTiny
from threestudio.models.trt_model import TensorRTModel

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

def normalize_camera(camera_matrix):
    ''' normalize the camera location onto a unit-sphere'''
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = camera_matrix.reshape(-1,4,4)
        translation = camera_matrix[:,:3,3]
        translation = translation / (np.linalg.norm(translation, axis=1, keepdims=True) + 1e-8)
        camera_matrix[:,:3,3] = translation
    else:
        camera_matrix = camera_matrix.reshape(-1,4,4)
        translation = camera_matrix[:,:3,3]
        translation = translation / (torch.norm(translation, dim=1, keepdim=True) + 1e-8)
        camera_matrix[:,:3,3] = translation
    return camera_matrix

@threestudio.register("multiview-unified-guidancev2")
class MultiViewUnifiedGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        # guidance type, in ["sds", "vsd"]
        guidance_type: str = "sds"
        normal_loss: bool = False
        rgb_loss: bool = False
        camera_condition_type: str = "rotation"
        prob: float = 1.0
        pretrained_model_name_or_path: str = "/mnt/pfs/users/liuzexiang/Code/tune-a-video/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/"
        unet_path: str = "/mnt/pfs/users/liyangguang/vastcode/tune-a-video/outputs/render-group-1-2-laion-ga-3dword-gray-laion20M/"
        guidance_scale: float = 100.0
        weighting_strategy: str = "dreamfusion"
        view_dependent_prompting: bool = True
        n_multi_views: int = 4

        min_step_percent: Any = 0.02
        max_step_percent: Any = 0.98
        grad_clip: Optional[Any] = None

        return_rgb_1step_orig: bool = False
        return_rgb_multistep_orig: bool = False
        n_rgb_multistep_orig_steps: int = 4

        # # TODO
        # # controlnet
        # controlnet_model_name_or_path: Optional[str] = None
        # preprocessor: Optional[str] = None
        # control_scale: float = 1.0

        # # TODO
        # # lora
        # lora_model_name_or_path: Optional[str] = None

        # efficiency-related configurations
        half_precision_weights: bool = True
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        # # VSD configurations, only used when guidance_type is "vsd"
        # vsd_phi_model_name_or_path: Optional[str] = None
        # vsd_guidance_scale_phi: float = 1.0
        # vsd_use_lora: bool = True
        # vsd_lora_cfg_training: bool = False
        # vsd_lora_n_timestamp_samples: int = 1
        # vsd_use_camera_condition: bool = True
        # # camera condition type, in ["extrinsics", "mvp", "spherical"]
        # vsd_camera_condition_type: Optional[str] = "extrinsics"
        rgb_loss_scale: float = 1.0
        normal_loss_scale: float = 1.0
        rgb_pipeline_start_step: int = 10000

    cfg: Config

    def configure(self) -> None:
        self.min_step: Optional[int] = None
        self.max_step: Optional[int] = None
        self.grad_clip_val: Optional[float] = None

        @dataclass
        class NonTrainableModules:
            pipe: TuneAVideoPipeline
            pipe_phi: Optional[TuneAVideoPipeline] = None
            controlnet: Optional[ControlNetModel] = None

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        threestudio.info(f"Loading MVDream...")

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        # unet = UNet3DConditionModel.from_pretrained(
        #     self.cfg.unet_path, subfolder="unet", torch_dtype=self.weights_dtype
        # ).to(self.device)
        engine_path = "/mnt/pfs/users/liuxuebo/project/threestudio_unidream2/trt/rgb_normal_fp16_1031/unet-4view.plan"
        batch = 4
        unet = TensorRTModel(trt_engine_path=engine_path, shape_list=[(batch, 4, 8, 32, 32), (batch,), (batch, 77, 1024), (batch, 4, 12), (2,), (batch, 4, 8, 32, 32)])
        engine_path = "/mnt/pfs/users/liuxuebo/project/threestudio_unidream2/trt/rgb_fp16_1031/unet-4view.plan"
        batch = 2
        self.unet_256 = TensorRTModel(trt_engine_path=engine_path, shape_list=[(batch, 4, 4, 32, 32), (batch,), (batch, 77, 1024), (batch, 4, 12), (batch, 4, 4, 32, 32)])
        pipe = TuneAVideoPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, **pipe_kwargs
        ).to(self.device)
        pipe.unet = unet
        self.prepare_pipe(pipe)
        self.configure_pipe_token_merging(pipe)

        # phi network for VSD
        # introduce two trainable modules:
        # - self.camera_embedding
        # - self.lora_layers
        pipe_phi = None
        threestudio.info(f"Loaded MVDream!")

        # controlnet
        controlnet = None

        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        # q(z_t|x) = N(alpha_t x, sigma_t^2 I)
        # in DDPM, alpha_t = sqrt(alphas_cumprod_t), sigma_t^2 = 1 - alphas_cumprod_t
        self.alphas_cumprod: Float[Tensor, "T"] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        self.alphas: Float[Tensor, "T"] = self.alphas_cumprod**0.5
        self.sigmas: Float[Tensor, "T"] = (1 - self.alphas_cumprod) ** 0.5
        # log SNR
        self.lambdas: Float[Tensor, "T"] = self.sigmas / self.alphas

        self._non_trainable_modules = NonTrainableModules(
            pipe=pipe,
            pipe_phi=pipe_phi,
            controlnet=controlnet,
        )

    @property
    def pipe(self) -> TuneAVideoPipeline:
        return self._non_trainable_modules.pipe

    @property
    def pipe_phi(self) -> TuneAVideoPipeline:
        if self._non_trainable_modules.pipe_phi is None:
            raise RuntimeError("phi model is not available.")
        return self._non_trainable_modules.pipe_phi

    @property
    def controlnet(self) -> ControlNetModel:
        if self._non_trainable_modules.controlnet is None:
            raise RuntimeError("ControlNet model is not available.")
        return self._non_trainable_modules.controlnet

    def prepare_pipe(self, pipe: TuneAVideoPipeline):
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=self.weights_dtype).to(self.device)
        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            pipe.unet.to(memory_format=torch.channels_last)

        # FIXME: pipe.__call__ requires text_encoder.dtype
        # pipe.text_encoder.to("meta")
        cleanup()

        pipe.vae.eval()
        # pipe.unet.eval()

        enable_gradient(pipe.vae, enabled=False)
        # enable_gradient(pipe.unet, enabled=False)

        # disable progress bar
        pipe.set_progress_bar_config(disable=True)

    def configure_pipe_token_merging(self, pipe: TuneAVideoPipeline):
        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(pipe.unet, **self.cfg.token_merging_params)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Int[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "..."]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Float[Tensor, "..."]] = None,
        mid_block_additional_residual: Optional[Float[Tensor, "..."]] = None,
        velocity_to_epsilon: bool = False,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        pred = unet(
            latents.to(unet.dtype),
            t.to(unet.dtype),
            encoder_hidden_states=encoder_hidden_states.to(unet.dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample
        if velocity_to_epsilon:
            pred = latents * self.sigmas[t].view(-1, 1, 1, 1) + pred * self.alphas[
                t
            ].view(-1, 1, 1, 1)
        return pred.to(input_dtype)

    # @torch.cuda.amp.autocast(enabled=False)
    # def vae_encode(
    #     self, vae: AutoencoderKL, imgs: Float[Tensor, "B 3 H W"], mode=False
    # ) -> Float[Tensor, "B 4 Hl Wl"]:
    #     # expect input in [-1, 1]
    #     input_dtype = imgs.dtype
    #     posterior = vae.encode(imgs.to(vae.dtype)).latent_dist
    #     if mode:
    #         latents = posterior.mode()
    #     else:
    #         latents = posterior.sample()
    #     latents = latents * vae.config.scaling_factor
    #     return latents.to(input_dtype)

    # @timing_decorator
    # @torch.cuda.amp.autocast(enabled=False)
    # def vae_encode(
    #     self, vae: AutoencoderTiny, imgs: Float[Tensor, "B 3 256 256"]
    # ) -> Float[Tensor, "B 4 32 32"]:
    #     input_dtype = imgs.dtype
    #     latents = vae.encode(imgs.to(self.weights_dtype)).latents
    #     return latents.to(input_dtype)

    @timing_decorator
    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        input_dtype = imgs.dtype
        latents = self.pipe.vae.encode(imgs.to(self.weights_dtype)).latents
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def vae_decode(
        self, vae: AutoencoderKL, latents: Float[Tensor, "B 4 Hl Wl"]
    ) -> Float[Tensor, "B 3 H W"]:
        # output in [0, 1]
        input_dtype = latents.dtype
        latents = 1 / vae.config.scaling_factor * latents
        image = vae.decode(latents.to(vae.dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    @contextmanager
    def set_scheduler(self, pipe: TuneAVideoPipeline, scheduler_class: Any, **kwargs):
        scheduler_orig = pipe.scheduler
        pipe.scheduler = scheduler_class.from_config(scheduler_orig.config, **kwargs)
        yield pipe
        pipe.scheduler = scheduler_orig

    def get_camera_cond(self, 
            camera: Float[Tensor, "B 4 4"],
            fovy = None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation": # normalized camera
            camera = normalize_camera(camera)
            # camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(f"Unknown camera_condition_type={self.cfg.camera_condition_type}")
        return camera

    def get_eps_pretrain(
        self,
        latents_noisy: Float[Tensor, "B 4 Hl Wl"],
        latents_noisy_normal: Float[Tensor, "B 4 Hl Wl"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        pred_rgb = True,
        pred_normal=True,
        rgb_pipeline=False,
    ) -> Float[Tensor, "B 4 Hl Wl"]:
        # batch_size = latents_noisy.shape[0]
        # batch_unet = batch_size // self.cfg.n_multi_views
        latents_multiview_noisy: Float[Tensor, "Bdivn 4 Nv Hl Wl"] = rearrange(
            latents_noisy, "(b nv) c h w -> b c nv h w", nv=self.cfg.n_multi_views
        )
        latents_multiview_noisy_normal: Float[Tensor, "Bdivn 4 Nv Hl Wl"] = rearrange(
            latents_noisy_normal, "(b nv) c h w -> b c nv h w", nv=self.cfg.n_multi_views
        )

        # camera_matrixs: Float[Tensor, "Bdivn Nv 12"] = rearrange(
        #     c2w[:, :3, :],
        #     "(b nv) three four -> b nv (three four)",
        #     nv=self.cfg.n_multi_views,
        # )
        camera_matrixs = c2w

        assert not prompt_utils.use_perp_neg
        elevation_multiview = rearrange(
            elevation, "(b nv) -> b nv", nv=self.cfg.n_multi_views
        )
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation_multiview[:, 0], None, None, self.cfg.view_dependent_prompting
        )
        if camera_matrixs is not None:
            camera_matrixs = self.get_camera_cond(camera_matrixs)
            camera_matrixs: Float[Tensor, "Bdivn Nv 12"] = rearrange(
                camera_matrixs[:, :3, :],
                "(b nv) three four -> b nv (three four)",
                nv=self.cfg.n_multi_views,
            )
        if rgb_pipeline:
            trt_input_dict = {
                "sample": torch.cat([latents_multiview_noisy] * 2, dim=0),
                "timestep": torch.cat([t] * 2, dim=0),
                "encoder_hidden_states": text_embeddings,
                "camera": torch.cat([camera_matrixs] * 2, dim=0),
            }
            with torch.no_grad():
                noise_pred = self.pipe.unet(**trt_input_dict)['latent']
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
                
            noise_pred = rearrange(noise_pred, "b c nv h w -> (b nv) c h w")

            return noise_pred, 0
        else:
            with torch.no_grad():
                if pred_rgb and pred_normal:
                    class_labels = torch.tensor([0,1]).cuda()
                    latents_multiview_noisy = torch.cat([latents_multiview_noisy, latents_multiview_noisy_normal], dim=2)
                elif pred_normal:
                    class_labels = torch.tensor([1]).cuda()
                    latents_multiview_noisy = latents_multiview_noisy_normal
                else:
                    class_labels = torch.tensor([0]).cuda()
                trt_input_dict = {
                    "sample": torch.cat([latents_multiview_noisy] * 2, dim=0),
                    "timestep": torch.cat([t] * 2, dim=0),
                    "encoder_hidden_states": text_embeddings,
                    "camera": torch.cat([camera_matrixs] * 2, dim=0),
                    "class_labels": class_labels
                }
                print(f"sample: {trt_input_dict['sample'].shape}")
                t1 = time.time()
                noise_pred = self.pipe.unet(**trt_input_dict)['latent']
                torch.cuda.synchronize()
                t2 = time.time()
                print(f"unet time: {t2-t1}")

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            if pred_rgb and pred_normal:
                noise_pred, noise_pred_normal = noise_pred.chunk(2, dim=2)
                noise_pred = rearrange(noise_pred, "b c nv h w -> (b nv) c h w")
                noise_pred_normal = rearrange(noise_pred_normal, "b c nv h w -> (b nv) c h w")
            elif pred_rgb:
                noise_pred = rearrange(noise_pred, "b c nv h w -> (b nv) c h w")
                noise_pred_normal = None
            elif pred_normal:
                noise_pred_normal = rearrange(noise_pred, "b c nv h w -> (b nv) c h w")
                noise_pred = None
            # noise_pred = torch.cat([noise_pred,noise_pred_normal],dim=0)
            return noise_pred, noise_pred_normal

    def get_eps_phi(
        self,
        latents_noisy: Float[Tensor, "B 4 Hl Wl"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera_condition: Float[Tensor, "B ..."],
    ) -> Float[Tensor, "B 4 Hl Wl"]:
        batch_size = latents_noisy.shape[0]

        # not using view-dependent prompting in LoRA
        text_embeddings, _ = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        ).chunk(2)
        with torch.no_grad():
            noise_pred = self.forward_unet(
                self.pipe_phi.unet,
                torch.cat([latents_noisy] * 2, dim=0),
                torch.cat([t] * 2, dim=0),
                encoder_hidden_states=torch.cat([text_embeddings] * 2, dim=0),
                class_labels=torch.cat(
                    [
                        camera_condition.view(batch_size, -1),
                        torch.zeros_like(camera_condition.view(batch_size, -1)),
                    ],
                    dim=0,
                )
                if self.cfg.vsd_use_camera_condition
                else None,
                cross_attention_kwargs={"scale": 1.0},
                velocity_to_epsilon=self.pipe_phi.scheduler.config.prediction_type
                == "v_prediction",
            )

        noise_pred_camera, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.vsd_guidance_scale_phi * (
            noise_pred_camera - noise_pred_uncond
        )

        return noise_pred

    def train_phi(
        self,
        latents: Float[Tensor, "B 4 Hl Wl"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera_condition: Float[Tensor, "B ..."],
    ):
        B = latents.shape[0]
        latents = latents.detach().repeat(
            self.cfg.vsd_lora_n_timestamp_samples, 1, 1, 1
        )

        num_train_timesteps = self.pipe_phi.scheduler.config.num_train_timesteps
        t = torch.randint(
            int(num_train_timesteps * 0.0),
            int(num_train_timesteps * 1.0),
            [B * self.cfg.vsd_lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        latents_noisy = self.pipe_phi.scheduler.add_noise(latents, noise, t)
        if self.pipe_phi.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.pipe_phi.scheduler.prediction_type == "v_prediction":
            target = self.pipe_phi.scheduler.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.pipe_phi.scheduler.prediction_type}"
            )

        # not using view-dependent prompting in LoRA
        text_embeddings, _ = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        ).chunk(2)

        if (
            self.cfg.vsd_use_camera_condition
            and self.cfg.vsd_lora_cfg_training
            and random.random() < 0.1
        ):
            camera_condition = torch.zeros_like(camera_condition)

        noise_pred = self.forward_unet(
            self.pipe_phi.unet,
            latents_noisy,
            t,
            encoder_hidden_states=text_embeddings.repeat(
                self.cfg.vsd_lora_n_timestamp_samples, 1, 1
            ),
            class_labels=camera_condition.view(B, -1).repeat(
                self.cfg.vsd_lora_n_timestamp_samples, 1
            )
            if self.cfg.vsd_use_camera_condition
            else None,
            cross_attention_kwargs={"scale": 1.0},
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    @timing_decorator
    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        normal: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        step = 0,
        **kwargs,
    ):  
        rgb_pipeline = False
        print(f"step: {step}")
        if step>self.cfg.rgb_pipeline_start_step:
            self.pipe.unet = self.unet_256
            rgb_pipeline = True
            self.cfg.normal_loss = 0
            self.cfg.rgb_loss = 1
            self.cfg.normal_loss_scale = 0
            self.cfg.rgb_loss_scale = 1
            
        pred_rgb = True
        pred_normal = True
        if rgb is None:
            pred_rgb = False
            rgb = normal
        if normal is None:
            pred_normal = False
            normal = rgb
        batch_size = rgb.shape[0]
        assert batch_size % self.cfg.n_multi_views == 0
        batch_unet = batch_size // (self.cfg.n_multi_views)

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        normal_BCHW = normal.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 Hl Wl"]
        if rgb_as_latents:
            # treat input rgb as latents
            # input rgb should be in range [-1, 1]
            latents = F.interpolate(
                rgb_BCHW, (32, 32), mode="bilinear", align_corners=False
            )
            latents_normal = F.interpolate(
                normal_BCHW, (32, 32), mode="bilinear", align_corners=False
            )
        else:
            # treat input rgb as rgb
            # input rgb should be in range [0, 1]
            rgb_BCHW = F.interpolate(
                rgb_BCHW, (256, 256), mode="bilinear", align_corners=False
            )
            normal_BCHW = F.interpolate(
                normal_BCHW, (256, 256), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            # latents = self.vae_encode(self.pipe.vae, rgb_BCHW * 2.0 - 1.0)
            # latents_normal = self.vae_encode(self.pipe.vae, normal_BCHW * 2.0 - 1.0)
            latents = self.encode_images(rgb_BCHW)
            if rgb_pipeline:
                latents_normal = latents
            else:
                latents_normal = self.encode_images(normal_BCHW)
            # alpha = 0.5
            # latents_normal = latents_normal * alpha + latents * (1-alpha)
            # latents = torch.cat([latents, latents_normal], dim = 0)
        # sample timestep
        # use the same timestep for each batch
        assert self.min_step is not None and self.max_step is not None

        t_one = torch.randint(
            self.min_step,
            self.max_step + 1,
            [1],
            dtype=torch.long,
            device=self.device,
        )
        t = t_one.repeat(batch_size)
        # sample noise
        noise = torch.randn_like(latents)
        noise_normal = torch.randn_like(latents_normal)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        latents_noisy_normal = self.scheduler.add_noise(latents_normal, noise_normal, t)
        eps_pretrain, eps_pretrain_normal = self.get_eps_pretrain(
            latents_noisy,
            latents_noisy_normal,
            t_one.repeat(batch_unet),
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
            c2w,
            pred_rgb,
            pred_normal,
            rgb_pipeline,
        )
        if pred_rgb:
            latents_1step_orig = (
                1
                / self.alphas[t].view(-1, 1, 1, 1)
                * (latents_noisy - self.sigmas[t].view(-1, 1, 1, 1) * eps_pretrain)
            ).detach()
        else:
            latents_1step_orig = (
                1
                / self.alphas[t].view(-1, 1, 1, 1)
                * (latents_noisy_normal - self.sigmas[t].view(-1, 1, 1, 1) * eps_pretrain_normal)
            ).detach()
        if self.cfg.guidance_type == "sds":
            eps_phi = noise
        elif self.cfg.guidance_type == "vsd":
            if self.cfg.vsd_camera_condition_type == "extrinsics":
                camera_condition = c2w
            elif self.cfg.vsd_camera_condition_type == "mvp":
                camera_condition = mvp_mtx
            elif self.cfg.vsd_camera_condition_type == "spherical":
                camera_condition = torch.stack(
                    [
                        torch.deg2rad(elevation),
                        torch.sin(torch.deg2rad(azimuth)),
                        torch.cos(torch.deg2rad(azimuth)),
                        camera_distances,
                    ],
                    dim=-1,
                )
            else:
                raise ValueError(
                    f"Unknown camera_condition_type {self.cfg.vsd_camera_condition_type}"
                )
            eps_phi = self.get_eps_phi(
                latents_noisy,
                t,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                camera_condition,
            )

            loss_train_phi = self.train_phi(
                latents,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                camera_condition,
            )

        if self.cfg.weighting_strategy == "dreamfusion":
            w = (1.0 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1.0
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )
        if self.cfg.normal_loss:
            grad_normal = w * (eps_pretrain_normal - noise_normal)
        else:
            grad_normal = 0.0
        if self.cfg.rgb_loss:
            grad_rgb = w * (eps_pretrain - eps_phi)
        else:
            grad_rgb = 0.0
        if self.grad_clip_val is not None:
            if self.cfg.normal_loss:
                grad_normal = grad_normal.clamp(-self.grad_clip_val, self.grad_clip_val)
            if self.cfg.rgb_loss:
                grad_rgb = grad_rgb.clamp(-self.grad_clip_val, self.grad_clip_val)
        # reparameterization trick:
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        if self.cfg.normal_loss:
            target_normal = (latents_normal - grad_normal).detach()
            loss_sd_normal = 0.5 * F.mse_loss(latents_normal, target_normal, reduction="sum") / batch_size
        else:
            loss_sd_normal = 0.0
        if self.cfg.rgb_loss:
            target_rgb = (latents - grad_rgb).detach()
            loss_sd_rgb = 0.5 * F.mse_loss(latents, target_rgb, reduction="sum") / batch_size
        else:
            loss_sd_rgb = 0.0

        if self.cfg.prob ==1.0:
            loss_sd = self.cfg.rgb_loss_scale*loss_sd_rgb + self.cfg.normal_loss_scale*loss_sd_normal
        elif random.random()<self.cfg.prob:
            loss_sd = loss_sd_rgb
        else:
            loss_sd = loss_sd_normal

        grad = self.cfg.rgb_loss_scale*grad_rgb + self.cfg.normal_loss_scale*grad_normal
        guidance_out = {
            "loss_sd": loss_sd,
            "grad_norm": grad.norm(),
            "timesteps": t,
            "min_step": self.min_step,
            "max_step": self.max_step,
            "latents": latents,
            "latents_1step_orig": latents_1step_orig,
            "rgb": rgb_BCHW.permute(0, 2, 3, 1),
            "weights": w,
            "lambdas": self.lambdas[t],
        }

        if self.cfg.return_rgb_1step_orig:
            with torch.no_grad():
                rgb_1step_orig = self.vae_decode(
                    self.pipe.vae, latents_1step_orig
                ).permute(0, 2, 3, 1)
            guidance_out.update({"rgb_1step_orig": rgb_1step_orig})
        
        if self.cfg.return_rgb_multistep_orig:
        #     camera_matrixs = rearrange(
        #     c2w[:, :3, :],
        #     "(b nv) three four -> b nv (three four)",
        #     nv=self.cfg.n_multi_views,
        # )[0]
        #     latents_multiview_noisy = rearrange(
        #     latents_noisy, "(b nv) c h w -> b c nv h w", nv=self.cfg.n_multi_views
        #     )
        #     latents_multiview_noisy_normal = rearrange(
        #     latents_noisy_normal, "(b nv) c h w -> b c nv h w", nv=self.cfg.n_multi_views
        #     )
        #     import pdb;pdb.set_trace()
        #     latents = torch.cat([latents_multiview_noisy[0].unsqueeze(0), latents_multiview_noisy_normal[0].unsqueeze(0)],dim=2)
        #     videos = self.pipe(0, "a blue motorcycle", camera_matrixs=camera_matrixs, latents=latents, video_length=4, height=256, width=256, num_inference_steps=50, guidance_scale=100, pred_normal=True).videos
        #     videos = rearrange(videos, "b c t h w -> t b c h w")
        #     image_list = []
        #     import torchvision
        #     import numpy as np
        #     from PIL import Image
            
        #     for i, x in enumerate(videos):
        #         x = torchvision.utils.make_grid(x, nrow=1)
        #         x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        #         x = (x * 255).numpy().astype(np.uint8)
        #         image_list.append(x)
        #     image_output_list = [Image.fromarray(img, mode='RGB') for img in image_list]
        #     target = Image.new('RGB', (256 * len(videos), 256 * 1))
        #     for row in range(1):
        #         for col in range(len(videos)):
        #             target.paste(image_output_list[len(videos)*row+col], (0 + 256*col, 0 + 256*row))
            
        #     target.save(f'test.jpg', quality=50)
        #     import pdb;pdb.set_trace()
            with self.set_scheduler(
                self.pipe,
                DPMSolverSinglestepScheduler,
                solver_order=1,
                num_train_timesteps=int(t[0]),
            ) as pipe:
                text_embeddings = prompt_utils.get_text_embeddings(
                    elevation,
                    azimuth,
                    camera_distances,
                    self.cfg.view_dependent_prompting,
                )
                text_embeddings_cond, text_embeddings_uncond = text_embeddings.chunk(2)
                with torch.cuda.amp.autocast(enabled=False):
                    latents_multistep_orig = pipe(
                        num_inference_steps=self.cfg.n_rgb_multistep_orig_steps,
                        guidance_scale=self.cfg.guidance_scale,
                        eta=1.0,
                        latents=latents_noisy.to(pipe.unet.dtype),
                        prompt_embeds=text_embeddings_cond.to(pipe.unet.dtype),
                        negative_prompt_embeds=text_embeddings_uncond.to(
                            pipe.unet.dtype
                        ),
                        cross_attention_kwargs={"scale": 0.0}
                        if self.vsd_share_model
                        else None,
                        output_type="latent",
                    ).images.to(latents.dtype)
            with torch.no_grad():
                rgb_multistep_orig = self.vae_decode(
                    self.pipe.vae, latents_multistep_orig
                )
            guidance_out.update(
                {
                    "latents_multistep_orig": latents_multistep_orig,
                    "rgb_multistep_orig": rgb_multistep_orig.permute(0, 2, 3, 1),
                }
            )

        if self.cfg.guidance_type == "vsd":
            guidance_out.update(
                {
                    "loss_train_phi": loss_train_phi,
                }
            )

        return guidance_out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.min_step = int(
            self.num_train_timesteps * C(self.cfg.min_step_percent, epoch, global_step)
        )
        self.max_step = int(
            self.num_train_timesteps * C(self.cfg.max_step_percent, epoch, global_step)
        )
