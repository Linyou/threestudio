import random
from contextlib import contextmanager
from dataclasses import dataclass, field

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

from extern.mvdream.models.camera_utils import convert_opengl_to_blender, normalize_camera
from extern.mvdream.models.unet import UNet3DConditionModel
from extern.mvdream.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from einops import rearrange
import numpy as np

@threestudio.register("multiview-unified-guidance")
class MultiViewUnifiedGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        # guidance type, in ["sds", "vsd"]
        guidance_type: str = "sds"

        pretrained_model_name_or_path: str = "/mnt/pfs/users/liuzexiang/Code/tune-a-video/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/"
        unet_path: str = "/mnt/pfs/users/liyangguang/vastcode/tune-a-video/outputs/render-group-1-2-laion-ga-3dword-gray-laion20M/"
        guidance_scale: float = 100.0
        weighting_strategy: str = "dreamfusion"
        view_dependent_prompting: bool = False
        n_multi_views: int = 4

        min_step_percent: Any = 0.02
        max_step_percent: Any = 0.98
        grad_clip: Optional[Any] = None

        recon_loss: bool = True
        recon_std_rescale: float = 0.5

        camera_condition_type: str = "rotation"

        return_rgb_1step_orig: bool = False
        return_rgb_multistep_orig: bool = False
        n_rgb_multistep_orig_steps: int = 4

        half_precision_weights: bool = False
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)


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

        unet = UNet3DConditionModel.from_pretrained(
            self.cfg.unet_path, subfolder="unet", torch_dtype=self.weights_dtype
        ).to(self.device)
        pipe = TuneAVideoPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, unet=unet, **pipe_kwargs
        ).to(self.device)

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
        pipe.unet.eval()

        enable_gradient(pipe.vae, enabled=False)
        enable_gradient(pipe.unet, enabled=False)

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

    @torch.cuda.amp.autocast(enabled=False)
    def vae_encode(
        self, vae: AutoencoderKL, imgs: Float[Tensor, "B 3 H W"], mode=False
    ) -> Float[Tensor, "B 4 Hl Wl"]:
        # expect input in [-1, 1]
        input_dtype = imgs.dtype
        posterior = vae.encode(imgs.to(vae.dtype)).latent_dist
        if mode:
            latents = posterior.mode()
        else:
            latents = posterior.sample()
        latents = latents * vae.config.scaling_factor
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
            camera = normalize_camera(camera)[...,:12]
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(f"Unknown camera_condition_type={self.cfg.camera_condition_type}")
        return camera


    def get_eps_pretrain(
        self,
        latents_noisy: Float[Tensor, "B 4 Hl Wl"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
    ) -> Float[Tensor, "B 4 Hl Wl"]:
        # batch_size = latents_noisy.shape[0]
        # batch_unet = batch_size // self.cfg.n_multi_views

        latents_multiview_noisy: Float[Tensor, "Bdivn 4 Nv Hl Wl"] = rearrange(
            latents_noisy, "(b nv) c h w -> b c nv h w", nv=self.cfg.n_multi_views
        ) #[2, 4, 4, 32, 32]

        camera = self.get_camera_cond(c2w)
        camera_matrixs: Float[Tensor, "Bdivn Nv 12"] = rearrange(
            camera,
            "(b nv) three -> b nv three",
            nv=self.cfg.n_multi_views,
        ) #[2, 4, 12]

        # import pdb
        # pdb.set_trace()
        # camera_matrixs: Float[Tensor, "Bdivn Nv 12"] = rearrange(
        #     c2w[:, :3, :],
        #     "(b nv) three four -> b nv (three four)",
        #     nv=self.cfg.n_multi_views,
        # )

        assert not prompt_utils.use_perp_neg

        elevation_multiview = rearrange(
            elevation, "(b nv) -> b nv", nv=self.cfg.n_multi_views
        )
        text_embeddings = prompt_utils.get_text_embeddings( #4, 77, 1024
            elevation_multiview[:, 0], None, None, self.cfg.view_dependent_prompting
        )

        with torch.no_grad():
            noise_pred = self.pipe.unet(
                torch.cat([latents_multiview_noisy] * 2, dim=0),
                torch.cat([t] * 2, dim=0),
                encoder_hidden_states=text_embeddings,
                camera_matrixs=torch.cat([camera_matrixs] * 2, dim=0),
            ).sample

        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
            
        noise_pred = rearrange(noise_pred, "b c nv h w -> (b nv) c h w")

        return noise_pred

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        assert batch_size % self.cfg.n_multi_views == 0
        batch_unet = batch_size // self.cfg.n_multi_views

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 Hl Wl"]
        if rgb_as_latents:
            # treat input rgb as latents
            # input rgb should be in range [-1, 1]
            latents = F.interpolate(
                rgb_BCHW, (32, 32), mode="bilinear", align_corners=False
            )
        else:
            # treat input rgb as rgb
            # input rgb should be in range [0, 1]
            rgb_BCHW = F.interpolate(
                rgb_BCHW, (256, 256), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.vae_encode(self.pipe.vae, rgb_BCHW * 2.0 - 1.0)
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
        t = t_one.repeat(batch_size) #8

        # sample noise
        noise = torch.randn_like(latents) #[8, 4, 32, 32]
        latents_noisy = self.scheduler.add_noise(latents, noise, t) #[8, 4, 32, 32]

        eps_pretrain = self.get_eps_pretrain(
            latents_noisy,
            t_one.repeat(batch_unet),
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
            c2w,
        )

        latents_1step_orig = (
            1
            / self.alphas[t].view(-1, 1, 1, 1)
            * (latents_noisy - self.sigmas[t].view(-1, 1, 1, 1) * eps_pretrain)
        ).detach()

        if self.cfg.guidance_type == "sds":
            eps_phi = noise
        else:
            raise ValueError(
                f"Unknown guidance_type: {self.cfg.guidance_type}"
            )

        if self.cfg.weighting_strategy == "dreamfusion":
            # w = (1.0 - self.alphas[t]).view(-1, 1, 1, 1)
            w = (1.0 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1.0
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (eps_pretrain - eps_phi)

        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        grad = torch.nan_to_num(grad)

        # reparameterization trick:
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss_sd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

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
            import pdb;pdb.set_trace()
            import cv2 
            import numpy as np    
            tmp = rgb_1step_orig.detach().cpu().numpy()
            rgb_o0 = np.uint8(tmp[0, :, :, :] * 255)
            rgb_o1 = np.uint8(tmp[1, :, :, :] * 255)
            cv2.imwrite('debug/step1_0.jpg', rgb_o0)
            cv2.imwrite('debug/step1_0.jpg', rgb_o1)
            exit()
        if self.cfg.return_rgb_multistep_orig:
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
