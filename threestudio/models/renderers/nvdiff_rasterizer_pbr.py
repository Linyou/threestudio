from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("nvdiff-rasterizer-pbr")
class NVDiffRasterizerPBR(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"
        positions_jitter: bool = False

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        c2w: Float[Tensor, "B 4 4"] = None,
        render_rgb: bool = True,
        render_depth: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]
        mesh = self.geometry.isosurface()

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        out = {"opacity": mask_aa, "mesh": mesh, "face_id": rast[..., 3:]}


        selector = mask[..., 0]
        gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
        gb_viewdirs = F.normalize(
            gb_pos - camera_positions[:, None, None, :], dim=-1
        )
        
        gb_rgb_bg = self.background(dirs=gb_viewdirs)
        if self.training:
            sel_rgb_bg = gb_rgb_bg
        else:
            sel_rgb_bg = torch.zeros_like(gb_rgb_bg) + 0.5
            sel_rgb_bg = torch.tensor([233, 239, 248], dtype=torch.float32).to(sel_rgb_bg) / 255.0
        
        
        gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
        gb_normal = F.normalize(gb_normal, dim=-1)
        gb_normal_aa = torch.lerp(
            sel_rgb_bg, (gb_normal + 1.0) / 2.0, mask.float()
        )
        gb_normal_aa = self.ctx.antialias(
            gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

        # TODO: make it clear whether to compute the normal, now we compute it in all cases
        # consider using: require_normal_computation = render_normal or (render_rgb and material.requires_normal)
        # or
        # render_normal = render_normal or (render_rgb and material.requires_normal)

        if render_rgb:

            gb_light_positions = light_positions[:, None, None, :].expand(
                -1, height, width, -1
            )

            positions = gb_pos[selector]    
            gb_select_dir = gb_viewdirs[selector]
            geo_out = self.geometry(
                positions, 
                output_normal=False, 
                # input_dir=gb_select_dir
            )

            extra_geo_info = {}
            if self.material.requires_normal:
                extra_geo_info["shading_normal"] = gb_normal[selector]
            if self.material.requires_tangent:
                gb_tangent, _ = self.ctx.interpolate_one(
                    mesh.v_tng, rast, mesh.t_pos_idx
                )
                gb_tangent = F.normalize(gb_tangent, dim=-1)
                extra_geo_info["tangent"] = gb_tangent[selector]

            results_dict = self.material(
                viewdirs=gb_select_dir,
                positions=positions,
                light_positions=gb_light_positions[selector],
                step=kwargs.get("step", 0),
                render_visiblity=kwargs.get("render_visiblity", False),
                azimuth=kwargs.get("azimuth", None),
                detach_albedo=kwargs.get("detach_albedo", False),
                **extra_geo_info,
                **geo_out
            )
            
            if self.cfg.positions_jitter:
                jitter = torch.normal(
                    mean=0, std=0.05,
                    size=positions.shape,
                    device=positions.device,
                )
                positions_jitter = positions + jitter
                geo_out_jitter = self.geometry(
                    positions_jitter, 
                    output_normal=False, 
                    # input_dir=gb_select_dir
                )
                
                results_jitter = self.material(
                    viewdirs=gb_select_dir,
                    positions=geo_out_jitter,
                    light_positions=gb_light_positions[selector],
                    step=kwargs.get("step", 0),
                    render_visiblity=kwargs.get("render_visiblity", False),
                    azimuth=kwargs.get("azimuth", None),
                    detach_albedo=kwargs.get("detach_albedo", False),
                    **extra_geo_info,
                    **geo_out_jitter
                )
                
                # import pdb; pdb.set_trace()
                out.update({
                    # "raw_albedo": results_dict_jitter['albedo'],
                    "kd_grad": torch.sum(torch.abs(
                        results_jitter['albedo'] - results_dict['albedo']
                    ), dim=-1),
                    "roughness_grad": torch.abs(
                        results_jitter['roughness'] - results_dict['roughness']
                    ),
                    "metallic_grad": torch.abs(
                        results_jitter['specular_metallic'] - results_dict['specular_metallic']
                    ),
                })
            
            out.update({
                "raw_color": results_dict['color'],
                "raw_albedo": results_dict['albedo'],
                "raw_roughness": results_dict['roughness'],
                "raw_metallic": results_dict['specular_metallic'],
            })
            
            
            def post_process(
                    x_fg: Float[Tensor, "*B Nf"], 
                    dim: int,
                    use_black_bg: bool = False,
                    use_II_bg: bool = False,
                ):
                if use_black_bg:
                    x_bg = torch.zeros_like(gb_rgb_bg)
                elif use_II_bg and self.training:
                    x_bg = kwargs["background_II"](dirs=gb_viewdirs)
                else:
                    x_bg = sel_rgb_bg
                    
                gb_x_fg = torch.zeros(
                    batch_size, 
                    height, 
                    width, 
                    dim
                ).to(x_bg)
                gb_x_fg[selector] = x_fg
                gb_x = torch.lerp(x_bg, gb_x_fg, mask.float())
                gb_x_aa = self.ctx.antialias(
                    gb_x, rast, v_pos_clip, mesh.t_pos_idx
                )
                
                return gb_x_aa
            
            for key in results_dict.keys():
                if key != "raw_albedo":
                    x_fg = results_dict[key]
                    out.update({
                        f"comp_{key}": post_process(
                            x_fg, 
                            x_fg.shape[-1],
                            use_black_bg=False,
                            use_II_bg=("background_II" in kwargs) and (key == "albedo")
                        )
                    })
                
                
            # out.update({
            #     f"raw_albedo": results_dict['albedo']
            # })


            # gb_rgb_fg = torch.zeros(
            #     batch_size, 
            #     height, 
            #     width, 
            #     3
            # ).to(rgb_fg)
            # gb_rgb_fg[selector] = rgb_fg
            # gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())

            # gb_roughness = torch.zeros(batch_size, height, width, 1).to(roughness)
            # gb_metallic = torch.zeros(batch_size, height, width, 1).to(metallic)
            # gb_albedo = torch.zeros(batch_size, height, width, 3).to(albedo)
            # gb_roughness[selector] = roughness
            # gb_metallic[selector] = metallic
            # gb_albedo[selector] = albedo

            # white_bg = torch.ones_like(gb_rgb_bg)
            # gb_roughness = torch.lerp(white_bg, gb_roughness, mask.float())
            # gb_metallic = torch.lerp(white_bg, gb_metallic, mask.float())
            # gb_albedo = torch.lerp(white_bg, gb_albedo, mask.float())
            # gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)
            # gb_roughness_aa = self.ctx.antialias(gb_roughness, rast, v_pos_clip, mesh.t_pos_idx)
            # gb_metallic_aa = self.ctx.antialias(gb_metallic, rast, v_pos_clip, mesh.t_pos_idx)
            # gb_albedo_aa = self.ctx.antialias(gb_albedo, rast, v_pos_clip, mesh.t_pos_idx)

            # out.update({
            #     "comp_rgb": gb_rgb_aa, 
            #     "comp_albedo": gb_albedo_aa,
            #     "comp_roughness": gb_roughness_aa,
            #     "comp_metallic": gb_metallic_aa,
            #     "comp_rgb_bg": gb_rgb_bg,
            # })
            
        if render_depth:
            # calculate w2c from c2w: R' = Rt, t' = -Rt * t
            # mathematically equivalent to (c2w)^-1
            
            # xyz map
            world_coordinates, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
            out.update({"comp_xyz": world_coordinates})
            # true normal map
            gb_normal_normalize = torch.lerp(
            torch.zeros_like(gb_normal), gb_normal, mask.float()
            )
            out.update({"comp_normal_normalize": gb_normal_normalize})  # in [-1, 1]

            w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
            w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
            w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
            w2c[:, 3, 3] = 1.0
            # render depth
            world_coordinates_homogeneous = torch.cat([world_coordinates, torch.ones_like(world_coordinates[..., :1])], dim=-1) # shape: [batch_size, height, width, 4]
            camera_coordinates_homogeneous = torch.einsum('bijk,bkl->bijl', world_coordinates_homogeneous, w2c.transpose(-2, -1)) # shape: [batch_size, height, width, 4]
            camera_coordinates = camera_coordinates_homogeneous[..., :3] # shape: [batch_size, height, width, 3]
            depth = camera_coordinates[..., 2] # shape: [batch_size, height, width]

            mask_depth = mask.squeeze(2).squeeze(3)
            foreground_depth = depth[mask_depth]

            if foreground_depth.numel() > 0:
                min_depth = torch.min(foreground_depth)
                max_depth = torch.max(foreground_depth)
                normalized_depth = (depth - min_depth) / (max_depth - min_depth+1e-6)
            else:
                normalized_depth = (depth)*0

            background_value = 0
            depth_blended = normalized_depth * mask_depth.float() + background_value * (1 - mask_depth.float())

            out.update({"depth": depth_blended.unsqueeze(3)})

        return out
