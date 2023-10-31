from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.exporters.base import Exporter, ExporterOutput
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.mesh import Mesh
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *
from threestudio.utils.saving import SaverMixin
import pymeshlab
import trimesh

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

def reduce_face(obj_path, out_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    current_face_num = ms.current_mesh().face_number()
    
    targetperc = min(50000 / current_face_num, 1)
    
    ms.meshing_decimation_quadric_edge_collapse(
        targetperc=targetperc,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=True,
        planarquadric=False,
        autoclean=True,
    )
    ms.save_current_mesh(out_path, save_vertex_normal=False)
    print(f"before_face_num: {current_face_num}, after: {ms.current_mesh().face_number()}")

    ms.clear()

def load_mesh_from_obj(obj_file_path: str) -> Mesh:
    trimesh_mesh = trimesh.load(obj_file_path, force="mesh", process=False)
    v_pos = torch.from_numpy(trimesh_mesh.vertices).cuda().float() #TODO: hard code for device
    t_pos_idx = torch.from_numpy(trimesh_mesh.faces).cuda() #TODO: hard code for device

    return Mesh(v_pos, t_pos_idx)

@threestudio.register("mesh-exporter")
class MeshExporter(Exporter):
    @dataclass
    class Config(Exporter.Config):
        fmt: str = "obj-mtl"  # in ['obj-mtl', 'obj'], TODO: fbx
        save_name: str = "model"
        save_normal: bool = False
        save_uv: bool = True
        save_texture: bool = True
        texture_size: int = 1024
        texture_format: str = "jpg"
        xatlas_chart_options: dict = field(default_factory=dict)
        xatlas_pack_options: dict = field(default_factory=dict)
        context_type: str = "gl"

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, self.device)

    def __call__(self) -> List[ExporterOutput]:
        mesh: Mesh = self.geometry.isosurface()

        # reduce faces
        save_helper = SaverMixin()
        save_helper.set_save_dir("tmp")
        save_helper.save_obj(filename="model", mesh=mesh)
        reduce_face("tmp/model.obj", "tmp/model.obj")
        # load new mesh
        mesh = load_mesh_from_obj("tmp/model.obj")

        if self.cfg.fmt == "obj-mtl":
            return self.export_obj_with_mtl(mesh)
        elif self.cfg.fmt == "obj":
            return self.export_obj(mesh)
        else:
            raise ValueError(f"Unsupported mesh export format: {self.cfg.fmt}")

    def export_obj_with_mtl(self, mesh: Mesh) -> List[ExporterOutput]:
        params = {
            "mesh": mesh,
            "save_mat": True,
            "save_normal": self.cfg.save_normal,
            "save_uv": self.cfg.save_uv,
            "save_vertex_color": False,
            "map_Kd": None,  # Base Color
            "map_Ks": None,  # Specular
            "map_Bump": None,  # Normal
            # ref: https://en.wikipedia.org/wiki/Wavefront_.obj_file#Physically-based_Rendering
            "map_Pm": None,  # Metallic
            "map_Pr": None,  # Roughness
            "map_format": self.cfg.texture_format,
        }

        if self.cfg.save_uv:
            mesh.unwrap_uv(self.cfg.xatlas_chart_options, self.cfg.xatlas_pack_options)

        if self.cfg.save_texture:
            threestudio.info("Exporting textures ...")
            assert self.cfg.save_uv, "save_uv must be True when save_texture is True"
            # clip space transform
            uv_clip = mesh.v_tex * 2.0 - 1.0
            # pad to four component coordinate
            uv_clip4 = torch.cat(
                (
                    uv_clip,
                    torch.zeros_like(uv_clip[..., 0:1]),
                    torch.ones_like(uv_clip[..., 0:1]),
                ),
                dim=-1,
            )
            # rasterize
            rast, _ = self.ctx.rasterize_one(
                uv_clip4, mesh.t_tex_idx, (self.cfg.texture_size, self.cfg.texture_size)
            )

            hole_mask = ~(rast[:, :, 3] > 0)

            def uv_padding(image):
                h, w = image.shape[:2]
                new_h = int(h / 2)
                new_w = int(w / 2)
                uv_padding_size = self.cfg.xatlas_pack_options.get("padding", 2)
                image_tmp = image.detach().cpu().numpy() * 255
                hole_mask_tmp = hole_mask.detach().cpu().numpy() * 255
                image_resize = cv2.resize(image_tmp.astype(np.uint8), (new_h, new_w))
                hole_mask_resize = cv2.resize(hole_mask_tmp.astype(np.uint8), (new_h, new_w))
                inpaint_image = (
                    cv2.inpaint(
                        image_resize.astype(np.uint8),
                        hole_mask_resize.astype(np.uint8),
                        uv_padding_size,
                        cv2.INPAINT_TELEA,
                    )
                    # / 255.0
                )
                inpaint_image = cv2.resize(inpaint_image, (h, w))
                result_image = np.copy(image_tmp)
                result_image[hole_mask_tmp != 0] = inpaint_image[hole_mask_tmp != 0]
                return torch.from_numpy(result_image / 255.0).to(image)

            # Interpolate world space position
            gb_pos, _ = self.ctx.interpolate_one(
                mesh.v_pos, rast[None, ...], mesh.t_pos_idx
            )
            gb_pos = gb_pos[0]

            # Sample out textures from MLP
            geo_out = self.geometry.export(points=gb_pos)
            mat_out = self.material.export(points=gb_pos, **geo_out)

            threestudio.info(
                "Perform UV padding on texture maps to avoid seams, may take a while ..."
            )

            if "albedo" in mat_out:
                params["map_Kd"] = uv_padding(mat_out["albedo"])
            else:
                threestudio.warn(
                    "save_texture is True but no albedo texture found, using default white texture"
                )
            if "metallic" in mat_out:
                params["map_Pm"] = uv_padding(mat_out["metallic"])
            if "roughness" in mat_out:
                params["map_Pr"] = uv_padding(mat_out["roughness"])
            if "bump" in mat_out:
                params["map_Bump"] = uv_padding(mat_out["bump"])
            # TODO: map_Ks
        return [
            ExporterOutput(
                save_name=f"{self.cfg.save_name}.obj", save_type="obj", params=params
            )
        ]

    def export_obj(self, mesh: Mesh) -> List[ExporterOutput]:
        params = {
            "mesh": mesh,
            "save_mat": False,
            "save_normal": self.cfg.save_normal,
            "save_uv": self.cfg.save_uv,
            "save_vertex_color": False,
            "map_Kd": None,  # Base Color
            "map_Ks": None,  # Specular
            "map_Bump": None,  # Normal
            # ref: https://en.wikipedia.org/wiki/Wavefront_.obj_file#Physically-based_Rendering
            "map_Pm": None,  # Metallic
            "map_Pr": None,  # Roughness
            "map_format": self.cfg.texture_format,
        }

        if self.cfg.save_uv:
            mesh.unwrap_uv(self.cfg.xatlas_chart_options, self.cfg.xatlas_pack_options)

        if self.cfg.save_texture:
            threestudio.info("Exporting textures ...")
            geo_out = self.geometry.export(points=mesh.v_pos)
            mat_out = self.material.export(points=mesh.v_pos, **geo_out)

            if "albedo" in mat_out:
                mesh.set_vertex_color(mat_out["albedo"])
                params["save_vertex_color"] = True
            else:
                threestudio.warn(
                    "save_texture is True but no albedo texture found, not saving vertex color"
                )

        return [
            ExporterOutput(
                save_name=f"{self.cfg.save_name}.obj", save_type="obj", params=params
            )
        ]
