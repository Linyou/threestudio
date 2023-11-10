import cv2
import numpy as np

import torch
import torch.nn.functional as F
from controlnet_aux import NormalBaeDetector, CannyDetector, PidiNetDetector, HEDdetector

from threestudio.utils.misc import enable_gradient
from threestudio.utils.typing import *


class ControlNetConditionPreprocessor:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image: Float[Tensor, "B H W C"]) -> Float[Tensor, "B 3 512 512"]:
        raise NotImplementedError


class IdentityPreprocessor(ControlNetConditionPreprocessor):
    def __call__(self, image: Float[Tensor, "B H W C"]) -> Float[Tensor, "B 3 512 512"]:
        if image.shape[-1] == 1:
            image = image.repeat(1, 1, 1, 3)
        image = image.permute(0, 3, 1, 2)
        image = F.interpolate(
            image, (512, 512), mode="bilinear", align_corners=False
        )
        return image

class RGB2CannyPreprocessor(ControlNetConditionPreprocessor):
    def __init__(self, canny_lower_bound: int = 50, canny_upper_bound: int = 100):
        super().__init__()
        self.canny_lower_bound = canny_lower_bound
        self.canny_upper_bound = canny_upper_bound
        self.detector = CannyDetector()
    
    def __call__(self, image: Float[Tensor, "B H W C"]) -> Float[Tensor, "B 3 512 512"]:
        assert image.shape[-1] == 3
        images = []
        for i in range(image.shape[0]):
            image_ = (image[i].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            blurred_img = cv2.blur(image_, ksize=(5, 5))
            detected_map = self.detector(blurred_img, self.canny_lower_bound, self.canny_upper_bound)
            control = torch.from_numpy(np.array(detected_map)).float().to(image.device) / 255.0
            images.append(control)
        images = torch.stack(images, dim=0)
        images = images.unsqueeze(-1).repeat(1, 1, 1, 3)
        images = images.permute(0, 3, 1, 2)
        images = F.interpolate(
            images, (512, 512), mode="bilinear", align_corners=False
        )        
        return images

class RGB2NormalPreprocessor(ControlNetConditionPreprocessor):
    def __init__(self):
        super().__init__()
        self.detector = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        self.detector.model.to(self.device)
        enable_gradient(self.detector.model, enabled=False)
        self.detector.model.eval()
    
    def __call__(self, image: Float[Tensor, "B H W C"]) -> Float[Tensor, "B 3 512 512"]:
        assert image.shape[-1] == 3
        images = []
        for i in range(image.shape[0]):
            image_ = (image[i].detach().cpu().numpy() * 255).astype(np.uint8)
            detected_map = self.detector(image_)
            control = torch.from_numpy(np.array(detected_map)).float().to(image.device) / 255.0
            images.append(control)
        images = torch.stack(images, dim=0)
        images = images.permute(0, 3, 1, 2)
        images = F.interpolate(
            images, (512, 512), mode="bilinear", align_corners=False
        )
        return images