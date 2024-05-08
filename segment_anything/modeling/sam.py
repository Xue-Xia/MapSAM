# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic
import numpy as np

from typing import Any, Dict, List, Tuple

from .common import LayerNorm2d
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


def point_selection(mask, topk=1):
    # Top-1 point selection
    w, h = mask.shape
    topk_xy = mask.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = torch.tensor([1] * topk, device=mask.device)

    # Top-last point selection
    last_xy = mask.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = torch.tensor([0] * topk, device=mask.device)

    return topk_xy, topk_label, last_xy, last_label

def get_max_dist_point(mask_tensor):
    # Compute the distance transform of the binary mask
    kernel = torch.tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], dtype=torch.float32, device=mask_tensor.device)
    dist_transform = F.conv2d(mask_tensor, kernel, padding=1)

    # Find the location of the point with maximum distance value
    max_dist = torch.max(dist_transform)
    max_dist_idx = torch.where(dist_transform == max_dist)
    coord = torch.tensor([max_dist_idx[2][0], max_dist_idx[1][0]], dtype=torch.float32, device=mask_tensor.device).unsqueeze(0).unsqueeze(0)  # (x, y) coordinates
    label = torch.tensor([1], dtype=torch.float32, device=mask_tensor.device).unsqueeze(0)
    point = (coord, label)

    return point

def get_max_dist_points(mask_tensor, num_points=2):
    # Compute the distance transform of the binary mask
    kernel = torch.tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], dtype=torch.float32, device=mask_tensor.device)
    dist_transform = F.conv2d(mask_tensor, kernel, padding=1)

    # Find the top num_points maximum distance values and their corresponding indices
    max_dists, max_dist_indices = torch.topk(dist_transform.view(-1), k=num_points)

    # Initialize arrays for coordinates
    coords = torch.zeros((1, num_points, 2), dtype=torch.float32, device=mask_tensor.device)  # Added extra dimension

    # Create a labels tensor with all ones, matching the number of points
    labels = torch.ones(num_points, dtype=torch.float32, device=mask_tensor.device).unsqueeze(0)

    # Convert indices to coordinates
    for i in range(num_points):
        y_coord = max_dist_indices[i] // dist_transform.shape[3]
        x_coord = max_dist_indices[i] % dist_transform.shape[3]
        coords[0, i, :] = torch.tensor([y_coord, x_coord], dtype=torch.float32, device=mask_tensor.device)

    point = (coords, labels)
    return point


class SPGen(nn.Module):
    def __init__(self):
        super(SPGen, self).__init__()

        self.hint = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, stride=1),
                                 LayerNorm2d(64),
                                 nn.GELU(),
                                 nn.Conv2d(64, 16, kernel_size=1, stride=1),
                                 LayerNorm2d(16),
                                 nn.GELU(),
                                 nn.Conv2d(16, 1, kernel_size=1))

    def forward(self, x):
        x = self.hint(x)
        return x

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.spgen = SPGen()
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input, multimask_output, image_size):
        if isinstance(batched_input, list):
            outputs = self.forward_test(batched_input, multimask_output)
        else:
            outputs = self.forward_train(batched_input, multimask_output, image_size)
        return outputs

    def forward_train(self, batched_input, multimask_output, image_size):
        input_images = self.preprocess(batched_input)
        image_embeddings = self.image_encoder(input_images)
        coarse_mask = self.spgen(image_embeddings)
        coarse_mask_up4 = F.interpolate(coarse_mask, size=(int(image_size/4), int(image_size/4)), mode="bilinear")
        coarse_mask_up16 = F.interpolate(coarse_mask, size=(image_size, image_size), mode="bilinear")

        spgen_prob = torch.sigmoid(coarse_mask_up4.detach())
        spgen_prob[spgen_prob >= 0.5] = 1
        spgen_prob[spgen_prob < 0.5] = 0

        outputs = {
                'masks': [],
                'iou_predictions': [],
                'low_res_logits': []
            }
        for idx in range(batched_input.shape[0]): # for each batch

            # Obtain the target guidance for cross-attention layers
            attn_mask = (coarse_mask[idx].sigmoid().unsqueeze(0).unsqueeze(0).flatten(3) < 0.5).bool()
            attn_mask = attn_mask.detach()

            # Positive-negative location prior
            topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(coarse_mask_up16[idx].squeeze(0), topk=1)
            topk_xy = torch.cat([topk_xy_i, last_xy_i], dim=0)
            topk_label = torch.cat([topk_label_i, last_label_i], dim=0)
            fg_points = (topk_xy.unsqueeze(0), topk_label.unsqueeze(0))

            # use distance transform to find a point inside the mask
            # fg_points = get_max_dist_points(coarse_mask_up16[idx].unsqueeze(0))

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=fg_points,
                boxes=None,
                masks=spgen_prob[idx].unsqueeze(0),
            )

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings[idx].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                attn_mask=attn_mask,
                target_embedding=None
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=(image_size, image_size),
                original_size=(image_size, image_size)
            )
            outputs['masks'].append(masks)
            outputs['iou_predictions'].append(iou_predictions)
            outputs['low_res_logits'].append(low_res_masks)

        outputs['masks'] = torch.stack(outputs['masks'], dim=0).squeeze(1)
        outputs['iou_predictions'] = torch.stack(outputs['iou_predictions'], dim=0).squeeze(1)
        outputs['low_res_logits'] = torch.stack(outputs['low_res_logits'], dim=0).squeeze(1)

        return outputs, coarse_mask_up4

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        coarse_mask = self.spgen(image_embeddings)
        coarse_mask_up4 = self.up4(coarse_mask)
        coarse_mask_up16 = self.up4(coarse_mask_up4)

        spgen_prob = torch.sigmoid(coarse_mask_up4.detach())
        spgen_prob[spgen_prob >= 0.75] = 1
        spgen_prob[spgen_prob < 0.75] = 0

        outputs = []
        for image_record, curr_embedding, curr_mask, curr_spgen in zip(batched_input, image_embeddings, coarse_mask_up16, spgen_prob):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = get_max_dist_point(curr_mask.unsqueeze(0))
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=curr_spgen.unsqueeze(0),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

