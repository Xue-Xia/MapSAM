from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic


class _DoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            weight,
            bias,
            dim,
            # dim_out,
            # m,
            # lora_A,
            # lora_B
            m_q,
            m_v,
            lora_A_q,
            lora_B_q,
            lora_A_v,
            lora_B_v
    ):
        super().__init__()
        # self.qkv = qkv
        # self.dim = qkv.in_features
        # self.w_identity = torch.eye(qkv.in_features)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.dim = dim
        # self.dim_out = dim_out
        # self.m = m
        # self.lora_A = lora_A
        # self.lora_B = lora_B
        self.m_q = nn.Parameter(m_q)
        self.m_v = nn.Parameter(m_v)
        self.lora_A_q = nn.Parameter(lora_A_q)
        self.lora_B_q = nn.Parameter(lora_B_q)
        self.lora_A_v = nn.Parameter(lora_A_v)
        self.lora_B_v = nn.Parameter(lora_B_v)

    def forward(self, x):
        lora_q = torch.matmul(self.lora_A_q, self.lora_B_q)
        adapted_q = self.weight[:self.dim, :] + lora_q
        column_norm_q = adapted_q.norm(p=2, dim=0, keepdim=True)
        norm_adapted_q = adapted_q / column_norm_q
        calc_weights_q = self.m_q * norm_adapted_q

        lora_v = torch.matmul(self.lora_A_v, self.lora_B_v)
        adapted_v = self.weight[-self.dim:, :] + lora_v
        column_norm_v = adapted_v.norm(p=2, dim=0, keepdim=True)
        norm_adapted_v = adapted_v / column_norm_v
        calc_weights_v = self.m_v * norm_adapted_v

        # lora = torch.matmul(self.lora_A, self.lora_B)
        # adapted = self.weight + lora
        # column_norm = adapted.norm(p=2, dim=0, keepdim=True)
        # norm_adapted = adapted / column_norm
        # new_weights = self.m * norm_adapted

        new_weights = torch.cat((calc_weights_q, self.weight[self.dim:-self.dim, :], calc_weights_v), dim=0)

        return F.linear(x, new_weights, self.bias)


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        self.ms = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            # self.dim_out = w_qkv_linear.out_features
            self.weight = w_qkv_linear.weight.data.clone()
            self.bias = w_qkv_linear.bias.data.clone()
            self.m_q = w_qkv_linear.weight[: self.dim, :].norm(p=2, dim=0, keepdim=True)
            self.m_v = w_qkv_linear.weight[-self.dim:, :].norm(p=2, dim=0, keepdim=True)
            # self.m_q = nn.Parameter(w_qkv_linear.weight[: self.dim, :].norm(p=2, dim=0, keepdim=True))
            # self.m_v = nn.Parameter(w_qkv_linear.weight[-self.dim:, :].norm(p=2, dim=0, keepdim=True))
            # self.m = nn.Parameter(w_qkv_linear.weight.norm(p=2, dim=0, keepdim=True))

            std_dev = 1 / torch.sqrt(torch.tensor(r).float())
            self.lora_A_q = torch.randn(self.dim, r) * std_dev
            self.lora_B_q = torch.zeros(r, self.dim)
            self.lora_A_v = torch.randn(self.dim, r) * std_dev
            self.lora_B_v = torch.zeros(r, self.dim)
            # self.lora_A_q = nn.Parameter(torch.randn(self.dim, r) * std_dev)
            # self.lora_B_q = nn.Parameter(torch.zeros(r, self.dim))
            # self.lora_A_v = nn.Parameter(torch.randn(self.dim, r) * std_dev)
            # self.lora_B_v = nn.Parameter(torch.zeros(r, self.dim))

            # self.lora_A = nn.Parameter(torch.randn(self.dim_out, r) * std_dev)
            # self.lora_B = nn.Parameter(torch.zeros(r, self.dim_in))

            # w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            # w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            # w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            # w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(self.lora_A_q)
            self.w_Bs.append(self.lora_B_q)
            self.ms.append(self.m_q)
            self.w_As.append(self.lora_A_v)
            self.w_Bs.append(self.lora_B_v)
            self.ms.append(self.m_v)

            # self.w_As.append(self.lora_A)
            # self.w_Bs.append(self.lora_B)
            # self.ms.append(self.m)
            blk.attn.qkv = _DoRA_qkv(
                self.weight,
                self.bias,
                self.dim,
                # self.dim_out,
                # self.m,
                # self.lora_A,
                # self.lora_B
                self.m_q,
                self.m_v,
                self.lora_A_q,
                self.lora_B_q,
                self.lora_A_v,
                self.lora_B_v
            )

        # for param in sam_model.prompt_encoder.parameters():
        #     param.requires_grad = False
        # for param in sam_model.mask_decoder.transformer.parameters():
        #     param.requires_grad = False

        # self.reset_parameters()
        self.sam = sam_model

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        # num_layer = len(self.w_As)  # actually, it is half
        # a_tensors = {f"w_a_{i:03d}": self.w_As[i] for i in range(num_layer)}
        # b_tensors = {f"w_b_{i:03d}": self.w_Bs[i] for i in range(num_layer)}
        # m_tensors = {f"m_{i:03d}": self.ms[i] for i in range(num_layer)}

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}
        spgen_tensors = {}
        lora_tensors = {}
        m_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value
            if 'spgen' in key:
                spgen_tensors[key] = value

            if 'qkv' in key:
                if 'm_' in key:
                    m_tensors[key] = value
                elif 'lora' in key:
                    lora_tensors[key] = value

        # merged_dict = {**a_tensors, **b_tensors, **m_tensors, **spgen_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        merged_dict = {**m_tensors, **lora_tensors, **spgen_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)

        # for i, w_A_linear in enumerate(self.w_As):
        #     saved_key = f"w_a_{i:03d}"
        #     saved_tensor = state_dict[saved_key]
        #     w_A_linear.weight = Parameter(saved_tensor)
        #
        # for i, w_B_linear in enumerate(self.w_Bs):
        #     saved_key = f"w_b_{i:03d}"
        #     saved_tensor = state_dict[saved_key]
        #     w_B_linear.weight = Parameter(saved_tensor)

        # for i, m in enumerate(self.ms):
        #     saved_key = f"m_{i:03d}"
        #     saved_tensor = state_dict[saved_key]
        #     m = Parameter(saved_tensor)
        #
        # for i, lora_A in enumerate(self.w_As):
        #     saved_key = f"w_a_{i:03d}"
        #     saved_tensor = state_dict[saved_key]
        #     lora_A = Parameter(saved_tensor)
        #
        # for i, lora_B in enumerate(self.w_Bs):
        #     saved_key = f"w_b_{i:03d}"
        #     saved_tensor = state_dict[saved_key]
        #     lora_B = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load dora
        m_keys = [k for k in sam_keys if 'qkv' in k and 'm_' in k]
        m_values = [state_dict[k] for k in m_keys]
        m_new_state_dict = {k: v for k, v in zip(m_keys, m_values)}
        sam_dict.update(m_new_state_dict)

        lora_keys = [k for k in sam_keys if 'qkv' in k and 'lora' in k]
        lora_values = [state_dict[k] for k in lora_keys]
        lora_new_state_dict = {k: v for k, v in zip(lora_keys, lora_values)}
        sam_dict.update(lora_new_state_dict)

        # load self prompt generator
        spgen_keys = [k for k in sam_keys if 'spgen' in k]
        spgen_values = [state_dict[k] for k in spgen_keys]
        spgen_new_state_dict = {k: v for k, v in zip(spgen_keys, spgen_values)}
        sam_dict.update(spgen_new_state_dict)

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        self.sam.load_state_dict(sam_dict)
    #
    # def reset_parameters(self) -> None:
    #     for w_A in self.w_As:
    #         nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
    #     for w_B in self.w_Bs:
    #         nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output, image_size):
        return self.sam(batched_input, multimask_output, image_size)

    # def forward(self, x: Tensor) -> Tensor:
    #     return self.lora_vit(x)


if __name__ == "__main__":
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    lora_sam = LoRA_Sam(sam, 4)
    lora_sam.sam.image_encoder(torch.rand(size=(1, 3, 1024, 1024)))
