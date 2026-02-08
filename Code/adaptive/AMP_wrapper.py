from typing import Literal, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from AMP.enhancer.sd2 import SD2Enhancer
from AMP.utils.common import wavelet_reconstruction


class TrainablePathSD2(nn.Module, SD2Enhancer):
    """
    Thin wrapper over SD2Enhancer that keeps the diffusion path differentiable.
    AMP weights remain frozen; gradients flow back to the input tensor.
    """

    def __init__(self, generator_device: str | None = None, vae_device: str | None = None, *args, **kwargs):
        self.text_device = torch.device(kwargs.get("device", "cuda:0"))
        self.generator_device = torch.device(generator_device or self.text_device)
        self.vae_device = torch.device(vae_device or self.text_device)
        nn.Module.__init__(self)
        SD2Enhancer.__init__(self, *args, **kwargs)

    def forward(
        self,
        lq: torch.Tensor,
        prompt: str = "",
        scale_by: Literal["factor", "longest_side"] = "factor",
        upscale: int = 1,
        target_longest_side: Optional[int] = None,
    ) -> torch.Tensor:
        if scale_by == "factor":
            lq = F.interpolate(lq, scale_factor=upscale, mode="bicubic")
        elif scale_by == "longest_side":
            if target_longest_side is None:
                raise ValueError("target_longest_side must be specified when scale_by is 'longest_side'.")
            h, w = lq.shape[2:]
            if h >= w:
                new_h = target_longest_side
                new_w = int(w * (target_longest_side / h))
            else:
                new_w = target_longest_side
                new_h = int(h * (target_longest_side / w))
            lq = F.interpolate(lq, size=(new_h, new_w), mode="bicubic")
        else:
            raise ValueError(f"Unsupported scale_by method: {scale_by}")

        ref = lq
        h0, w0 = lq.shape[2:]

        lq = (lq * 2 - 1).to(dtype=self.weight_dtype, device=self.vae_device)
        h1, w1 = lq.shape[2:]

        vae_scale_factor = 8
        ph = (h1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - h1
        pw = (w1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - w1
        lq = F.pad(lq, (0, pw, 0, ph), mode="constant", value=0)
        z_lq = self.vae.encode(lq).latent_dist.sample()

        self.prepare_inputs(batch_size=len(lq), prompt=prompt)

        self.inputs["c_txt"]["text_embed"] = self.inputs["c_txt"]["text_embed"].to(self.generator_device)
        self.inputs["timesteps"] = self.inputs["timesteps"].to(self.generator_device)
        z = self.forward_generator(z_lq.to(self.generator_device))
        x = self.vae.decode(z.to(self.vae_device, dtype=self.weight_dtype)).sample.float()
        x = x[..., :h1, :w1]
        x = (x + 1) / 2
        x = F.interpolate(input=x, size=(h0, w0), mode="bicubic", antialias=True)
        x = wavelet_reconstruction(x, ref.to(device=self.vae_device))
        return x

    def freeze(self):
        for p in self.parameters(recurse=True):
            p.requires_grad = False
        return self

    def precompute_inputs(self, prompt: str, batch_size: int = 1):
        """
        Optional helper to cache text embeddings for a fixed prompt.
        """
        self.prepare_inputs(batch_size=batch_size, prompt=prompt)


    def init_text_models(self):
        from transformers import CLIPTokenizer, CLIPTextModel
        self.tokenizer = CLIPTokenizer.from_pretrained(self.base_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_model_path, subfolder="text_encoder", torch_dtype=self.weight_dtype
        ).to(self.text_device)
        self.text_encoder.eval().requires_grad_(False)
        self.init_coord_proj()

    def init_coord_proj(self):
        hidden = self.text_encoder.config.hidden_size
        self.coord_proj = nn.Sequential(nn.Linear(4, 256), nn.SiLU(), nn.Linear(256, hidden)).to(self.text_device)
        self.coord_proj.eval()
        for p in self.coord_proj.parameters():
            p.requires_grad = False

    def init_vae(self):
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model_path, subfolder="vae", torch_dtype=self.weight_dtype
        ).to(self.vae_device)
        self.vae.eval().requires_grad_(False)

    def init_generator(self):
        from diffusers import UNet2DConditionModel
        from peft import LoraConfig
        from safetensors.torch import load_file as safe_load

        self.G: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            self.base_model_path, subfolder="unet", torch_dtype=self.weight_dtype
        ).to(self.generator_device)
        target_modules = self.lora_modules
        G_lora_cfg = LoraConfig(
            r=self.lora_rank, lora_alpha=self.lora_rank, init_lora_weights="gaussian", target_modules=target_modules
        )
        self.G.add_adapter(G_lora_cfg)

        def _load_state_dict(path: str):
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                try:
                    return safe_load(path, device="cpu")
                except Exception as e:
                    raise RuntimeError(f"Failed to load AMP weight {path}: {e}")

        state_dict = _load_state_dict(self.weight_path)
        self.G.load_state_dict(state_dict, strict=False)
        input_keys = set(state_dict.keys())
        required_keys = set([k for k in self.G.state_dict().keys() if "lora" in k])
        missing = required_keys - input_keys
        unexpected = input_keys - required_keys
        assert required_keys == input_keys, f"Missing: {missing}, Unexpected: {unexpected}"
        self.G.eval().requires_grad_(False)
