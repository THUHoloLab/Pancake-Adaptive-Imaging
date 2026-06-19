import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer

from HYPIR.enhancer.base import BaseEnhancer


class SD2Enhancer(BaseEnhancer):
    def init_scheduler(self):
        self.scheduler = DDPMScheduler.from_pretrained(self.base_model_path, subfolder="scheduler")

    def init_text_models(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.base_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_model_path,
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.text_encoder.eval().requires_grad_(False)
        self.init_coord_proj()

    def init_coord_proj(self):
        hidden = self.text_encoder.config.hidden_size
        self.coord_proj = nn.Sequential(
            nn.Linear(4, 256),
            nn.SiLU(),
            nn.Linear(256, hidden),
        ).to(self.device)
        self.coord_proj.eval()
        for p in self.coord_proj.parameters():
            p.requires_grad = False

    def init_generator(self):
        self.G: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            self.base_model_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        target_modules = self.lora_modules
        G_lora_cfg = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self.G.add_adapter(G_lora_cfg)

        print(f"Load model weights from {self.weight_path}")
        state_dict = torch.load(self.weight_path, map_location="cpu", weights_only=False)
        self.G.load_state_dict(state_dict, strict=False)
        input_keys = set(state_dict.keys())
        required_keys = {k for k in self.G.state_dict().keys() if "lora" in k}
        missing = required_keys - input_keys
        unexpected = input_keys - required_keys
        assert required_keys == input_keys, f"Missing: {missing}, Unexpected: {unexpected}"

        self.G.eval().requires_grad_(False)

    def prepare_inputs(self, batch_size, prompt, tile_xy_wh=None):
        txt_ids = self.tokenizer(
            [prompt or ""] * batch_size,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_embed = self.text_encoder(txt_ids.to(self.device))[0]

        if tile_xy_wh is None:
            coords = torch.tensor([[0.5, 0.5, 1.0, 1.0]], device=self.device).repeat(batch_size, 1)
        else:
            x, y, width, height = tile_xy_wh
            if not torch.is_tensor(x):
                x = torch.tensor(x, device=self.device)
                y = torch.tensor(y, device=self.device)
                width = torch.tensor(width, device=self.device)
                height = torch.tensor(height, device=self.device)
            tile_w = tile_h = getattr(self, "tile_size", 512)
            x_center = (x + 0.5 * tile_w) / (width + 1e-6)
            y_center = (y + 0.5 * tile_h) / (height + 1e-6)
            wh = torch.stack(
                [
                    width / (width + height + 1e-6),
                    height / (width + height + 1e-6),
                ],
                dim=0,
            )
            coords = torch.stack([x_center, y_center, wh[0], wh[1]], dim=-1)
            coords = coords.unsqueeze(0).repeat(batch_size, 1)

        coord_embed = self.coord_proj(coords.float()).to(dtype=text_embed.dtype)
        coord_embed = coord_embed.unsqueeze(1).expand_as(text_embed)
        text_embed = text_embed + coord_embed

        self.inputs = {
            "c_txt": {"text_embed": text_embed},
            "timesteps": torch.full((batch_size,), self.model_t, dtype=torch.long, device=self.device),
        }

    def forward_generator(self, z_lq):
        z_in = z_lq * self.vae.config.scaling_factor
        eps = self.G(
            z_in,
            self.inputs["timesteps"],
            encoder_hidden_states=self.inputs["c_txt"]["text_embed"],
        ).sample
        z = self.scheduler.step(eps, self.coeff_t, z_in).pred_original_sample
        return z / self.vae.config.scaling_factor
