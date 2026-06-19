from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_UFORMER_DEPTHS = (1, 2, 2, 2, 2, 2, 2, 2, 1)
DEFAULT_UFORMER_NUM_HEADS = (1, 2, 4, 8, 16, 16, 8, 4, 2)
DEFAULT_LORA_MODULES = (
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "conv",
    "conv1",
    "conv2",
    "conv_shortcut",
    "conv_out",
    "proj_in",
    "proj_out",
    "ff.net.2",
    "ff.net.0.proj",
)


@dataclass
class UformerConfig:

    img_size: int = 256
    in_chans: int = 3
    dd_in: int = 3
    embed_dim: int = 28
    depths: Sequence[int] = DEFAULT_UFORMER_DEPTHS
    num_heads: Sequence[int] = DEFAULT_UFORMER_NUM_HEADS
    win_size: int = 16
    mlp_ratio: float = 1.65
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    token_projection: str = "linear"
    token_mlp: str = "leff"
    modulator: bool = True
    scale: int = 1
    use_checkpoint: bool = True
    attn_chunk: int = 16
    debug_mem: bool = False


@dataclass
class DataConfig:
    dataroot_gt: Path
    dataroot_lq: Path
    filename_tmpl: str = "{}"


@dataclass
class TrainConfig:
    batch_size: int = 1
    num_workers: int = 2
    epochs: int = 10
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.99
    lambda_mid: float = 0.3
    lambda_hypir: float = 0.7
    log_every: int = 10
    ckpt_every: int = 1000
    output_dir: Path = Path("adaptive_outputs")
    val_freq: int = 0
    save_val_img: bool = False
    pretrain_path: Path | None = None
    strict_load: bool = False
    val_max_samples: int | None = 10
    initial_val: bool = True
    total_iters: int | None = None
    checkpoint_path: Path | None = None


@dataclass
class HypirConfig:
    base_model_path: Path
    weight_path: Path
    lora_rank: int = 256
    model_t: int = 200
    coeff_t: int = 200
    lora_modules: Sequence[str] = DEFAULT_LORA_MODULES
    text_device: str = "cuda:1"
    generator_device: str | None = None
    vae_device: str | None = None


@dataclass
class SystemConfig:
    device_uformer: str = "cuda:0"
    device_hypir: str = "cuda:1"
    mixed_precision: bool = False


@dataclass
class PipelineConfig:
    data: DataConfig
    train: TrainConfig
    uformer: UformerConfig = field(default_factory=UformerConfig)
    hypir: HypirConfig = field(default_factory=lambda: HypirConfig(
        base_model_path=Path("models/HYPIR/stable-diffusion-2-1-base"),
        weight_path=Path("models/HYPIR/HYPIR_sd2.pth"),
    ))
    system: SystemConfig = field(default_factory=SystemConfig)
    run_name: str = "adaptive_run"
