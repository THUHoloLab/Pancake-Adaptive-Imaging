from typing import Any, Dict

import torch


def build_deconv(config: Dict[str, Any]) -> torch.nn.Module:

    from adaptive.models.Deconv.Uformer_with_winpad_chunk_arch import Uformer_with_winpad_chunk
    from adaptive.models.Deconv.Uformer_with_squarepad_arch import Uformer_with_squarepad

    deconv_type = config.get("type", "Deconv_with_winpad_chunk")
    type_key = str(deconv_type).lower()
    if type_key in {"deconv_with_squarepad"}:
        Net = Uformer_with_squarepad
    else:
        Net = Uformer_with_winpad_chunk

    net = Net(
        img_size=config.get("img_size", 256),
        in_chans=config.get("in_chans", 3),
        dd_in=config.get("dd_in", 3),
        embed_dim=config.get("embed_dim", 28),
        depths=config.get("depths", [1, 2, 2, 2, 2, 2, 2, 2, 1]),
        num_heads=config.get("num_heads", [1, 2, 4, 8, 16, 16, 8, 4, 2]),
        win_size=config.get("win_size", 8),
        mlp_ratio=config.get("mlp_ratio", 1.65),
        qkv_bias=config.get("qkv_bias", True),
        qk_scale=config.get("qk_scale", None),
        drop_rate=config.get("drop_rate", 0.0),
        attn_drop_rate=config.get("attn_drop_rate", 0.0),
        drop_path_rate=config.get("drop_path_rate", 0.1),
        token_projection=config.get("token_projection", "linear"),
        token_mlp=config.get("token_mlp", "leff"),
        modulator=config.get("modulator", True),
        scale=config.get("scale", 1),
        use_checkpoint=config.get("use_checkpoint", True),
        attn_chunk=config.get("attn_chunk", 16),
        debug_mem=config.get("debug_mem", False),
    )
    return net
