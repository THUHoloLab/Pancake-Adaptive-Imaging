import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Uformer_arch import Uformer, window_partition, window_reverse


def _miB(val: int) -> float:
    return float(val) / (1024.0 ** 2)


def _print_mem_if(self, tag: str):
    if not getattr(self, "debug_mem", False):
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    alloc = _miB(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0.0
    reserved = _miB(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0.0
    peak = _miB(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
    cid = getattr(self, "debug_id", "blk")
    print(f"[mem][{cid}] {tag}: alloc={alloc:.0f}MiB reserved={reserved:.0f}MiB peak={peak:.0f}MiB")


def _lewin_chunked_forward(self, x, mask=None):

    B, L, C = x.shape
    H = int(math.sqrt(L))
    W = int(math.sqrt(L))

    attn_mask = None
    if mask is not None:
        input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
        input_mask_windows = window_partition(input_mask, self.win_size)
        attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)
        attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    if self.shift_size > 0:
        shift_mask = torch.zeros((1, H, W, 1), dtype=x.dtype, device=x.device)
        h_slices = (slice(0, -self.win_size), slice(-self.win_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.win_size), slice(-self.win_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                shift_mask[:, h, w, :] = cnt
                cnt += 1
        shift_mask_windows = window_partition(shift_mask, self.win_size)
        shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)
        shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2)
        shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
        attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

    shortcut = x
    x = self.norm1(x)
    x = x.view(B, H, W, C)

    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x

    x_windows = window_partition(shifted_x, self.win_size)
    x_windows = x_windows.view(-1, self.win_size * self.win_size, C)

    if self.modulator is not None:
        wmsa_in = self.with_pos_embed(x_windows, self.modulator.weight)
    else:
        wmsa_in = x_windows

    BnW = wmsa_in.shape[0]
    chunk = getattr(self, "attn_chunk", 0) or 0
    _print_mem_if(self, f"before_attn BnW={BnW} chunk={chunk}")
    if chunk > 0 and BnW > chunk:
        outs = []
        for s in range(0, BnW, chunk):
            e = min(s + chunk, BnW)
            w_in = wmsa_in[s:e]
            mask_chunk = attn_mask[s:e] if attn_mask is not None else None
            _print_mem_if(self, f"attn_chunk [{s}:{e}]")
            outs.append(self.attn(w_in, mask=mask_chunk))
        attn_windows = torch.cat(outs, dim=0)
    else:
        _print_mem_if(self, "attn_full")
        attn_windows = self.attn(wmsa_in, mask=attn_mask)

    attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
    shifted_x = window_reverse(attn_windows, self.win_size, H, W)
    _print_mem_if(self, "after_merge")

    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)

    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    _print_mem_if(self, "after_ffn")
    return x


def _patch_lewin_blocks(module: nn.Module, attn_chunk: int, debug_mem: bool = False):
    idx = 0
    for m in module.modules():
        if m.__class__.__name__ == "LeWinTransformerBlock":
            m.attn_chunk = int(attn_chunk)
            m.debug_mem = bool(debug_mem)
            m.debug_id = f"lewin_{idx}"
            m.forward = types.MethodType(_lewin_chunked_forward, m)
            idx += 1


class Uformer_with_winpad_chunk(nn.Module):

    def __init__(self, attn_chunk: int = 256, debug_mem: bool = False, **kwargs):
        super().__init__()
        self.inner = Uformer(**kwargs)
        _patch_lewin_blocks(self.inner, attn_chunk, debug_mem)

    def forward(self, inp_img: torch.Tensor):
        b, c, h, w = inp_img.shape
        try:
            win = int(getattr(self.inner, "win_size", 8))
            enc_layers = int(getattr(self.inner, "num_enc_layers", 4))
        except Exception:
            win, enc_layers = 8, 4
        factor = win * (2 ** max(enc_layers, 0))
        side = int(math.ceil(max(h, w) / float(factor)) * factor)

        if h == side and w == side:
            out = self.inner(inp_img)
        else:
            pad_img = inp_img.new_zeros((b, c, side, side))
            top = (side - h) // 2
            left = (side - w) // 2
            pad_img[:, :, top : top + h, left : left + w] = inp_img
            out = self.inner(pad_img)
            scale = getattr(self.inner, "scale", 1)
            oh, ow = h * scale, w * scale
            otop = top * scale
            oleft = left * scale
            out = out[:, :, otop : otop + oh, oleft : oleft + ow]

        return out
