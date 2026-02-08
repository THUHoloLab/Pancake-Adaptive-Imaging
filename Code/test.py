import argparse
from pathlib import Path
import yaml

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import importlib

from adaptive import *
from adaptive.uformer_builder import build_uformer
from adaptive.AMP_wrapper import TrainablePathSD2
from adaptive.metrics import (
    load_lpips as metrics_load_lpips,
    load_niqe as metrics_load_niqe,
    load_musiq_model as metrics_load_musiq_model,
    load_nima,
    load_clipiqa,
    maybe_musiq,
    compute_psnr_ssim,
)


def load_image_paths(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]


def maybe_load_lpips(device):
    try:
        import lpips

        return lpips.LPIPS(net="vgg").to(device)
    except Exception as e:
        print(f"[warn] LPIPS not available: {e}")
        return None


def maybe_load_niqe(device):
    try:
        pyiqa = importlib.import_module("pyiqa")
        metric = pyiqa.create_metric("niqe", device=device)
        print("[info] NIQE backend: pyiqa")
        return metric
    except Exception as e:
        try:
            from skimage.metrics import niqe as skimage_niqe

            def _fn(img):
                arr = img.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
                if arr.ndim == 3 and arr.shape[2] == 3:
                    arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
                return skimage_niqe(arr)

            print("[info] NIQE backend: skimage.niqe")
            return _fn
        except Exception as e2:
            try:
                import sys
                import numpy as np
                import cv2
                from scipy.special import gamma
                from pathlib import Path
                from scipy.ndimage import convolve

                def find_pris_params():
                    for p in sys.path:
                        try:
                            cand = next(Path(p).rglob("niqe_pris_params.npz"))
                            return cand
                        except StopIteration:
                            continue
                    return None

                params_path = find_pris_params()
                if params_path is None:
                    raise RuntimeError("niqe_pris_params.npz not found in sys.path")
                pris = np.load(params_path)
                mu_pris_param = pris["mu_pris_param"]
                cov_pris_param = pris["cov_pris_param"]
                gaussian_window = pris["gaussian_window"]

                def estimate_aggd_param(block):
                    block = block.flatten()
                    gam = np.arange(0.2, 10.001, 0.001)
                    r_gam = (gamma(2 / gam) ** 2) / (gamma(1 / gam) * gamma(3 / gam))
                    left_std = np.sqrt(np.mean(block[block < 0]**2))
                    right_std = np.sqrt(np.mean(block[block > 0]**2))
                    gammahat = left_std / right_std
                    rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
                    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1)**2)
                    array_position = np.argmin((r_gam - rhatnorm) ** 2)
                    alpha = gam[array_position]
                    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
                    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
                    return alpha, beta_l, beta_r

                def compute_feature(block):
                    feat = []
                    alpha, beta_l, beta_r = estimate_aggd_param(block)
                    feat.extend([alpha, (beta_l + beta_r) / 2])
                    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
                    for shift in shifts:
                        shifted = np.roll(block, shift, axis=(0, 1))
                        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted)
                        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
                        feat.extend([alpha, mean, beta_l, beta_r])
                    return feat

                def niqe_core(img, block_h=96, block_w=96):
                    h, w = img.shape
                    num_block_h = int(np.floor(h / block_h))
                    num_block_w = int(np.floor(w / block_w))
                    img = img[0 : num_block_h * block_h, 0 : num_block_w * block_w]
                    distparam = []
                    for scale in (1, 2):
                        mu = convolve(img, gaussian_window, mode="nearest")
                        sigma = np.sqrt(np.abs(convolve(np.square(img), gaussian_window, mode="nearest") - np.square(mu)))
                        img_norm = (img - mu) / (sigma + 1)
                        feats = []
                        for idx_w in range(num_block_w):
                            for idx_h in range(num_block_h):
                                block = img_norm[
                                    idx_h * block_h // scale : (idx_h + 1) * block_h // scale,
                                    idx_w * block_w // scale : (idx_w + 1) * block_w // scale,
                                ]
                                feats.append(compute_feature(block))
                        distparam.append(np.array(feats))
                        if scale == 1:
                            img = cv2.resize(img / 255.0, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR) * 255.0
                    distparam = np.concatenate(distparam, axis=1)
                    mu_distparam = np.nanmean(distparam, axis=0)
                    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
                    cov_distparam = np.cov(distparam_no_nan, rowvar=False)
                    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
                    quality = np.matmul(np.matmul((mu_pris_param - mu_distparam), invcov_param), (mu_pris_param - mu_distparam).T)
                    return np.sqrt(quality)

                def _fn(img):
                    arr = img.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
                    if arr.shape[2] == 3:
                        arr = 0.257 * arr[..., 0] + 0.504 * arr[..., 1] + 0.098 * arr[..., 2] + 16 / 255
                    arr = arr * 255.0
                    return float(niqe_core(arr))

                print(f"[info] NIQE backend: local fallback using {params_path}")
                return _fn
            except Exception as e3:
                print(f"[warn] NIQE not available: {e} | skimage fallback error: {e2} | local fallback error: {e3}")
                return None


def maybe_musiq(model, img_tensor):
    if model is None:
        return None
    try:
        with torch.no_grad():
            return float(model(img_tensor).mean().item())
    except Exception:
        return None


def load_musiq_model(meta: dict):
    name = meta.get("musiq_metric_name", "musiq-koniq")
    weight = meta.get("musiq_weight", None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    err_primary = None
    try:
        pyiqa = importlib.import_module("pyiqa")
        metric = pyiqa.create_metric(name, device=device)
        return metric
    except Exception as e1:
        err_primary = e1

    try:
        from pyiqa.archs.musiq_arch import MUSIQ

        model = MUSIQ(pretrained=True, pretrained_model_path=weight)
        model.to(device)
        model.eval()
        return model
    except Exception as e2:
        print(f"[warn] MUSIQ not available: {err_primary} | manual_load_error: {e2}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Inference: Uformer + AMP sweep over model_t with metrics")
    parser.add_argument("--config_yaml", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    devices = cfg.get("devices", {})
    device_uformer = devices.get("device_uformer", "cuda:0")
    device_AMP_text = devices.get("device_AMP_text", "cuda:1")
    device_AMP_vae = devices.get("device_AMP_vae", device_AMP_text)
    device_AMP_gen = devices.get("device_AMP_gen", device_AMP_text)

    net_cfg = cfg.get("network_g", {})
    uformer = build_uformer(net_cfg).to(device_uformer)
    uformer.eval().requires_grad_(False)
    weight_path = Path(cfg["path"]["pretrain_network_g"])
    raw_state = torch.load(weight_path, map_location=device_uformer)
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]
    if isinstance(raw_state, dict) and "params" in raw_state:
        raw_state = raw_state["params"]
    prefixed = {}
    has_inner = any(k.startswith("inner.") for k in raw_state.keys())
    if not has_inner:
        for k, v in raw_state.items():
            prefixed[f"inner.{k}"] = v
    else:
        prefixed = raw_state
    missing, unexpected = uformer.load_state_dict(prefixed, strict=False)
    print(
        f"Loaded Uformer weight from {weight_path}, "
        f"missing={len(missing)} keys, unexpected={len(unexpected)} keys"
    )

    AMP_cfg = cfg.get("AMP", {})
    base_model_path = str(AMP_cfg["base_model_path"])
    weight_AMP = str(AMP_cfg["weight_path"])
    AMP = TrainablePathSD2(
        base_model_path=base_model_path,
        weight_path=weight_AMP,
        lora_modules=AMP_cfg.get("lora_modules", []),
        lora_rank=AMP_cfg.get("lora_rank", 256),
        model_t=AMP_cfg.get("model_t", 150),
        coeff_t=AMP_cfg.get("coeff_t", 150),
        device=device_AMP_text,
        generator_device=device_AMP_gen,
        vae_device=device_AMP_vae,
    )
    AMP.init_models()
    AMP.freeze()

    input_dir = Path(cfg["data"]["input_dir"])
    gt_dir = Path(cfg["data"].get("gt_dir")) if cfg["data"].get("gt_dir") else None
    output_root = Path(cfg["data"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    paths = load_image_paths(input_dir)
    print(f"Found {len(paths)} images in {input_dir}")
    to_tensor = transforms.ToTensor()

    sweep = cfg.get("sweep")
    sweep_enabled = True
    if isinstance(sweep, dict):
        sweep_enabled = sweep.get("enabled", True)
    if sweep and sweep_enabled:
        t_start = int(sweep.get("t_start", 100))
        t_end = int(sweep.get("t_end", 200))
        t_step = int(sweep.get("t_step", 10))
        t_values = list(range(t_start, t_end + 1, t_step))
    else:
        default_t = int(cfg.get("AMP", {}).get("model_t", 150))
        t_values = [default_t]

    lpips_model = maybe_load_lpips(device_uformer) if gt_dir else None
    lpips_device = None
    if lpips_model is not None:
        lpips_device = next(lpips_model.parameters()).device
    metrics_cfg = cfg.get("metrics", {})
    musiq_model = load_musiq_model(metrics_cfg)
    nima_metric = load_nima(metrics_cfg, device=device_uformer)
    clipiqa_metric = load_clipiqa(metrics_cfg, device=device_uformer)
    musiq_device = None
    if musiq_model is not None:
        try:
            musiq_device = next(musiq_model.parameters()).device
        except Exception:
            musiq_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    niqe_metric = maybe_load_niqe(device_uformer)
    niqe_device = getattr(niqe_metric, "device", device_uformer) if hasattr(niqe_metric, "to") else "cpu"

    for t in t_values:
        AMP.model_t = t
        AMP.coeff_t = t
        out_dir = output_root / f"t_{t}"
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_txt = out_dir / "metrics.txt"
        mid_lpips_scores: list[float] = []
        mid_niqe_scores: list[float] = []
        mid_musiq_scores: list[float] = []
        mid_nima_scores: list[float] = []
        mid_psnr_scores: list[float] = []
        mid_ssim_scores: list[float] = []
        mid_clipiqa_scores: list[float] = []
        ref_lpips_scores: list[float] = []
        ref_niqe_scores: list[float] = []
        ref_musiq_scores: list[float] = []
        ref_nima_scores: list[float] = []
        ref_psnr_scores: list[float] = []
        ref_ssim_scores: list[float] = []
        ref_clipiqa_scores: list[float] = []
        with open(metrics_txt, "w", encoding="utf-8") as ftxt:
            for p in paths:
                img = Image.open(p).convert("RGB")
                lq = to_tensor(img).unsqueeze(0).to(device_uformer)
                orig_h, orig_w = lq.shape[-2:]
                pad_h = (128 - orig_h % 128) % 128
                pad_w = (128 - orig_w % 128) % 128
                if pad_h or pad_w:
                    lq = F.pad(lq, (0, pad_w, 0, pad_h), mode="reflect")
                with torch.no_grad():
                    mid = uformer(lq)
                    refined = AMP(mid, prompt="", upscale=1)
                mid_safe = mid.detach().clamp(0, 1)[..., :orig_h, :orig_w]
                refined_safe = refined.detach().clamp(0, 1)[..., :orig_h, :orig_w]
                mid_np = mid_safe.cpu().squeeze(0).permute(1, 2, 0).numpy()
                ref_np = refined_safe.cpu().squeeze(0).permute(1, 2, 0).numpy()
                Image.fromarray((mid_np * 255.0).round().clip(0, 255).astype(np.uint8)).save(
                    out_dir / f"mid_{p.name}"
                )
                Image.fromarray((ref_np * 255.0).round().clip(0, 255).astype(np.uint8)).save(out_dir / p.name)

                mid_lp = None
                ref_lp = None
                gt_tensor = None
                gt_path = gt_dir / p.name if gt_dir else None
                if gt_path and gt_path.exists():
                    gt_img = Image.open(gt_path).convert("RGB")
                    gt_tensor = to_tensor(gt_img).unsqueeze(0).to(device_uformer)
                    if lpips_model is not None:
                        gt_lp = gt_tensor.to(lpips_device)
                        ref_for_lp = refined_safe.to(lpips_device)
                        mid_for_lp = mid_safe.to(lpips_device)
                        ref_lp = float(lpips_model(ref_for_lp, gt_lp).item())
                        mid_lp = float(lpips_model(mid_for_lp, gt_lp).item())
                        ref_lpips_scores.append(ref_lp)
                        mid_lpips_scores.append(mid_lp)
                mid_niqe = None
                ref_niqe = None
                if niqe_metric is not None:
                    try:
                        if hasattr(niqe_metric, "to"):
                            mid_niqe = float(niqe_metric(mid_safe.to(niqe_device)).item())
                            ref_niqe = float(niqe_metric(refined_safe.to(niqe_device)).item())
                        else:
                            mid_niqe = float(niqe_metric(mid_safe))
                            ref_niqe = float(niqe_metric(refined_safe))
                        mid_niqe_scores.append(mid_niqe)
                        ref_niqe_scores.append(ref_niqe)
                    except Exception as e:
                        print(f"[warn] NIQE failed for {p.name}: {e}")
                mid_mus = None
                ref_mus = None
                if musiq_model is not None:
                    mid_mus = maybe_musiq(musiq_model, mid_safe.to(musiq_device))
                    ref_mus = maybe_musiq(musiq_model, refined_safe.to(musiq_device))
                if mid_mus is not None:
                    mid_musiq_scores.append(mid_mus)
                if ref_mus is not None:
                    ref_musiq_scores.append(ref_mus)
                mid_nima = None
                ref_nima = None
                if nima_metric is not None:
                    try:
                        mid_nima = float(nima_metric(mid_safe.to(device_uformer)).mean().item())
                        ref_nima = float(nima_metric(refined_safe.to(device_uformer)).mean().item())
                        mid_nima_scores.append(mid_nima)
                        ref_nima_scores.append(ref_nima)
                    except Exception:
                        pass
                mid_clip = None
                ref_clip = None
                if clipiqa_metric is not None:
                    try:
                        mid_clip = float(clipiqa_metric(mid_safe.to(device_uformer)).mean().item())
                        ref_clip = float(clipiqa_metric(refined_safe.to(device_uformer)).mean().item())
                        mid_clipiqa_scores.append(mid_clip)
                        ref_clipiqa_scores.append(ref_clip)
                    except Exception:
                        pass
                mid_psnr = None
                mid_ssim = None
                ref_psnr = None
                ref_ssim = None
                if gt_tensor is not None:
                    try:
                        mid_psnr, mid_ssim = compute_psnr_ssim(mid_safe, gt_tensor)
                        ref_psnr, ref_ssim = compute_psnr_ssim(refined_safe, gt_tensor)
                        mid_psnr_scores.append(mid_psnr)
                        mid_ssim_scores.append(mid_ssim)
                        ref_psnr_scores.append(ref_psnr)
                        ref_ssim_scores.append(ref_ssim)
                    except Exception:
                        pass
                line = (
                    f"{p.name}: "
                    f"mid_lpips={mid_lp} mid_niqe={mid_niqe} mid_musiq={mid_mus} mid_nima={mid_nima} mid_clipiqa={mid_clip} mid_psnr={mid_psnr} mid_ssim={mid_ssim} | "
                    f"ref_lpips={ref_lp} ref_niqe={ref_niqe} ref_musiq={ref_mus} ref_nima={ref_nima} ref_clipiqa={ref_clip} ref_psnr={ref_psnr} ref_ssim={ref_ssim}"
                )
                print(line)
                ftxt.write(line + "\n")

            def _avg(lst):
                return sum(lst) / len(lst) if lst else None

            ftxt.write(
                "\nAVERAGE: "
                f"mid_lpips={_avg(mid_lpips_scores)} mid_niqe={_avg(mid_niqe_scores)} mid_musiq={_avg(mid_musiq_scores)} mid_nima={_avg(mid_nima_scores)} mid_clipiqa={_avg(mid_clipiqa_scores)} "
                f"mid_psnr={_avg(mid_psnr_scores)} mid_ssim={_avg(mid_ssim_scores)} | "
                f"ref_lpips={_avg(ref_lpips_scores)} ref_niqe={_avg(ref_niqe_scores)} ref_musiq={_avg(ref_musiq_scores)} ref_nima={_avg(ref_nima_scores)} ref_clipiqa={_avg(ref_clipiqa_scores)} "
                f"ref_psnr={_avg(ref_psnr_scores)} ref_ssim={_avg(ref_ssim_scores)}\n"
            )
        print(f"Saved outputs and metrics for model_t={t} to {out_dir}")


if __name__ == "__main__":
    main()
