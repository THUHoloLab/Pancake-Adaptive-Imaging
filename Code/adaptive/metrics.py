from pathlib import Path
import importlib
import sys
import torch
import numpy as np
from scipy.ndimage import convolve
from scipy.special import gamma
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_lpips(device: str):
    try:
        import lpips

        return lpips.LPIPS(net="vgg").to(device)
    except Exception as e:
        print(f"[warn] LPIPS not available: {e}")
        return None


def load_pyiqa_metric(name: str, device: str, weight: str | None = None):
    try:
        pyiqa = importlib.import_module("pyiqa")
        kwargs = {"device": device}
        if weight:
            kwargs["weight_path"] = weight
        return pyiqa.create_metric(name, **kwargs)
    except Exception as e:
        print(f"[warn] pyiqa metric '{name}' unavailable: {e}")
        return None


def load_musiq_model(meta: dict, device: str):
    """
    Try loading MUSIQ metric via pyiqa or manual arch.
    meta keys: musiq_metric_name, musiq_weight
    """
    name = meta.get("musiq_metric_name", "musiq-koniq")
    weight = meta.get("musiq_weight", None)
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


def maybe_musiq(model, img):
    if model is None:
        return None
    try:
        with torch.no_grad():
            return float(model(img).mean().item())
    except Exception as e:
        print(f"[warn] MUSIQ failed: {e}")
        return None


def load_clipiqa(meta: dict, device: str):
    """
    Load CLIP-IQA metric via pyiqa.
    meta keys:
      - clipiqa_metric_name: default 'clipiqa'
      - clipiqa_weight: optional checkpoint path
    """
    name = meta.get("clipiqa_metric_name", "clipiqa")
    weight = meta.get("clipiqa_weight")
    metric = load_pyiqa_metric(name, device, weight)
    if metric is None:
        print("[warn] CLIP-IQA not available")
    return metric


def load_nima(meta: dict, device: str):
    """
    Load NIMA metric via pyiqa.
    meta keys:
      - nima_metric_name: default 'nima'
      - nima_weight: optional checkpoint path
    """
    name = meta.get("nima_metric_name", "nima")
    weight = meta.get("nima_weight")
    metric = load_pyiqa_metric(name, device, weight)
    if metric is None:
        print("[warn] NIMA not available")
    return metric


def load_niqe(device: str):
    """Try multiple backends to get NIQE."""
    metric = load_pyiqa_metric("niqe", device)
    if metric is not None:
        return metric
    try:
        from skimage.metrics import niqe as skimage_niqe

        def _fn(img: torch.Tensor):
            arr = img.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            return skimage_niqe(arr)

        print("[info] NIQE backend: skimage.niqe")
        return _fn
    except Exception:
        pass

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
        print("[warn] NIQE params not found")
        return None
    pris = np.load(params_path)
    mu_pris_param = pris["mu_pris_param"]
    cov_pris_param = pris["cov_pris_param"]
    gaussian_window = pris["gaussian_window"]

    def estimate_aggd_param(block):
        block = block.flatten()
        gam = np.arange(0.2, 10.001, 0.001)
        r_gam = (gamma(2 / gam) ** 2) / (gamma(1 / gam) * gamma(3 / gam))
        left_std = np.sqrt(np.mean(block[block < 0] ** 2))
        right_std = np.sqrt(np.mean(block[block > 0] ** 2))
        gammahat = left_std / right_std
        rhat = (np.mean(np.abs(block))) ** 2 / np.mean(block**2)
        rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1) ** 2)
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
                img = torch.tensor(img / 255.0).float()
                img = torch.nn.functional.interpolate(
                    img[None, None, ...], scale_factor=0.5, mode="bicubic", align_corners=False
                )[0, 0].numpy() * 255.0
        distparam = np.concatenate(distparam, axis=1)
        mu_distparam = np.nanmean(distparam, axis=0)
        distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
        cov_distparam = np.cov(distparam_no_nan, rowvar=False)
        invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
        quality = np.matmul(np.matmul((mu_pris_param - mu_distparam), invcov_param), (mu_pris_param - mu_distparam).T)
        return np.sqrt(quality)

    def _fn(img: torch.Tensor):
        arr = img.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
        if arr.shape[2] == 3:
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        arr = arr * 255.0
        return float(niqe_core(arr))

    print(f"[info] NIQE backend: local params at {params_path}")
    return _fn


def compute_psnr_ssim(pred: torch.Tensor, gt: torch.Tensor):
    pred_np = pred.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    gt_np = gt.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    psnr_val = float(peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0))
    try:
        ssim_val = float(structural_similarity(gt_np, pred_np, channel_axis=2, data_range=1.0))
    except TypeError:
        ssim_val = float(structural_similarity(gt_np, pred_np, multichannel=True, data_range=1.0))
    return psnr_val, ssim_val
