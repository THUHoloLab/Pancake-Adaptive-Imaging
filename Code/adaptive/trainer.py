from pathlib import Path
from typing import Tuple, Optional
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from adaptive.config import PipelineConfig
from adaptive.wrapper import TrainablePathSD2
from adaptive.builder import build_uformer
from adaptive.metrics import load_lpips, load_niqe, load_musiq_model, maybe_musiq, compute_psnr_ssim


class DualStageTrainer:

    def __init__(self, cfg: PipelineConfig, val_loader: Optional[DataLoader] = None, metrics_cfg: Optional[dict] = None):
        self.cfg = cfg
        self.device_uformer = torch.device(cfg.system.device_uformer)
        self.device_hypir = torch.device(cfg.system.device_hypir)
        self.metrics_cfg = metrics_cfg or {}
        root = Path(__file__).resolve().parents[1]
        base_model_path = (root / cfg.hypir.base_model_path).resolve()
        weight_path = (root / cfg.hypir.weight_path).resolve()
        self.val_loader = val_loader
        self.log_file = Path(cfg.train.output_dir) / "log.txt"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.uformer = build_uformer(vars(cfg.uformer)).to(self.device_uformer)
        if cfg.train.pretrain_path and Path(cfg.train.pretrain_path).exists():
            state_raw = torch.load(cfg.train.pretrain_path, map_location=self.device_uformer)

            if isinstance(state_raw, dict):
                for key in ["state_dict", "model", "params", "params_ema", "net_g"]:
                    if key in state_raw:
                        state_raw = state_raw[key]
                        break
            if isinstance(state_raw, dict) and "model" in state_raw:
                state_raw = state_raw["model"]

            if not cfg.train.strict_load:
                filtered = {}
                own_state = self.uformer.state_dict()
                for k, v in state_raw.items():
                    if k in own_state and own_state[k].shape == v.shape:
                        filtered[k] = v
                state_raw = filtered
            missing, unexpected = self.uformer.load_state_dict(state_raw, strict=cfg.train.strict_load)
            self._log(
                f"Loaded Uformer weights from {cfg.train.pretrain_path}, "
                f"missing={list(missing)}, unexpected={list(unexpected)}"
            )
        self.hypir = TrainablePathSD2(
            base_model_path=str(base_model_path),
            weight_path=str(weight_path),
            lora_modules=cfg.hypir.lora_modules,
            lora_rank=cfg.hypir.lora_rank,
            model_t=cfg.hypir.model_t,
            coeff_t=cfg.hypir.coeff_t,
            device=cfg.hypir.text_device,
            generator_device=cfg.hypir.generator_device or cfg.hypir.text_device,
            vae_device=cfg.hypir.vae_device or cfg.hypir.text_device,
        )
        self.hypir.init_models()
        self.hypir.freeze()

        self.optimizer = torch.optim.Adam(
            self.uformer.parameters(),
            lr=cfg.train.lr,
            betas=(cfg.train.beta1, cfg.train.beta2),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.system.mixed_precision)

        self.global_step = 0
        self.lpips_model = None
        self.lpips_device = self.device_uformer
        self.niqe_metric = None
        self.musiq_model = None
        self.musiq_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if cfg.train.checkpoint_path and Path(cfg.train.checkpoint_path).exists():
            ckpt = torch.load(cfg.train.checkpoint_path, map_location=self.device_uformer)
            uformer_state = ckpt.get("uformer")
            optimizer_state = ckpt.get("optimizer")
            self.global_step = ckpt.get("step", 0)
            if uformer_state:
                missing, unexpected = self.uformer.load_state_dict(uformer_state, strict=False)
                self._log(
                    f"Resumed Uformer from checkpoint {cfg.train.checkpoint_path}, "
                    f"missing={list(missing)}, unexpected={list(unexpected)}"
                )
            if optimizer_state:
                self.optimizer.load_state_dict(optimizer_state)
            self._log(f"Loaded optimizer state and step={self.global_step} from checkpoint {cfg.train.checkpoint_path}")

    def _forward_stage(self, lq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast("cuda", enabled=self.cfg.system.mixed_precision):
            mid = self.uformer(lq)
        prior = self.hypir(
            mid.to(self.device_hypir),
            prompt="",
            scale_by="factor",
            upscale=self.cfg.uformer.scale,
        )
        prior = prior.to(self.device_uformer)
        return mid, prior

    def train_epoch(self, loader: DataLoader, epoch: int, max_iters: int | None = None):
        if max_iters is not None and self.global_step >= max_iters:
            return
        self.uformer.train()
        end = time.time()
        for batch_idx, (lq, gt) in enumerate(loader):
            if max_iters is not None and self.global_step >= max_iters:
                break
            data_time = time.time() - end
            lq = lq.to(self.device_uformer, non_blocking=True)
            gt = gt.to(self.device_uformer, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.cfg.system.mixed_precision):
                mid, prior = self._forward_stage(lq)
                loss_mid = F.l1_loss(mid, gt)
                loss_prior = F.l1_loss(prior, gt)
                loss = self.cfg.train.lambda_mid * loss_mid + self.cfg.train.lambda_hypir * loss_prior
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step += 1
            if self.global_step % self.cfg.train.log_every == 0:
                iter_time = time.time() - end
                lr = self.optimizer.param_groups[0]["lr"]
                self._log(
                    f"[{self.cfg.run_name}][epoch:{epoch} iter:{self.global_step}, lr:({lr:.3e})] "
                    f"time(data): {iter_time:.3f} ({data_time:.3f}) "
                    f"l_pix: {loss.item():.4e} mid: {loss_mid.item():.4e} prior: {loss_prior.item():.4e}"
                )

            if self.global_step % self.cfg.train.ckpt_every == 0:
                self.save_checkpoint(Path(self.cfg.train.output_dir) / f"step_{self.global_step}.pt")

            if self.val_loader and self.cfg.train.val_freq > 0 and self.global_step % self.cfg.train.val_freq == 0:
                self.validate()
            end = time.time()
            if max_iters is not None and self.global_step >= max_iters:
                break

    def save_checkpoint(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "uformer": self.uformer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
        }
        torch.save(state, path)

        weights_only = path.with_name(f"uformer_step_{self.global_step}.pth")
        torch.save(self.uformer.state_dict(), weights_only)
        self._log(f"Checkpoint saved to {path}, weights-only saved to {weights_only}")

    @torch.no_grad()
    def validate(self):
        start_time = time.time()
        self.uformer.eval()
        hypir_state = self.hypir.training
        self.hypir.eval()
        psnr_mid_list = []
        ssim_mid_list = []
        lpips_mid_list = []
        niqe_mid_list = []
        musiq_mid_list = []
        psnr_prior_list = []
        ssim_prior_list = []
        lpips_prior_list = []
        niqe_prior_list = []
        musiq_prior_list = []
        out_dir = Path(self.cfg.train.output_dir) / "val_images"
        metrics_file = out_dir / f"metrics_step_{self.global_step}.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        max_samples = self.cfg.train.val_max_samples


        if self.lpips_model is None:
            self.lpips_model = load_lpips(str(self.device_uformer))
            self.lpips_device = self.device_uformer if self.lpips_model is None else next(self.lpips_model.parameters()).device
        if self.niqe_metric is None:
            self.niqe_metric = load_niqe("cpu")
        if self.musiq_model is None:
            self.musiq_model = load_musiq_model(self.metrics_cfg, device=str(self.musiq_device))

        for idx, (lq, gt) in enumerate(self.val_loader):
            lq = lq.to(self.device_uformer)
            gt = gt.to(self.device_uformer)
            mid, prior = self._forward_stage(lq)
            prior_safe = prior.detach().clamp(0, 1)
            mid_safe = mid.detach().clamp(0, 1)
            gt_safe = gt.detach().clamp(0, 1)
            prior_np = prior_safe.cpu().numpy().transpose(0, 2, 3, 1)
            mid_np = mid_safe.cpu().numpy().transpose(0, 2, 3, 1)
            for b in range(prior_np.shape[0]):
                pred_mid = mid_safe[b:b+1]
                pred_prior = prior_safe[b:b+1]
                gt_b = gt_safe[b:b+1]
                psnr_mid, ssim_mid = compute_psnr_ssim(pred_mid, gt_b)
                psnr_prior, ssim_prior = compute_psnr_ssim(pred_prior, gt_b)
                psnr_mid_list.append(psnr_mid)
                ssim_mid_list.append(ssim_mid)
                psnr_prior_list.append(psnr_prior)
                ssim_prior_list.append(ssim_prior)
                lp_mid = None
                lp_prior = None
                if self.lpips_model is not None:
                    lp_mid = float(self.lpips_model(pred_mid.to(self.lpips_device), gt_b.to(self.lpips_device)).item())
                    lp_prior = float(self.lpips_model(pred_prior.to(self.lpips_device), gt_b.to(self.lpips_device)).item())
                    lpips_mid_list.append(lp_mid)
                    lpips_prior_list.append(lp_prior)
                niqe_mid = None
                niqe_prior = None
                if self.niqe_metric is not None:
                    try:
                        niqe_mid = float(self.niqe_metric(pred_mid))
                        niqe_prior = float(self.niqe_metric(pred_prior))
                        niqe_mid_list.append(niqe_mid)
                        niqe_prior_list.append(niqe_prior)
                    except Exception as e:
                        self._log(f"[warn] NIQE failed in val: {e}")
                mu_mid = None
                mu_prior = None
                if self.musiq_model is not None:
                    mu_mid = maybe_musiq(self.musiq_model, pred_mid.to(self.musiq_device))
                    mu_prior = maybe_musiq(self.musiq_model, pred_prior.to(self.musiq_device))
                    if mu_mid is not None:
                        musiq_mid_list.append(mu_mid)
                    if mu_prior is not None:
                        musiq_prior_list.append(mu_prior)

                try:
                    with open(metrics_file, "a", encoding="utf-8") as mf:
                        mf.write(
                            f"step={self.global_step} sample={idx}_{b} "
                            f"mid_psnr={psnr_mid} mid_ssim={ssim_mid} mid_lpips={lp_mid} mid_niqe={niqe_mid} mid_musiq={mu_mid} "
                            f"prior_psnr={psnr_prior} prior_ssim={ssim_prior} prior_lpips={lp_prior} prior_niqe={niqe_prior} prior_musiq={mu_prior}\n"
                        )
                except Exception:
                    pass
                self._log(
                    f"[Val @ step {self.global_step}] sample {idx}_{b}: "
                    f"mid(psnr={psnr_mid:.3f}, ssim={ssim_mid:.4f}, lpips={lp_mid}, niqe={niqe_mid}, musiq={mu_mid}) "
                    f"prior(psnr={psnr_prior:.3f}, ssim={ssim_prior:.4f}, lpips={lp_prior}, niqe={niqe_prior}, musiq={mu_prior})"
                )
                if self.cfg.train.save_val_img:
                    mid_img = (mid_np[b] * 255.0).clip(0, 255).astype(np.uint8)
                    prior_img = (prior_np[b] * 255.0).clip(0, 255).astype(np.uint8)
                    mid_name = out_dir / f"step{self.global_step}_sample{idx}_{b}_mid.png"
                    prior_name = out_dir / f"step{self.global_step}_sample{idx}_{b}_prior.png"
                    Image.fromarray(mid_img).save(mid_name)
                    Image.fromarray(prior_img).save(prior_name)
                    self._log(f"[Val @ step {self.global_step}] saved {mid_name} and {prior_name}")

            if max_samples is not None and len(psnr_prior_list) >= max_samples:
                break

        if psnr_prior_list:
            psnr_mid_mean = float(np.mean(psnr_mid_list)) if psnr_mid_list else None
            ssim_mid_mean = float(np.mean(ssim_mid_list)) if ssim_mid_list else None
            lp_mid_mean = float(np.mean(lpips_mid_list)) if lpips_mid_list else None
            niqe_mid_mean = float(np.mean(niqe_mid_list)) if niqe_mid_list else None
            musiq_mid_mean = float(np.mean(musiq_mid_list)) if musiq_mid_list else None
            psnr_prior_mean = float(np.mean(psnr_prior_list))
            ssim_prior_mean = float(np.mean(ssim_prior_list))
            lp_prior_mean = float(np.mean(lpips_prior_list)) if lpips_prior_list else None
            niqe_prior_mean = float(np.mean(niqe_prior_list)) if niqe_prior_list else None
            musiq_prior_mean = float(np.mean(musiq_prior_list)) if musiq_prior_list else None
            elapsed = time.time() - start_time
            self._log(
                f"[Val @ step {self.global_step}] samples={len(psnr_prior_list)}, "
                f"MID -> PSNR: {psnr_mid_mean}, SSIM: {ssim_mid_mean}, LPIPS: {lp_mid_mean}, NIQE: {niqe_mid_mean}, MUSIQ: {musiq_mid_mean}; "
                f"PRIOR -> PSNR: {psnr_prior_mean}, SSIM: {ssim_prior_mean}, LPIPS: {lp_prior_mean}, NIQE: {niqe_prior_mean}, MUSIQ: {musiq_prior_mean}, "
                f"time: {elapsed:.1f}s"
            )

            try:
                with open(metrics_file, "a", encoding="utf-8") as mf:
                    mf.write(
                        f"AVERAGE step={self.global_step} "
                        f"mid_psnr={psnr_mid_mean} mid_ssim={ssim_mid_mean} mid_lpips={lp_mid_mean} mid_niqe={niqe_mid_mean} mid_musiq={musiq_mid_mean} "
                        f"prior_psnr={psnr_prior_mean} prior_ssim={ssim_prior_mean} prior_lpips={lp_prior_mean} prior_niqe={niqe_prior_mean} prior_musiq={musiq_prior_mean}\n"
                    )
            except Exception:
                pass

        self.uformer.train()
        if hypir_state:
            self.hypir.train()

    def _log(self, msg: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} INFO: {msg}"
        print(line)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
