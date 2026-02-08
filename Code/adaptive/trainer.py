from pathlib import Path
from typing import Tuple, Optional
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from adaptive.config import PipelineConfig
from adaptive.AMP_wrapper import TrainablePathSD2
from adaptive.deconv_builder import build_deconv
from adaptive.metrics import load_lpips, load_niqe, load_musiq_model, maybe_musiq, compute_psnr_ssim


class DualStageTrainer:

    def __init__(self, cfg: PipelineConfig, val_loader: Optional[DataLoader] = None, metrics_cfg: Optional[dict] = None):
        self.cfg = cfg
        self.device_deconv = torch.device(cfg.system.device_deconv)
        self.device_amp = torch.device(cfg.system.device_amp)
        self.metrics_cfg = metrics_cfg or {}
        root = Path(__file__).resolve().parents[1]
        base_model_path = (root / cfg.amp.base_model_path).resolve()
        weight_path = (root / cfg.amp.weight_path).resolve()
        self.val_loader = val_loader
        self.log_file = Path(cfg.train.output_dir) / "log.txt"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.deconv = build_deconv(vars(cfg.deconv)).to(self.device_deconv)
        if cfg.train.pretrain_path and Path(cfg.train.pretrain_path).exists():
            state_raw = torch.load(cfg.train.pretrain_path, map_location=self.device_deconv)

            if isinstance(state_raw, dict):
                for key in ["state_dict", "model", "params", "params_ema", "net_g"]:
                    if key in state_raw:
                        state_raw = state_raw[key]
                        break
            if isinstance(state_raw, dict) and "model" in state_raw:
                state_raw = state_raw["model"]

            if not cfg.train.strict_load:
                filtered = {}
                own_state = self.deconv.state_dict()
                for k, v in state_raw.items():
                    if k in own_state and own_state[k].shape == v.shape:
                        filtered[k] = v
                state_raw = filtered
            missing, unexpected = self.deconv.load_state_dict(state_raw, strict=cfg.train.strict_load)
            self._log(
                f"Loaded Deconv weights from {cfg.train.pretrain_path}, "
                f"missing={list(missing)}, unexpected={list(unexpected)}"
            )
        self.amp = TrainablePathSD2(
            base_model_path=str(base_model_path),
            weight_path=str(weight_path),
            lora_modules=cfg.amp.lora_modules,
            lora_rank=cfg.amp.lora_rank,
            model_t=cfg.amp.model_t,
            coeff_t=cfg.amp.coeff_t,
            device=cfg.amp.text_device,
            generator_device=cfg.amp.generator_device or cfg.amp.text_device,
            vae_device=cfg.amp.vae_device or cfg.amp.text_device,
        )
        self.amp.init_models()
        self.amp.freeze()

        self.optimizer = torch.optim.Adam(
            self.deconv.parameters(),
            lr=cfg.train.lr,
            betas=(cfg.train.beta1, cfg.train.beta2),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.system.mixed_precision)

        self.global_step = 0
        self.lpips_model = None
        self.lpips_device = self.device_deconv
        self.niqe_metric = None
        self.musiq_model = None
        self.musiq_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if cfg.train.checkpoint_path and Path(cfg.train.checkpoint_path).exists():
            ckpt = torch.load(cfg.train.checkpoint_path, map_location=self.device_deconv)
            deconv_state = ckpt.get("deconv")
            optimizer_state = ckpt.get("optimizer")
            self.global_step = ckpt.get("step", 0)
            if deconv_state:
                missing, unexpected = self.deconv.load_state_dict(deconv_state, strict=False)
                self._log(
                    f"Resumed Deconv from checkpoint {cfg.train.checkpoint_path}, "
                    f"missing={list(missing)}, unexpected={list(unexpected)}"
                )
            if optimizer_state:
                self.optimizer.load_state_dict(optimizer_state)
            self._log(f"Loaded optimizer state and step={self.global_step} from checkpoint {cfg.train.checkpoint_path}")

    def _forward_stage(self, lq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast("cuda", enabled=self.cfg.system.mixed_precision):
            mid = self.deconv(lq)
        refined = self.amp(
            mid.to(self.device_amp),
            prompt="",
            scale_by="factor",
            upscale=self.cfg.deconv.scale,
        )
        refined = refined.to(self.device_deconv)
        return mid, refined

    def train_epoch(self, loader: DataLoader, epoch: int, max_iters: int | None = None):
        if max_iters is not None and self.global_step >= max_iters:
            return
        self.deconv.train()
        end = time.time()
        for batch_idx, (lq, gt) in enumerate(loader):
            if max_iters is not None and self.global_step >= max_iters:
                break
            data_time = time.time() - end
            lq = lq.to(self.device_deconv, non_blocking=True)
            gt = gt.to(self.device_deconv, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.cfg.system.mixed_precision):
                mid, refined = self._forward_stage(lq)
                loss_mid = F.l1_loss(mid, gt)
                loss_refined = F.l1_loss(refined, gt)
                loss = self.cfg.train.lambda_mid * loss_mid + self.cfg.train.lambda_amp * loss_refined
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
                    f"l_pix: {loss.item():.4e} mid: {loss_mid.item():.4e} refined: {loss_refined.item():.4e}"
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
            "deconv": self.deconv.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
        }
        torch.save(state, path)

        weights_only = path.with_name(f"deconv_step_{self.global_step}.pth")
        torch.save(self.deconv.state_dict(), weights_only)
        self._log(f"Checkpoint saved to {path}, weights-only saved to {weights_only}")

    @torch.no_grad()
    def validate(self):
        start_time = time.time()
        self.deconv.eval()
        amp_state = self.amp.training
        self.amp.eval()
        psnr_mid_list = []
        ssim_mid_list = []
        lpips_mid_list = []
        niqe_mid_list = []
        musiq_mid_list = []
        psnr_ref_list = []
        ssim_ref_list = []
        lpips_ref_list = []
        niqe_ref_list = []
        musiq_ref_list = []
        out_dir = Path(self.cfg.train.output_dir) / "val_images"
        metrics_file = out_dir / f"metrics_step_{self.global_step}.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        max_samples = self.cfg.train.val_max_samples


        if self.lpips_model is None:
            self.lpips_model = load_lpips(str(self.device_deconv))
            self.lpips_device = self.device_deconv if self.lpips_model is None else next(self.lpips_model.parameters()).device
        if self.niqe_metric is None:
            self.niqe_metric = load_niqe("cpu")
        if self.musiq_model is None:
            self.musiq_model = load_musiq_model(self.metrics_cfg, device=str(self.musiq_device))

        for idx, (lq, gt) in enumerate(self.val_loader):
            lq = lq.to(self.device_deconv)
            gt = gt.to(self.device_deconv)
            mid, refined = self._forward_stage(lq)
            refined_safe = refined.detach().clamp(0, 1)
            mid_safe = mid.detach().clamp(0, 1)
            gt_safe = gt.detach().clamp(0, 1)
            refined_np = refined_safe.cpu().numpy().transpose(0, 2, 3, 1)
            mid_np = mid_safe.cpu().numpy().transpose(0, 2, 3, 1)
            for b in range(refined_np.shape[0]):
                pred_mid = mid_safe[b:b+1]
                pred_ref = refined_safe[b:b+1]
                gt_b = gt_safe[b:b+1]
                psnr_mid, ssim_mid = compute_psnr_ssim(pred_mid, gt_b)
                psnr_ref, ssim_ref = compute_psnr_ssim(pred_ref, gt_b)
                psnr_mid_list.append(psnr_mid)
                ssim_mid_list.append(ssim_mid)
                psnr_ref_list.append(psnr_ref)
                ssim_ref_list.append(ssim_ref)
                lp_mid = None
                lp_ref = None
                if self.lpips_model is not None:
                    lp_mid = float(self.lpips_model(pred_mid.to(self.lpips_device), gt_b.to(self.lpips_device)).item())
                    lp_ref = float(self.lpips_model(pred_ref.to(self.lpips_device), gt_b.to(self.lpips_device)).item())
                    lpips_mid_list.append(lp_mid)
                    lpips_ref_list.append(lp_ref)
                niqe_mid = None
                niqe_ref = None
                if self.niqe_metric is not None:
                    try:
                        niqe_mid = float(self.niqe_metric(pred_mid))
                        niqe_ref = float(self.niqe_metric(pred_ref))
                        niqe_mid_list.append(niqe_mid)
                        niqe_ref_list.append(niqe_ref)
                    except Exception as e:
                        self._log(f"[warn] NIQE failed in val: {e}")
                mu_mid = None
                mu_ref = None
                if self.musiq_model is not None:
                    mu_mid = maybe_musiq(self.musiq_model, pred_mid.to(self.musiq_device))
                    mu_ref = maybe_musiq(self.musiq_model, pred_ref.to(self.musiq_device))
                    if mu_mid is not None:
                        musiq_mid_list.append(mu_mid)
                    if mu_ref is not None:
                        musiq_ref_list.append(mu_ref)

                try:
                    with open(metrics_file, "a", encoding="utf-8") as mf:
                        mf.write(
                            f"step={self.global_step} sample={idx}_{b} "
                            f"mid_psnr={psnr_mid} mid_ssim={ssim_mid} mid_lpips={lp_mid} mid_niqe={niqe_mid} mid_musiq={mu_mid} "
                            f"ref_psnr={psnr_ref} ref_ssim={ssim_ref} ref_lpips={lp_ref} ref_niqe={niqe_ref} ref_musiq={mu_ref}\n"
                        )
                except Exception:
                    pass
                self._log(
                    f"[Val @ step {self.global_step}] sample {idx}_{b}: "
                    f"mid(psnr={psnr_mid:.3f}, ssim={ssim_mid:.4f}, lpips={lp_mid}, niqe={niqe_mid}, musiq={mu_mid}) "
                    f"ref(psnr={psnr_ref:.3f}, ssim={ssim_ref:.4f}, lpips={lp_ref}, niqe={niqe_ref}, musiq={mu_ref})"
                )
                if self.cfg.train.save_val_img:
                    mid_img = (mid_np[b] * 255.0).clip(0, 255).astype(np.uint8)
                    ref_img = (refined_np[b] * 255.0).clip(0, 255).astype(np.uint8)
                    mid_name = out_dir / f"step{self.global_step}_sample{idx}_{b}_mid.png"
                    ref_name = out_dir / f"step{self.global_step}_sample{idx}_{b}_refined.png"
                    Image.fromarray(mid_img).save(mid_name)
                    Image.fromarray(ref_img).save(ref_name)
                    self._log(f"[Val @ step {self.global_step}] saved {mid_name} and {ref_name}")

            if max_samples is not None and len(psnr_ref_list) >= max_samples:
                break

        if psnr_ref_list:
            psnr_mid_mean = float(np.mean(psnr_mid_list)) if psnr_mid_list else None
            ssim_mid_mean = float(np.mean(ssim_mid_list)) if ssim_mid_list else None
            lp_mid_mean = float(np.mean(lpips_mid_list)) if lpips_mid_list else None
            niqe_mid_mean = float(np.mean(niqe_mid_list)) if niqe_mid_list else None
            musiq_mid_mean = float(np.mean(musiq_mid_list)) if musiq_mid_list else None
            psnr_ref_mean = float(np.mean(psnr_ref_list))
            ssim_ref_mean = float(np.mean(ssim_ref_list))
            lp_ref_mean = float(np.mean(lpips_ref_list)) if lpips_ref_list else None
            niqe_ref_mean = float(np.mean(niqe_ref_list)) if niqe_ref_list else None
            musiq_ref_mean = float(np.mean(musiq_ref_list)) if musiq_ref_list else None
            elapsed = time.time() - start_time
            self._log(
                f"[Val @ step {self.global_step}] samples={len(psnr_ref_list)}, "
                f"MID -> PSNR: {psnr_mid_mean}, SSIM: {ssim_mid_mean}, LPIPS: {lp_mid_mean}, NIQE: {niqe_mid_mean}, MUSIQ: {musiq_mid_mean}; "
                f"REFINED -> PSNR: {psnr_ref_mean}, SSIM: {ssim_ref_mean}, LPIPS: {lp_ref_mean}, NIQE: {niqe_ref_mean}, MUSIQ: {musiq_ref_mean}, "
                f"time: {elapsed:.1f}s"
            )

            try:
                with open(metrics_file, "a", encoding="utf-8") as mf:
                    mf.write(
                        f"AVERAGE step={self.global_step} "
                        f"mid_psnr={psnr_mid_mean} mid_ssim={ssim_mid_mean} mid_lpips={lp_mid_mean} mid_niqe={niqe_mid_mean} mid_musiq={musiq_mid_mean} "
                        f"ref_psnr={psnr_ref_mean} ref_ssim={ssim_ref_mean} ref_lpips={lp_ref_mean} ref_niqe={niqe_ref_mean} ref_musiq={musiq_ref_mean}\n"
                    )
            except Exception:
                pass

        self.deconv.train()
        if amp_state:
            self.amp.train()

    def _log(self, msg: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} INFO: {msg}"
        print(line)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
