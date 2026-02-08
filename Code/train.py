import argparse
from pathlib import Path
import yaml
import warnings

from adaptive.config import DataConfig, PipelineConfig, TrainConfig, UformerConfig
from adaptive.dataset import PairedImageFolder
from adaptive.trainer import DualStageTrainer
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end Uformer + HYPIR training pipeline")
    parser.add_argument("--config_yaml", type=Path, default=None, help="Optional YAML to override defaults")
    parser.add_argument("--dataroot_gt", type=Path, default=Path(r"F:\wjw\Pancake\Datapancake1808-1017\train_ground_truth"))
    parser.add_argument("--dataroot_lq", type=Path, default=Path(r"F:\wjw\Pancake\Datapancake1808-1017\train_capture"))
    parser.add_argument("--dataroot_gt_val", type=Path, default=Path(r"F:\wjw\Pancake\Datapancake1808-1017\valid_ground_truth"))
    parser.add_argument("--dataroot_lq_val", type=Path, default=Path(r"F:\wjw\Pancake\Datapancake1808-1017\valid_capture"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=Path, default=Path("adaptive_outputs"))
    parser.add_argument("--device_uformer", type=str, default="cuda:0")
    parser.add_argument("--device_hypir_text", type=str, default="cuda:1")
    parser.add_argument("--device_hypir_vae", type=str, default=None)
    parser.add_argument("--device_hypir_gen", type=str, default=None)
    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter")
    warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
    warnings.filterwarnings("ignore", category=FutureWarning)

    args = parse_args()
    yaml_cfg = {}
    if args.config_yaml and args.config_yaml.exists():
        with open(args.config_yaml, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}

    ds_cfg = yaml_cfg.get("datasets", {})
    train_ds = ds_cfg.get("train", {})
    val_ds = ds_cfg.get("val_1", {})
    dataroot_gt = Path(train_ds.get("dataroot_gt", yaml_cfg.get("dataroot_gt", args.dataroot_gt)))
    dataroot_lq = Path(train_ds.get("dataroot_lq", yaml_cfg.get("dataroot_lq", args.dataroot_lq)))

    data_cfg = DataConfig(dataroot_gt=dataroot_gt, dataroot_lq=dataroot_lq)

    train_section = yaml_cfg.get("train", {})
    optim_g = train_section.get("optim_g", {})
    betas = optim_g.get("betas", [TrainConfig.beta1, TrainConfig.beta2])
    logger_cfg = yaml_cfg.get("logger", {})
    total_iters = train_section.get("total_iter", yaml_cfg.get("total_iter"))
    if total_iters is not None:
        total_iters = int(total_iters)
    checkpoint_path = train_section.get("checkpoint_path", yaml_cfg.get("checkpoint_path"))
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)

    train_cfg = TrainConfig(
        batch_size=train_ds.get("batch_size_per_gpu", yaml_cfg.get("batch_size", args.batch_size)),
        epochs=yaml_cfg.get("epochs", args.epochs),
        lr=float(optim_g.get("lr", yaml_cfg.get("lr", args.lr))),
        output_dir=yaml_cfg.get("output_dir", args.output_dir),
        beta1=betas[0] if len(betas) > 0 else TrainConfig.beta1,
        beta2=betas[1] if len(betas) > 1 else TrainConfig.beta2,
        lambda_mid=train_section.get("lambda_mid", yaml_cfg.get("lambda_mid", TrainConfig.lambda_mid)),
        lambda_hypir=train_section.get("lambda_hypir", yaml_cfg.get("lambda_hypir", TrainConfig.lambda_hypir)),
        log_every=int(logger_cfg.get("print_freq", yaml_cfg.get("log_every", TrainConfig.log_every))),
        ckpt_every=int(logger_cfg.get("save_checkpoint_freq", yaml_cfg.get("ckpt_every", TrainConfig.ckpt_every))),
        val_freq=int(train_section.get("val_freq", yaml_cfg.get("val_freq", TrainConfig.val_freq))),
        save_val_img=bool(train_section.get("save_val_img", yaml_cfg.get("save_val_img", TrainConfig.save_val_img))),
        val_max_samples=train_section.get("val_max_samples", yaml_cfg.get("val_max_samples", TrainConfig.val_max_samples)),
        num_workers=train_ds.get("num_worker_per_gpu", yaml_cfg.get("num_workers", TrainConfig.num_workers)),
        pretrain_path=Path(yaml_cfg.get("path", {}).get("pretrain_network_g")) if yaml_cfg.get("path", {}) else None,
        strict_load=bool(yaml_cfg.get("path", {}).get("strict_load_g", False)) if yaml_cfg.get("path", {}) else False,
        initial_val=bool(train_section.get("initial_val", yaml_cfg.get("initial_val", TrainConfig.initial_val))),
        total_iters=total_iters,
        checkpoint_path=checkpoint_path,
    )
    net_cfg = yaml_cfg.get("network_g", {})
    uformer_cfg = UformerConfig(
        img_size=net_cfg.get("img_size", 256),
        in_chans=net_cfg.get("in_chans", 3),
        dd_in=net_cfg.get("dd_in", 3),
        embed_dim=net_cfg.get("embed_dim", 28),
        depths=net_cfg.get("depths", UformerConfig().depths),
        num_heads=net_cfg.get("num_heads", UformerConfig().num_heads),
        win_size=net_cfg.get("win_size", 16),
        mlp_ratio=net_cfg.get("mlp_ratio", 1.65),
        qkv_bias=net_cfg.get("qkv_bias", True),
        qk_scale=net_cfg.get("qk_scale", None),
        drop_rate=net_cfg.get("drop_rate", 0.0),
        attn_drop_rate=net_cfg.get("attn_drop_rate", 0.0),
        drop_path_rate=net_cfg.get("drop_path_rate", 0.1),
        token_projection=net_cfg.get("token_projection", "linear"),
        token_mlp=net_cfg.get("token_mlp", "leff"),
        modulator=net_cfg.get("modulator", True),
        scale=net_cfg.get("scale", 1),
        use_checkpoint=net_cfg.get("use_checkpoint", True),
        attn_chunk=net_cfg.get("attn_chunk", 16),
        debug_mem=net_cfg.get("debug_mem", False),
    )

    cfg = PipelineConfig(
        data=data_cfg,
        train=train_cfg,
    )
    cfg.run_name = yaml_cfg.get("general", {}).get("name", "adaptive_run")
    devices_cfg = yaml_cfg.get("devices", {})
    cfg.system.device_uformer = devices_cfg.get("device_uformer", args.device_uformer)
    cfg.system.device_hypir = devices_cfg.get("device_hypir_text", args.device_hypir_text)
    cfg.uformer = uformer_cfg
    cfg.hypir.text_device = devices_cfg.get("device_hypir_text", args.device_hypir_text)
    cfg.hypir.vae_device = devices_cfg.get("device_hypir_vae", args.device_hypir_vae or args.device_hypir_text)
    cfg.hypir.generator_device = devices_cfg.get("device_hypir_gen", args.device_hypir_gen or args.device_hypir_text)
    hypir_cfg = yaml_cfg.get("hypir", {})
    if hypir_cfg:
        cfg.hypir.model_t = hypir_cfg.get("model_t", cfg.hypir.model_t)
        cfg.hypir.coeff_t = hypir_cfg.get("coeff_t", cfg.hypir.coeff_t)
        if hypir_cfg.get("base_model_path"):
            cfg.hypir.base_model_path = Path(hypir_cfg["base_model_path"])
        if hypir_cfg.get("weight_path"):
            cfg.hypir.weight_path = Path(hypir_cfg["weight_path"])

    dataset = PairedImageFolder(cfg.data.dataroot_lq, cfg.data.dataroot_gt, filename_tmpl=cfg.data.filename_tmpl)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if yaml_cfg.get("datasets", {}).get("val_1"):
        val_loader = DataLoader(
            PairedImageFolder(
                Path(yaml_cfg["datasets"]["val_1"]["dataroot_lq"]),
                Path(yaml_cfg["datasets"]["val_1"]["dataroot_gt"]),
                filename_tmpl=yaml_cfg["datasets"]["val_1"].get("filename_tmpl", "{}"),
            ),
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    trainer = DualStageTrainer(cfg, val_loader=val_loader, metrics_cfg=yaml_cfg.get("metrics", {}))
    if val_loader is not None and cfg.train.val_freq > 0 and cfg.train.initial_val:
        trainer.validate()
    if cfg.train.total_iters and cfg.train.total_iters > 0:
        epoch = 0
        while trainer.global_step < cfg.train.total_iters:
            trainer.train_epoch(loader, epoch, max_iters=cfg.train.total_iters)
            epoch += 1
    else:
        for epoch in range(cfg.train.epochs):
            trainer.train_epoch(loader, epoch)

    trainer.save_checkpoint(Path(cfg.train.output_dir) / "latest.pt")


if __name__ == "__main__":
    main()
