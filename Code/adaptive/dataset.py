from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class PairedImageFolder(Dataset):
    """
    Simple paired dataset that matches files by stem between lq and gt folders.
    """

    def __init__(self, dataroot_lq: Path, dataroot_gt: Path, filename_tmpl: str = "{}"):
        self.dataroot_lq = Path(dataroot_lq)
        self.dataroot_gt = Path(dataroot_gt)
        self.filename_tmpl = filename_tmpl

        self.lq_files = sorted([p for p in self.dataroot_lq.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not self.lq_files:
            raise ValueError(f"No images found in {self.dataroot_lq}")

        self.gt_map = {p.stem: p for p in self.dataroot_gt.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}

        missing: List[str] = []
        for lq in self.lq_files:
            stem = lq.stem
            if stem not in self.gt_map:
                missing.append(stem)
        if missing:
            raise ValueError(f"Missing ground-truth files for {len(missing)} items, e.g., {missing[:5]}")

    def __len__(self) -> int:
        return len(self.lq_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lq_path = self.lq_files[idx]
        stem = lq_path.stem
        gt_path = self.gt_map[stem]
        lq_img = Image.open(lq_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        lq = torch.from_numpy(np.array(lq_img).astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        gt = torch.from_numpy(np.array(gt_img).astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        return lq, gt
