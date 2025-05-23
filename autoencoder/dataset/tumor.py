import typing as t
from pathlib import Path

import nibabel
import numpy as np
import numpy.random
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
import torch
import nibabel
from nibabel import nifti1
import torch.nn.functional as F
import random


class BraTS2021Dataset:
    def __init__(self,
                 root: Path, modalites: t.Tuple[str],
                 slice_range: t.List[int],
                 masked: t.List[int] | None = None,
                 ):
        root = Path(root)
        self.root = root
        self.modalites = modalites
        self.masked = masked
        self.slice_range = slice_range

        if not root.exists():
            raise FileNotFoundError(f"Dataset root {root} does not exist.")
        if masked is not None and any(masked) > len(modalites):
            raise ValueError(
                f"Masked indices {masked} are out of range for modalites {modalites}.")
        self.slice_range = sorted(slice_range)  # Ensure range is sorted

        self.samples = []
        for sample in root.iterdir():
            if sample.is_dir():
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples) * (self.slice_range[1] - self.slice_range[0])

    def __getitem__(self, idx: int):
        sample_idx = idx // (self.slice_range[1] - self.slice_range[0])
        slice_idx = idx % (self.slice_range[1] - self.slice_range[0])
        slice_idx = self.slice_range[0] + slice_idx
        sample = self.samples[sample_idx]
        modalities = []
        if getattr(self, 'normalize_params', None) is None:
            self.normalize_params = {}
        if self.normalize_params.get(sample_idx, None) is None:
            self.normalize_params[sample_idx] = {}

        for mod in self.modalites:
            mod_path = sample / f"{sample.stem}_{mod}.nii.gz"
            if not mod_path.exists():
                raise FileNotFoundError(
                    f"Modality {mod} for sample {sample} does not exist.")

            img = nifti1.load(mod_path)
            if self.normalize_params[sample_idx].get(mod, None) is None:
                data = img.get_fdata(dtype=np.float32)
                vmax = np.max(data)
                vmin = np.min(data)
                self.normalize_params[sample_idx][mod] = (vmin, vmax)

            img_slice = img.dataobj[:, :, slice_idx]
            vmin, vmax = self.normalize_params[sample_idx][mod]
            # normalize to [-1, 1]
            img_slice = (img_slice - vmin) / (vmax - vmin) * 2 - 1
            img_slice = np.clip(img_slice, -1, 1)
            img_slice = img_slice.astype(np.float32)

            modalities.append(img_slice)
        modalities = np.stack(modalities, axis=-1)

        if self.masked is not None:
            mask = np.ones_like(modalities)
            for mod in self.masked:
                mask[:, :, mod] = 0
        else:
            prob = np.random.uniform(0.25, 0.75)
            mask = np.random.uniform(0, 1, (1, 1, modalities.shape[-1]))
            mask = np.where(mask > prob, 1, 0)

        masked_modalities = (modalities * mask).astype(np.float32)
        ground_truth = modalities.astype(np.float32)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(192),
            AddGaussianNoise(), 
            transforms.Lambda(lambda x: torch.clamp(x, -1.0, 1.0))
        ])
        return transform(masked_modalities), transform(ground_truth)
    
class AddGaussianNoise:
    def __init__(self, sigma_range=(0, 0.1)):
        self.sigma_range = sigma_range

    def __call__(self, img):
        sigma = random.uniform(*self.sigma_range)
        noise = torch.randn_like(img) * sigma
        return img + noise

