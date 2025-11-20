from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png"}

class ImageFolderFlat(Dataset):
    def __init__(self, root: str, split: str = "val", transform=None, limit: Optional[int] = None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.paths: List[Path] = []
        split_dir = self.root / split
        for p in split_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                self.paths.append(p)
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"No images found under {split_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(path)
