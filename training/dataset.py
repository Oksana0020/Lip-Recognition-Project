"""Dataset classes for lip-reading .npy clips, shared by word and viseme."""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LipReadingDataset(Dataset):
    """Base dataset: loads .npy clips, resamples frames, crops, normalises"""
    def __init__(
        self,
        num_frames: int = 8,
        height: int = 64,
        width: int = 64,
        augment: bool = False,
        lip_bbox_lookup: Optional[Dict[str, List[int]]] = None
    ) -> None:
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.augment = augment
        self.lip_bbox_lookup = lip_bbox_lookup or {}
        self.samples = self.index_samples()
        print(f"Loaded {len(self.samples):,} samples")

    def index_samples(self) -> List[Dict]:
        """Override in subclass to return list of sample dicts."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        path = Path(sample["path"])
        label_index = sample["label_index"]
        try:
            frames = np.load(path, allow_pickle=False)
            if frames.size == 0:
                raise ValueError("empty array")
        except Exception:
            frames = np.zeros(
                (self.num_frames, self.height, self.width, 3),
                dtype=np.float32)
        frames = self._resample(frames)
        frames = self._crop_and_resize(frames, path)
        if self.augment:
            frames = self._augment(frames)
        # shape: [C=1, T, H, W]
        tensor = torch.from_numpy(frames).float()
        tensor = tensor.permute(3, 0, 1, 2)
        return tensor, label_index

    def _resample(self, frames: np.ndarray) -> np.ndarray:
        """Resample to self.num_frames by uniform sampling or padding."""
        n = frames.shape[0]
        if n >= self.num_frames:
            indices = np.linspace(0, n - 1, self.num_frames, dtype=int)
            return frames[indices]
        padded = list(frames)
        src = list(frames)
        while len(padded) < self.num_frames:
            padded.extend(src[:self.num_frames - len(padded)])
        return np.stack(padded[:self.num_frames], axis=0)

    def _crop_and_resize(self, frames: np.ndarray, path: Path) -> np.ndarray:
        """Crop lip region from bbox or fixed ratio, resize, to grayscale"""
        key = str(path.resolve()).lower()
        bbox = self.lip_bbox_lookup.get(key)
        result = []
        for frame in frames:
            h, w = frame.shape[:2]
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                frame = frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            else:
                frame = frame[
                    int(h * 0.66):int(h * 0.92),
                    int(w * 0.33):int(w * 0.67)]
            frame = cv2.resize(frame, (self.width, self.height))
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result.append(frame[..., np.newaxis])
        return np.stack(result, axis=0).astype(np.float32) / 255.0

    def _augment(self, frames: np.ndarray) -> np.ndarray:
        """Random horizontal flip and brightness jitter."""
        if random.random() < 0.5:
            frames = frames[:, :, ::-1, :].copy()
        if random.random() < 0.4:
            frames = np.clip(frames * random.uniform(0.8, 1.2), 0.0, 1.0)
        return frames


# concrete dataset subclasses
class WordDataset(LipReadingDataset):
    """Word-level dataset: indexes .npy clips by word label folder"""

    def __init__(
        self,
        data_root: Path,
        word_to_index: Dict[str, int],
        lip_bbox_lookup: Optional[Dict[str, List[int]]] = None,
        **kwargs
    ) -> None:
        self.data_root = data_root
        self.word_to_index = word_to_index
        super().__init__(lip_bbox_lookup=lip_bbox_lookup, **kwargs)

    def index_samples(self) -> List[Dict]:
        samples = []
        skip = {"extraction_summary.json", "word_statistics.json", "sp"}
        for folder in sorted(self.data_root.iterdir()):
            if not folder.is_dir() or folder.name in skip:
                continue
            word = folder.name
            if word not in self.word_to_index:
                continue
            for npy in folder.glob("*.npy"):
                samples.append({
                    "path": str(npy),
                    "label_index": self.word_to_index[word],
                    "word": word})
        return samples


class VisemeDataset(LipReadingDataset):
    """Viseme-level dataset: indexes .npy clips by viseme label folder."""

    def __init__(
        self,
        data_root: Path,
        viseme_to_index: Dict[str, int],
        **kwargs
    ) -> None:
        self.data_root = data_root
        self.viseme_to_index = viseme_to_index
        super().__init__(**kwargs)

    def index_samples(self) -> List[Dict]:
        samples = []
        for folder in sorted(self.data_root.iterdir()):
            if not folder.is_dir():
                continue
            viseme = folder.name
            if viseme not in self.viseme_to_index:
                continue
            for npy in folder.glob("*.npy"):
                samples.append({
                    "path": str(npy),
                    "label_index": self.viseme_to_index[viseme],
                    "viseme": viseme})
        return samples
