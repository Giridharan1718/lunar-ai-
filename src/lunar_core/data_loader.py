"""
Lunar Dataset Scanner & Image Loader
Handles scanning directories, format identification (PDS4, GeoTIFF, PNG, JPEG, NPY),
MD5 duplicate detection, corruption checking, and statistical aggregation.
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
from PIL import Image


class LunarImageLoader:
    """Robust image loader supporting 8-bit, 16-bit, GeoTIFF, PNG, JPG, and NPY arrays."""

    @staticmethod
    def load_image(filepath: str, to_grayscale: bool = True) -> Optional[np.ndarray]:
        path = Path(filepath)
        if not path.exists():
            return None

        ext = path.suffix.lower()
        try:
            if ext == ".npy":
                arr = np.load(str(path))
                if arr.ndim == 3 and to_grayscale:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                return arr.astype(np.uint8) if arr.max() <= 255 else (arr / arr.max() * 255).astype(np.uint8)

            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if to_grayscale else cv2.IMREAD_COLOR)
            if img is not None:
                return img

            # Fallback to Pillow
            with Image.open(str(path)) as pil_img:
                if to_grayscale:
                    pil_img = pil_img.convert("L")
                return np.array(pil_img)
        except Exception:
            return None


class LunarDatasetScanner:
    """Scans dataset directories, detects duplicates via MD5, checks corruption, and produces statistical summaries."""

    def __init__(self, data_dirs: List[str]):
        self.data_dirs = [Path(d) for d in data_dirs]

    @staticmethod
    def compute_md5(filepath: Path, block_size: int = 65536) -> str:
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(block_size), b""):
                hasher.update(block)
        return hasher.hexdigest()

    def scan(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        records = []
        md5_map: Dict[str, str] = {}
        duplicates = []
        corrupted = []

        valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".bmp"}

        for d in self.data_dirs:
            if not d.exists():
                continue
            for root, _, files in os.walk(d):
                for file in files:
                    fp = Path(root) / file
                    ext = fp.suffix.lower()
                    if ext not in valid_extensions:
                        continue

                    file_size_bytes = fp.stat().st_size
                    file_md5 = self.compute_md5(fp)

                    if file_md5 in md5_map:
                        duplicates.append((str(fp), md5_map[file_md5]))
                        is_dup = True
                    else:
                        md5_map[file_md5] = str(fp)
                        is_dup = False

                    img = LunarImageLoader.load_image(str(fp), to_grayscale=True)
                    if img is None or img.size == 0:
                        corrupted.append(str(fp))
                        is_corrupt = True
                        height, width = 0, 0
                        mean_b, std_b, min_b, max_b, contrast = 0.0, 0.0, 0, 0, 0.0
                    else:
                        is_corrupt = False
                        height, width = img.shape[:2]
                        mean_b = float(np.mean(img))
                        std_b = float(np.std(img))
                        min_b = int(np.min(img))
                        max_b = int(np.max(img))
                        contrast = float(std_b / (mean_b + 1e-5))

                    records.append({
                        "file_path": str(fp),
                        "file_name": fp.name,
                        "folder": fp.parent.name,
                        "extension": ext,
                        "file_size_kb": round(file_size_bytes / 1024.0, 2),
                        "md5_hash": file_md5,
                        "is_duplicate": is_dup,
                        "is_corrupted": is_corrupt,
                        "height": height,
                        "width": width,
                        "aspect_ratio": round(width / (height + 1e-5), 3) if height > 0 else 0,
                        "mean_brightness": round(mean_b, 2),
                        "std_brightness": round(std_b, 2),
                        "min_brightness": min_b,
                        "max_brightness": max_b,
                        "contrast_ratio": round(contrast, 3)
                    })

        df = pd.DataFrame(records)
        summary = {
            "total_images_scanned": len(records),
            "valid_images_count": len(records) - len(corrupted),
            "corrupted_images_count": len(corrupted),
            "duplicate_images_count": len(duplicates),
            "formats_distribution": df["extension"].value_counts().to_dict() if not df.empty else {},
            "folders_distribution": df["folder"].value_counts().to_dict() if not df.empty else {},
            "mean_resolution": f"{int(df['width'].mean())}x{int(df['height'].mean())}" if not df.empty and df['width'].sum() > 0 else "0x0",
            "mean_dataset_brightness": round(float(df['mean_brightness'].mean()), 2) if not df.empty else 0.0,
            "mean_dataset_contrast": round(float(df['contrast_ratio'].mean()), 3) if not df.empty else 0.0,
        }
        return df, summary
