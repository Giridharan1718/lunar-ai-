"""
Synthetic Lunar Terrain & Chandrayaan-2 Optical Image Pair Generator
Generates realistic cratered lunar surfaces, solar incidence shadowing,
homography projective transformations, and illumination shifts for testing and validation.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List


class LunarSyntheticGenerator:
    """Generates synthetic cratered lunar surfaces, warped test pairs, and ground truth correspondences."""

    def __init__(self, size: Tuple[int, int] = (1024, 1024), seed: int = 42):
        self.height, self.width = size
        self.seed = seed
        np.random.seed(seed)

    def generate_lunar_surface(
        self,
        num_craters: int = 45,
        sun_azimuth_deg: float = 45.0,
        sun_elevation_deg: float = 30.0,
        noise_level: float = 0.08
    ) -> np.ndarray:
        """
        Generate a single-channel 8-bit lunar surface simulation with realistic crater morphology:
        rim elevation, floor depression, solar directional shadowing, and micro-regolith texture.
        """
        # 1. Base fractal regolith noise (Perlin-like multi-scale noise)
        base = np.zeros((self.height, self.width), dtype=np.float32)
        for scale in [64, 32, 16, 8, 4]:
            h_s, w_s = max(2, self.height // scale), max(2, self.width // scale)
            noise = np.random.normal(0, 1.0, (h_s, w_s)).astype(np.float32)
            noise_resized = cv2.resize(noise, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            base += noise_resized * (scale / 64.0)

        # Normalize base surface height map [-1.0, 1.0]
        base = (base - base.min()) / (base.max() - base.min() + 1e-8) * 2.0 - 1.0
        elevation = base * 0.15

        # 2. Add randomized impact craters
        craters = []
        for _ in range(num_craters):
            cx = np.random.randint(int(self.width * 0.05), int(self.width * 0.95))
            cy = np.random.randint(int(self.height * 0.05), int(self.height * 0.95))
            radius = np.random.choice([
                np.random.randint(8, 25),
                np.random.randint(25, 60),
                np.random.randint(60, 140)
            ], p=[0.6, 0.3, 0.1])
            depth = np.random.uniform(0.3, 0.8)
            craters.append((cx, cy, radius, depth))

        # Sort craters by radius descending so smaller craters overlap larger ones
        craters.sort(key=lambda c: c[2], reverse=True)

        yy, xx = np.mgrid[0:self.height, 0:self.width]
        for cx, cy, radius, depth in craters:
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            mask = dist < (radius * 1.5)
            r_norm = dist / (radius + 1e-5)
            # Crater profile: central bowl depression + raised rim
            bowl = -depth * np.maximum(0.0, 1.0 - (r_norm ** 2))
            rim = (depth * 0.35) * np.exp(-((r_norm - 1.0) ** 2) / 0.05)
            elevation[mask] += (bowl + rim)[mask]

        # 3. Shaded relief / Lambertian illumination rendering
        # Gradients dz/dx, dz/dy
        sobel_x = cv2.Sobel(elevation, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(elevation, cv2.CV_32F, 0, 1, ksize=3)

        # Surface normal vector N = (-dz/dx, -dz/dy, 1) / |N|
        nx = -sobel_x
        ny = -sobel_y
        nz = np.ones_like(elevation)
        norm_len = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
        nx /= norm_len
        ny /= norm_len
        nz /= norm_len

        # Sun direction vector S
        az_rad = np.radians(sun_azimuth_deg)
        el_rad = np.radians(sun_elevation_deg)
        sx = np.cos(el_rad) * np.sin(az_rad)
        sy = np.cos(el_rad) * np.cos(az_rad)
        sz = np.sin(el_rad)

        # Lambertian reflectance: I = max(0, N . S)
        reflectance = nx * sx + ny * sy + nz * sz
        reflectance = np.clip(reflectance, 0.0, 1.0)

        # Add albedo variation and high-frequency regolith grain
        albedo = 0.5 + 0.5 * (base - base.min()) / (base.max() - base.min() + 1e-8)
        image = reflectance * albedo
        image += np.random.normal(0, noise_level, image.shape)
        image = np.clip(image, 0.0, 1.0)

        # Convert to uint8 (0-255)
        image_uint8 = (image * 255.0).astype(np.uint8)
        return image_uint8

    def generate_registered_pair(
        self,
        rotation_deg: float = 12.5,
        scale: float = 1.08,
        tx: float = 45.0,
        ty: float = -30.0,
        shear_x: float = 0.04,
        sun_diff_azimuth: float = 35.0
    ) -> Dict[str, Any]:
        """
        Generate a reference image and a warped, re-illuminated source image
        along with ground truth 3x3 Homography and correspondence points.
        """
        ref_img = self.generate_lunar_surface(
            num_craters=50,
            sun_azimuth_deg=45.0,
            sun_elevation_deg=35.0
        )

        # Build ground-truth homography transformation
        cx, cy = self.width / 2.0, self.height / 2.0
        rot_rad = np.radians(rotation_deg)
        cos_a, sin_a = np.cos(rot_rad), np.sin(rot_rad)

        T_center = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
        Aff = np.array([
            [scale * cos_a, -scale * sin_a + shear_x, 0],
            [scale * sin_a, scale * cos_a, 0],
            [0.00008, -0.00005, 1.0] # slight perspective tilt
        ], dtype=np.float64)
        T_back = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float64)

        H_gt = T_back @ Aff @ T_center

        # Source image is warped from the surface under different illumination angle
        src_base = self.generate_lunar_surface(
            num_craters=50,
            sun_azimuth_deg=45.0 + sun_diff_azimuth,
            sun_elevation_deg=25.0
        )
        src_img = cv2.warpPerspective(src_base, H_gt, (self.width, self.height), flags=cv2.INTER_LINEAR)

        # Generate ground truth correspondence grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(100, self.width - 100, 15),
            np.linspace(100, self.height - 100, 15)
        )
        pts_ref = np.vstack([grid_x.ravel(), grid_y.ravel(), np.ones_like(grid_x.ravel())])
        pts_src_homo = H_gt @ pts_ref
        pts_src_x = pts_src_homo[0] / pts_src_homo[2]
        pts_src_y = pts_src_homo[1] / pts_src_homo[2]

        pts_ref_2d = pts_ref[:2].T.astype(np.float32)
        pts_src_2d = np.column_stack([pts_src_x, pts_src_y]).astype(np.float32)

        # Keep points within frame bounds
        valid_mask = (
            (pts_src_2d[:, 0] >= 10) & (pts_src_2d[:, 0] < self.width - 10) &
            (pts_src_2d[:, 1] >= 10) & (pts_src_2d[:, 1] < self.height - 10)
        )

        return {
            "reference_image": ref_img,
            "source_image": src_img,
            "homography_ground_truth": H_gt,
            "ref_points": pts_ref_2d[valid_mask],
            "src_points": pts_src_2d[valid_mask],
            "rotation_deg": rotation_deg,
            "scale": scale,
            "translation": (tx, ty)
        }

    def populate_sample_datasets(self, data_root: str = "data") -> List[str]:
        """Populate subfolders with synthetic lunar test datasets."""
        subfolders = ["ohrc", "tmc", "iirc", "quickmap-lroc", "reference", "raw"]
        generated_paths = []
        for sf in subfolders:
            folder = Path(data_root) / sf
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(1, 6):
                img = self.generate_lunar_surface(
                    num_craters=30 + i * 5,
                    sun_azimuth_deg=30.0 + i * 25.0,
                    sun_elevation_deg=20.0 + i * 8.0,
                    seed=100 + i * 7
                )
                filename = folder / f"lunar_{sf}_{i:03d}.png"
                cv2.imwrite(str(filename), img)
                generated_paths.append(str(filename))
        return generated_paths
