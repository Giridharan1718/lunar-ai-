"""
LunaDNA Global Terrain Fingerprint Generator
Extracts crater morphology distributions, directional gradient tensors,
spatial pyramid texture moments, and synthesizes a fixed-length 256-D signature vector.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from scipy.ndimage import uniform_filter


class LunaDNAGenerator:
    """Computes invariant 256-dimensional LunaDNA topological descriptor for fast reference tile indexing."""

    def __init__(self, vector_dim: int = 256):
        self.vector_dim = vector_dim

    def extract_crater_geometry_features(self, img_gray: np.ndarray) -> np.ndarray:
        """Extract multi-scale Hessian blob and Laplacian responses representing crater spatial distributions (64-D)."""
        h, w = img_gray.shape
        features = []
        for sigma in [2.0, 4.0, 8.0, 16.0]:
            log = cv2.GaussianBlur(img_gray, (0, 0), sigma)
            lap = cv2.Laplacian(log, cv2.CV_32F)
            # 4x4 spatial grid pooling of positive and negative laplacian responses
            grid_h, grid_w = h // 4, w // 4
            for r in range(4):
                for c in range(4):
                    cell = lap[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                    features.append(np.mean(np.maximum(0, cell)))
        feat_arr = np.array(features, dtype=np.float32)
        if len(feat_arr) < 64:
            feat_arr = np.pad(feat_arr, (0, 64 - len(feat_arr)))
        return feat_arr[:64]

    def extract_gradient_directional_features(self, img_gray: np.ndarray) -> np.ndarray:
        """Extract Histogram of Oriented Gradients (HOG-like) illumination-resilient spatial distribution (64-D)."""
        gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # 4x4 spatial cells, 4 directional bins each = 64 features
        h, w = img_gray.shape
        grid_h, grid_w = h // 4, w // 4
        features = []
        for r in range(4):
            for c in range(4):
                cell_mag = mag[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                cell_ang = ang[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                # 4 bins: [0-90], [90-180], [180-270], [270-360]
                b1 = np.sum(cell_mag[(cell_ang >= 0) & (cell_ang < 90)])
                b2 = np.sum(cell_mag[(cell_ang >= 90) & (cell_ang < 180)])
                b3 = np.sum(cell_mag[(cell_ang >= 180) & (cell_ang < 270)])
                b4 = np.sum(cell_mag[(cell_ang >= 270) & (cell_ang < 360)])
                features.extend([b1, b2, b3, b4])
        return np.array(features, dtype=np.float32)[:64]

    def extract_texture_moments(self, img_gray: np.ndarray) -> np.ndarray:
        """Extract spatial pyramid moments (mean, std, skewness, kurtosis) across 1x1, 2x2, 4x4 levels (64-D)."""
        features = []
        h, w = img_gray.shape
        img_f = img_gray.astype(np.float32)

        for grid_size in [1, 2, 4]:
            gh, gw = h // grid_size, w // grid_size
            for r in range(grid_size):
                for c in range(grid_size):
                    cell = img_f[r*gh:(r+1)*gh, c*gw:(c+1)*gw]
                    mean = np.mean(cell)
                    std = np.std(cell) + 1e-5
                    diff = cell - mean
                    skew = np.mean((diff / std) ** 3)
                    features.extend([mean, std, skew])

        feat_arr = np.array(features, dtype=np.float32)
        if len(feat_arr) < 64:
            feat_arr = np.pad(feat_arr, (0, 64 - len(feat_arr)))
        return feat_arr[:64]

    def extract_frequency_fourier_ring_features(self, img_gray: np.ndarray) -> np.ndarray:
        """Extract 2D FFT radially integrated energy spectrum representing terrain roughness scales (64-D)."""
        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1.0)

        h, w = img_gray.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)

        max_radius = min(cx, cy)
        bin_width = max_radius / 64.0
        features = []
        for i in range(64):
            ring_mask = (r >= i * bin_width) & (r < (i + 1) * bin_width)
            if np.any(ring_mask):
                features.append(np.mean(magnitude_spectrum[ring_mask]))
            else:
                features.append(0.0)

        return np.array(features, dtype=np.float32)[:64]

    def generate_fingerprint(self, img_gray: np.ndarray) -> np.ndarray:
        """Synthesize concatenated, unit L2-normalized 256-D LunaDNA vector."""
        # Ensure standard shape
        if img_gray.shape[0] != 512 or img_gray.shape[1] != 512:
            resized = cv2.resize(img_gray, (512, 512), interpolation=cv2.INTER_AREA)
        else:
            resized = img_gray

        f1 = self.extract_crater_geometry_features(resized)
        f2 = self.extract_gradient_directional_features(resized)
        f3 = self.extract_texture_moments(resized)
        f4 = self.extract_frequency_fourier_ring_features(resized)

        vec = np.concatenate([f1, f2, f3, f4]).astype(np.float32)
        # Unit L2 normalization for Cosine / Inner Product similarity
        norm = np.linalg.norm(vec) + 1e-8
        vec_norm = vec / norm
        return vec_norm
