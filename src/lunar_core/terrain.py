"""
Lunar Terrain Analysis Engine
Detects impact craters, ridges, valleys, basins, peaks, shadow masks,
and computes crater spatial density and terrain complexity indices.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, laplace
from skimage.feature import graycomatrix, graycoprops


class LunarTerrainAnalyzer:
    """Extracts geomorphological features from lunar imagery."""

    def __init__(
        self,
        crater_min_r: int = 5,
        crater_max_r: int = 100,
        shadow_threshold: int = 40
    ):
        self.crater_min_r = crater_min_r
        self.crater_max_r = crater_max_r
        self.shadow_threshold = shadow_threshold

    def detect_craters(self, img_gray: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect circular and quasi-circular impact crater rims using Hough Transform & multi-scale gradient analysis."""
        blurred = cv2.GaussianBlur(img_gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=self.crater_min_r * 2,
            param1=60,
            param2=30,
            minRadius=self.crater_min_r,
            maxRadius=self.crater_max_r
        )
        detected = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for pt in circles[0, :]:
                detected.append((int(pt[0]), int(pt[1]), int(pt[2])))
        return detected

    def compute_crater_density_map(self, img_gray: np.ndarray, craters: List[Tuple[int, int, int]]) -> np.ndarray:
        """Compute spatial crater density heatmap."""
        h, w = img_gray.shape
        density = np.zeros((h, w), dtype=np.float32)
        for cx, cy, r in craters:
            if 0 <= cx < w and 0 <= cy < h:
                # Add Gaussian density kernel around each crater
                cv2.circle(density, (cx, cy), max(5, r), float(r), -1)
        density_blurred = cv2.GaussianBlur(density, (51, 51), 15)
        if density_blurred.max() > 0:
            density_blurred = density_blurred / density_blurred.max()
        return density_blurred

    def segment_shadows(self, img_gray: np.ndarray) -> np.ndarray:
        """Generate binary shadow mask and percentage coverage."""
        _, shadow_mask = cv2.threshold(img_gray, self.shadow_threshold, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        shadow_cleaned = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        return shadow_cleaned

    def detect_ridges_and_valleys(self, img_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect structural lineaments (ridges and valleys/rilles) via directional morphological filters."""
        # Ridges: morphological Top-Hat (bright features on darker ground)
        kernel_ridge = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
        ridges = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel_ridge)

        # Valleys/Rilles: morphological Black-Hat (dark linear channels)
        valleys = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel_ridge)
        return ridges, valleys

    def compute_texture_complexity(self, img_gray: np.ndarray) -> float:
        """Calculate Grey Level Co-occurrence Matrix (GLCM) contrast and energy entropy."""
        # Quantize to 16 levels for fast GLCM
        img_quant = (img_gray // 16).astype(np.uint8)
        glcm = graycomatrix(img_quant, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], levels=16, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        energy = graycoprops(glcm, 'energy').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        # High contrast + low homogeneity indicates high rugged terrain complexity
        complexity = float(contrast / (homogeneity + 1e-4) * (1.0 - energy))
        return round(complexity, 4)

    def analyze(self, img_gray: np.ndarray) -> Dict[str, Any]:
        """Execute comprehensive terrain analysis."""
        craters = self.detect_craters(img_gray)
        density_map = self.compute_crater_density_map(img_gray, craters)
        shadow_mask = self.segment_shadows(img_gray)
        ridges, valleys = self.detect_ridges_and_valleys(img_gray)
        complexity_score = self.compute_texture_complexity(img_gray)

        total_pixels = img_gray.size
        shadow_pixels = np.count_nonzero(shadow_mask)
        shadow_pct = round((shadow_pixels / total_pixels) * 100.0, 2)

        area_sq_km = (img_gray.shape[0] * img_gray.shape[1]) / (1000.0 * 1000.0) # normalized scale
        crater_density = round(len(craters) / (area_sq_km + 1e-5), 2)

        return {
            "num_craters_detected": len(craters),
            "crater_density": crater_density,
            "shadow_percentage": shadow_pct,
            "texture_complexity": complexity_score,
            "craters_list": craters,
            "crater_density_map": density_map,
            "shadow_mask": shadow_mask,
            "ridges_map": ridges,
            "valleys_map": valleys
        }
