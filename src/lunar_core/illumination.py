"""
Illumination Variation Analysis Engine
Calculates luminance statistics, shadow coverage, dynamic range,
and classifies lunar optical imagery into difficulty tiers (Easy, Moderate, Difficult).
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2


class IlluminationAnalyzer:
    """Analyzes photometric conditions and solar incidence effects on lunar imagery."""

    def __init__(
        self,
        shadow_cutoff: int = 35,
        highlight_cutoff: int = 220
    ):
        self.shadow_cutoff = shadow_cutoff
        self.highlight_cutoff = highlight_cutoff

    def compute_illumination_metrics(self, img_gray: np.ndarray) -> Dict[str, Any]:
        """Compute mean brightness, variances, shadow/highlight percentages, and histogram entropy."""
        mean_b = float(np.mean(img_gray))
        var_b = float(np.var(img_gray))
        std_b = float(np.std(img_gray))

        # Michelson contrast
        max_v = float(np.max(img_gray))
        min_v = float(np.min(img_gray))
        michelson = (max_v - min_v) / (max_v + min_v + 1e-5)

        # RMS Contrast
        rms_contrast = std_b

        # Shadow & Highlight percentages
        shadow_pixels = np.count_nonzero(img_gray <= self.shadow_cutoff)
        highlight_pixels = np.count_nonzero(img_gray >= self.highlight_cutoff)
        total_pixels = img_gray.size

        shadow_pct = (shadow_pixels / total_pixels) * 100.0
        highlight_pct = (highlight_pixels / total_pixels) * 100.0

        # Histogram distribution & entropy
        hist, _ = np.histogram(img_gray, bins=256, range=(0, 256), density=True)
        hist_non_zero = hist[hist > 0]
        entropy = -float(np.sum(hist_non_zero * np.log2(hist_non_zero)))

        # Multi-tier difficulty classification
        # Easy: High contrast, low shadows, balanced histogram
        # Moderate: Medium shadows, some high-incidence grazing illumination
        # Difficult: Extreme low-sun grazing angles (polar craters) or washed-out nadir noon
        if shadow_pct < 15.0 and rms_contrast > 35.0 and entropy > 6.0:
            classification = "Easy"
            difficulty_score = 1.0
        elif shadow_pct < 45.0 and rms_contrast > 18.0:
            classification = "Moderate"
            difficulty_score = 2.0
        else:
            classification = "Difficult"
            difficulty_score = 3.0

        # Illumination Heatmap (normalized photometric map with solar gradient)
        illum_heatmap = cv2.GaussianBlur(img_gray.astype(np.float32) / 255.0, (31, 31), 10)

        return {
            "mean_brightness": round(mean_b, 2),
            "brightness_variance": round(var_b, 2),
            "brightness_std": round(std_b, 2),
            "michelson_contrast": round(michelson, 3),
            "rms_contrast": round(rms_contrast, 2),
            "shadow_coverage_pct": round(shadow_pct, 2),
            "highlight_coverage_pct": round(highlight_pct, 2),
            "entropy": round(entropy, 3),
            "classification": classification,
            "difficulty_score": difficulty_score,
            "illumination_heatmap": illum_heatmap
        }
