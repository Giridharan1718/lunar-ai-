"""
Multi-Image Lunar Mosaic Builder & Seam Blender
Stitches multiple registered lunar frames into a seamless, high-resolution global terrain map
using multi-band / distance-weighted feather blending.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import cv2


class LunarMosaicBuilder:
    """Combines multiple registered lunar tiles into a composite mosaic."""

    def __init__(self, blend_mode: str = "feather", feather_width: int = 40):
        self.blend_mode = blend_mode
        self.feather_width = feather_width

    def blend_pair(self, base_canvas: np.ndarray, base_weight: np.ndarray, new_img: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Warp new image onto canvas coordinate frame and blend with existing mosaic canvas."""
        h_canv, w_canv = base_canvas.shape[:2]
        warped_img = cv2.warpPerspective(new_img, H, (w_canv, h_canv), flags=cv2.INTER_LINEAR)

        # Generate weight map (distance transform inside valid region)
        mask = (warped_img > 0).astype(np.uint8)
        if not np.any(mask):
            return base_canvas, base_weight

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() > 0:
            weight = np.clip(dist / float(self.feather_width), 0.0, 1.0)
        else:
            weight = mask.astype(np.float32)

        # Accumulate weighted image
        total_weight = base_weight + weight
        blended = (base_canvas * base_weight + warped_img.astype(np.float32) * weight) / (total_weight + 1e-5)

        return blended, total_weight

    def build_mosaic(
        self,
        reference_image: np.ndarray,
        registered_pairs: List[Tuple[np.ndarray, np.ndarray]], # [(img_i, H_i), ...]
        canvas_scale: float = 1.5
    ) -> Dict[str, Any]:
        """
        Build a wide-field mosaic from reference tile and registered source images.
        """
        h_ref, w_ref = reference_image.shape[:2]
        h_canv = int(h_ref * canvas_scale)
        w_canv = int(w_ref * canvas_scale)

        # Offset reference to center of canvas
        dx = (w_canv - w_ref) // 2
        dy = (h_canv - h_ref) // 2
        T_center = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)

        canvas = np.zeros((h_canv, w_canv), dtype=np.float32)
        weight_map = np.zeros((h_canv, w_canv), dtype=np.float32)

        # Place reference
        ref_warped = cv2.warpPerspective(reference_image, T_center, (w_canv, h_canv))
        ref_mask = (ref_warped > 0).astype(np.uint8)
        ref_dist = cv2.distanceTransform(ref_mask, cv2.DIST_L2, 5)
        ref_weight = np.clip(ref_dist / float(self.feather_width), 0.0, 1.0) if ref_dist.max() > 0 else ref_mask.astype(np.float32)

        canvas = ref_warped.astype(np.float32)
        weight_map = ref_weight

        for img, H in registered_pairs:
            H_composite = T_center @ H
            canvas, weight_map = self.blend_pair(canvas, weight_map, img, H_composite)

        final_mosaic = np.clip(canvas, 0, 255).astype(np.uint8)

        # Compute mosaic quality metrics
        valid_pixels = np.count_nonzero(final_mosaic > 0)
        coverage_pct = round((valid_pixels / final_mosaic.size) * 100.0, 2)
        seam_smoothness = round(float(1.0 / (np.std(final_mosaic[final_mosaic > 0]) + 1e-4) * 10.0), 3)

        return {
            "mosaic_image": final_mosaic,
            "dimensions": (w_canv, h_canv),
            "coverage_pct": coverage_pct,
            "seam_smoothness_score": seam_smoothness,
            "total_tiles_stitched": len(registered_pairs) + 1
        }
