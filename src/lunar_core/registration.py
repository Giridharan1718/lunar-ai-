"""
Lunar Image Registration Engine
Estimates Projective Transformation (Homography), decomposes rotation/scale/translation,
performs sub-pixel bilinear warping, and builds overlay and difference maps.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2


class RegistrationEngine:
    """Core image registration and geometric alignment engine."""

    @staticmethod
    def decompose_homography(H: np.ndarray) -> Dict[str, float]:
        """
        Decompose affine/projective homography to derive rotation angle, scale factors, and translations.
        """
        if H is None or H.shape != (3, 3):
            return {"rotation_deg": 0.0, "scale_x": 1.0, "scale_y": 1.0, "tx": 0.0, "ty": 0.0}

        # Normalize H such that H[2, 2] = 1
        H_norm = H / (H[2, 2] + 1e-10)

        a11, a12 = H_norm[0, 0], H_norm[0, 1]
        a21, a22 = H_norm[1, 0], H_norm[1, 1]
        tx, ty = H_norm[0, 2], H_norm[1, 2]

        scale_x = np.sqrt(a11**2 + a21**2)
        scale_y = np.sqrt(a12**2 + a22**2)
        rot_rad = np.arctan2(a21, a11)
        rot_deg = float(np.degrees(rot_rad))

        return {
            "rotation_deg": round(rot_deg, 3),
            "scale_x": round(float(scale_x), 4),
            "scale_y": round(float(scale_y), 4),
            "scale_mean": round(float((scale_x + scale_y) / 2.0), 4),
            "tx": round(float(tx), 2),
            "ty": round(float(ty), 2)
        }

    @staticmethod
    def warp_image(
        src_img: np.ndarray,
        H: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Warp source image onto reference coordinate frame."""
        h, w = target_shape[:2]
        return cv2.warpPerspective(src_img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    @staticmethod
    def create_checkerboard_overlay(img1: np.ndarray, img2: np.ndarray, square_size: int = 64) -> np.ndarray:
        """Generate checkerboard visualization interleaving reference and registered source."""
        h, w = img1.shape[:2]
        cb = np.zeros_like(img1)
        for y in range(0, h, square_size):
            for x in range(0, w, square_size):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    cb[y:y+square_size, x:x+square_size] = img1[y:y+square_size, x:x+square_size]
                else:
                    cb[y:y+square_size, x:x+square_size] = img2[y:y+square_size, x:x+square_size]
        return cb

    @staticmethod
    def create_difference_map(img_ref: np.ndarray, img_warped: np.ndarray) -> np.ndarray:
        """Compute absolute difference map highlighting alignment residuals."""
        mask = (img_warped > 0).astype(np.uint8)
        diff = cv2.absdiff(img_ref, img_warped) * mask
        # Colorize difference using colormap
        diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        diff_color[mask == 0] = 0
        return diff_color

    def register(
        self,
        src_img: np.ndarray,
        ref_img: np.ndarray,
        H: np.ndarray
    ) -> Dict[str, Any]:
        """Execute full registration pipeline."""
        warped = self.warp_image(src_img, H, ref_img.shape)
        params = self.decompose_homography(H)
        checkerboard = self.create_checkerboard_overlay(ref_img, warped)
        diff_map = self.create_difference_map(ref_img, warped)

        # False color overlay: Red = Ref, Green = Warped Source
        overlay = np.zeros((ref_img.shape[0], ref_img.shape[1], 3), dtype=np.uint8)
        overlay[:, :, 2] = ref_img # Red
        overlay[:, :, 1] = warped  # Green
        overlay[:, :, 0] = (ref_img // 2 + warped // 2) # Blue blend

        return {
            "registered_image": warped,
            "checkerboard": checkerboard,
            "difference_map": diff_map,
            "color_overlay": overlay,
            "decomposition": params,
            "transformation_matrix": H
        }
