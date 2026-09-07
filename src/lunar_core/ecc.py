"""
Enhanced Correlation Coefficient (ECC) Maximization Refinement
Performs direct sub-pixel photometric image alignment invariant to linear illumination transformations.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2


class ECCRefinementEngine:
    """Iterative sub-pixel optimization using the Enhanced Correlation Coefficient criterion."""

    def __init__(
        self,
        warp_mode: str = "HOMOGRAPHY",
        max_iterations: int = 150,
        termination_eps: float = 1e-6,
        gauss_filt_size: int = 5
    ):
        mode_map = {
            "TRANSLATION": cv2.MOTION_TRANSLATION,
            "EUCLIDEAN": cv2.MOTION_EUCLIDEAN,
            "AFFINE": cv2.MOTION_AFFINE,
            "HOMOGRAPHY": cv2.MOTION_HOMOGRAPHY
        }
        self.warp_mode_str = warp_mode.upper()
        self.warp_mode = mode_map.get(self.warp_mode_str, cv2.MOTION_HOMOGRAPHY)
        self.max_iterations = max_iterations
        self.termination_eps = termination_eps
        self.gauss_filt_size = gauss_filt_size

    def refine(
        self,
        ref_img: np.ndarray,
        coarse_warped_src: np.ndarray,
        initial_H: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Run ECC optimization to refine alignment parameters down to sub-pixel accuracy (< 0.1 pixel).
        """
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.max_iterations, self.termination_eps)

        if initial_H is None:
            if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, 3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:
            if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = initial_H.astype(np.float32)
            else:
                warp_matrix = initial_H[:2, :].astype(np.float32)

        try:
            ecc_score, refined_matrix = cv2.findTransformECC(
                templateImage=ref_img,
                inputImage=coarse_warped_src,
                warpMatrix=warp_matrix,
                motionType=self.warp_mode,
                criteria=criteria,
                inputMask=None,
                gaussFiltSize=self.gauss_filt_size
            )

            # Warp image with refined matrix
            h, w = ref_img.shape[:2]
            if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
                refined_warped = cv2.warpPerspective(coarse_warped_src, refined_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                H_final = refined_matrix
            else:
                refined_warped = cv2.warpAffine(coarse_warped_src, refined_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                H_final = np.vstack([refined_matrix, [0, 0, 1]])

            # Measure alignment error improvement
            initial_error = float(np.mean(np.abs(ref_img.astype(np.float32) - coarse_warped_src.astype(np.float32))))
            final_error = float(np.mean(np.abs(ref_img.astype(np.float32) - refined_warped.astype(np.float32))))
            improvement_pct = round(((initial_error - final_error) / (initial_error + 1e-5)) * 100.0, 2)

            return {
                "success": True,
                "ecc_score": round(float(ecc_score), 4),
                "initial_alignment_error": round(initial_error, 2),
                "final_alignment_error": round(final_error, 2),
                "pixel_improvement_pct": improvement_pct,
                "refined_image": refined_warped,
                "refined_matrix": H_final
            }
        except Exception as e:
            return {
                "success": False,
                "error_message": str(e),
                "ecc_score": 0.0,
                "initial_alignment_error": 0.0,
                "final_alignment_error": 0.0,
                "pixel_improvement_pct": 0.0,
                "refined_image": coarse_warped_src,
                "refined_matrix": initial_H if initial_H is not None else np.eye(3)
            }
