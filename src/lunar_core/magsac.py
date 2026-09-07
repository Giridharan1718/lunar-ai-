"""
MAGSAC++ Robust Geometric Filtering
Applies Marginalizing Sample Consensus (MAGSAC++) for robust outlier rejection,
epipolar/homography verification, and sub-pixel inlier determination.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2


class MAGSACFilter:
    """Filters noisy keypoint correspondences with MAGSAC++ robust estimator."""

    def __init__(
        self,
        threshold_px: float = 3.0,
        confidence: float = 0.999,
        max_iters: int = 10000,
        model_type: str = "HOMOGRAPHY"
    ):
        self.threshold_px = threshold_px
        self.confidence = confidence
        self.max_iters = max_iters
        self.model_type = model_type.upper()

    def filter(
        self,
        pts_src: np.ndarray,
        pts_ref: np.ndarray
    ) -> Dict[str, Any]:
        """
        Estimate transformation model using cv2.USAC_MAGSAC and separate inliers/outliers.
        """
        if len(pts_src) < 4 or len(pts_ref) < 4:
            return {
                "inlier_mask": np.zeros(len(pts_src), dtype=bool),
                "inliers_src": np.empty((0, 2), dtype=np.float32),
                "inliers_ref": np.empty((0, 2), dtype=np.float32),
                "outliers_src": pts_src,
                "outliers_ref": pts_ref,
                "inlier_count": 0,
                "outlier_count": len(pts_src),
                "inlier_ratio": 0.0,
                "matrix": None
            }

        # cv2.USAC_MAGSAC provides OpenCV's MAGSAC++ implementation
        method = cv2.USAC_MAGSAC

        if self.model_type == "AFFINE":
            matrix, inliers = cv2.estimateAffine2D(
                pts_src, pts_ref,
                method=method,
                ransacReprojThreshold=self.threshold_px,
                maxIters=self.max_iters,
                confidence=self.confidence
            )
            # Expand to 3x3 for consistency
            if matrix is not None:
                H = np.vstack([matrix, [0, 0, 1]])
            else:
                H = None
        else: # HOMOGRAPHY
            H, inliers = cv2.findHomography(
                pts_src, pts_ref,
                method=method,
                ransacReprojThreshold=self.threshold_px,
                maxIters=self.max_iters,
                confidence=self.confidence
            )

        if inliers is None or H is None:
            mask = np.zeros(len(pts_src), dtype=bool)
        else:
            mask = inliers.ravel().astype(bool)

        inlier_count = int(np.sum(mask))
        outlier_count = len(pts_src) - inlier_count
        inlier_ratio = float(inlier_count / len(pts_src)) if len(pts_src) > 0 else 0.0

        return {
            "inlier_mask": mask,
            "inliers_src": pts_src[mask],
            "inliers_ref": pts_ref[mask],
            "outliers_src": pts_src[~mask],
            "outliers_ref": pts_ref[~mask],
            "inlier_count": inlier_count,
            "outlier_count": outlier_count,
            "inlier_ratio": round(inlier_ratio, 4),
            "matrix": H
        }
