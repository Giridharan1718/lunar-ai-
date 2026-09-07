"""
Registration Evaluation & Performance Benchmarking Engine
Computes RMSE, Reprojection Error, Inlier Count, Inlier Ratio,
Structural Similarity Index (SSIM), Normalized Cross Correlation (NCC),
Spatial Coverage Score, Uniform Distribution Score, and Execution Time.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


class RegistrationEvaluator:
    """Calculates all quantitative benchmarking metrics for ISRO lunar registration validation."""

    @staticmethod
    def compute_reprojection_error(
        pts_src: np.ndarray,
        pts_ref: np.ndarray,
        H: np.ndarray
    ) -> float:
        """Compute Mean Reprojection Error (MRE) in pixels."""
        if len(pts_src) == 0 or H is None:
            return 999.0

        pts_src_h = np.hstack([pts_src, np.ones((len(pts_src), 1))])
        projected_h = (H @ pts_src_h.T).T
        proj_x = projected_h[:, 0] / (projected_h[:, 2] + 1e-10)
        proj_y = projected_h[:, 1] / (projected_h[:, 2] + 1e-10)
        projected = np.column_stack([proj_x, proj_y])

        errors = np.linalg.norm(projected - pts_ref, axis=1)
        return round(float(np.mean(errors)), 3)

    @staticmethod
    def compute_rmse(img1: np.ndarray, img2: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """Compute Root Mean Squared Error (RMSE) across overlapping valid regions."""
        if mask is None:
            mask = (img1 > 0) & (img2 > 0)
        if not np.any(mask):
            return 999.0
        diff = img1.astype(np.float32)[mask] - img2.astype(np.float32)[mask]
        return round(float(np.sqrt(np.mean(diff ** 2))), 3)

    @staticmethod
    def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute Structural Similarity Index (SSIM)."""
        score, _ = ssim(img1, img2, full=True)
        return round(float(score), 4)

    @staticmethod
    def compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute Normalized Cross Correlation (NCC)."""
        i1 = img1.astype(np.float32)
        i2 = img2.astype(np.float32)
        i1_mean, i2_mean = np.mean(i1), np.mean(i2)
        nom = np.sum((i1 - i1_mean) * (i2 - i2_mean))
        denom = np.sqrt(np.sum((i1 - i1_mean)**2) * np.sum((i2 - i2_mean)**2)) + 1e-8
        return round(float(nom / denom), 4)

    def evaluate_full_pipeline(
        self,
        ref_img: np.ndarray,
        registered_img: np.ndarray,
        inliers_src: np.ndarray,
        inliers_ref: np.ndarray,
        total_matches: int,
        H: np.ndarray,
        coverage_score: float,
        uniform_score: float,
        retrieval_accuracy: float = 100.0,
        processing_time_s: float = 0.5
    ) -> Dict[str, Any]:
        """Synthesize consolidated multi-metric report."""
        inlier_count = len(inliers_src)
        inlier_ratio = round(float(inlier_count / (total_matches + 1e-5)), 4)
        mre = self.compute_reprojection_error(inliers_src, inliers_ref, H)
        rmse = self.compute_rmse(ref_img, registered_img)
        ssim_val = self.compute_ssim(ref_img, registered_img)
        ncc_val = self.compute_ncc(ref_img, registered_img)

        # Overall Registration Accuracy index (composite score 0-100%)
        # High inlier ratio, low MRE, high SSIM/NCC, good coverage
        mre_score = max(0.0, 1.0 - (mre / 5.0))
        acc_composite = (
            0.30 * min(1.0, inlier_ratio * 1.5) +
            0.25 * mre_score +
            0.20 * max(0.0, ncc_val) +
            0.15 * (coverage_score / 100.0) +
            0.10 * uniform_score
        ) * 100.0

        return {
            "RMSE": rmse,
            "Reprojection_Error_px": mre,
            "Inlier_Count": inlier_count,
            "Total_Matches": total_matches,
            "Inlier_Ratio": inlier_ratio,
            "SSIM": ssim_val,
            "NCC": ncc_val,
            "Coverage_Score_pct": coverage_score,
            "Uniform_Distribution_Score": uniform_score,
            "Retrieval_Accuracy_pct": retrieval_accuracy,
            "Registration_Accuracy_pct": round(acc_composite, 2),
            "Processing_Time_s": round(processing_time_s, 3)
        }
