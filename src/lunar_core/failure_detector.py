"""
Automated Failure & Anomaly Detection System
Monitors registration health using multi-variable thresholding and generates triage reports (Success, Warning, Failure).
"""

from typing import Dict, Any, List


class FailureDetector:
    """Classifies registration outcomes into Success, Warning, or Failure based on mission safety criteria."""

    def __init__(
        self,
        min_inlier_ratio: float = 0.30,
        min_inlier_count: int = 15,
        max_reprojection_error_px: float = 4.0,
        min_coverage_score: float = 40.0,
        min_confidence: float = 0.40
    ):
        self.min_inlier_ratio = min_inlier_ratio
        self.min_inlier_count = min_inlier_count
        self.max_reprojection_error_px = max_reprojection_error_px
        self.min_coverage_score = min_coverage_score
        self.min_confidence = min_confidence

    def diagnose(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run rule-based anomaly detector on computed registration metrics.
        """
        reproj = metrics.get("Reprojection_Error_px", 999.0)
        inlier_ratio = metrics.get("Inlier_Ratio", 0.0)
        inlier_count = metrics.get("Inlier_Count", 0)
        coverage = metrics.get("Coverage_Score_pct", 0.0)
        conf = metrics.get("Average_Confidence", 0.6)

        flags = []
        severity = "SUCCESS"

        if inlier_count < self.min_inlier_count:
            flags.append(f"Insufficient inliers: {inlier_count} < {self.min_inlier_count}")
            severity = "FAILURE"
        elif inlier_ratio < self.min_inlier_ratio:
            flags.append(f"Low inlier ratio: {inlier_ratio:.2f} < {self.min_inlier_ratio}")
            severity = "WARNING" if severity != "FAILURE" else severity

        if reproj > self.max_reprojection_error_px:
            flags.append(f"High reprojection error: {reproj:.2f}px > {self.max_reprojection_error_px}px")
            severity = "FAILURE"

        if coverage < self.min_coverage_score:
            flags.append(f"Poor spatial coverage: {coverage:.1f}% < {self.min_coverage_score}%")
            severity = "WARNING" if severity == "SUCCESS" else severity

        if conf < self.min_confidence:
            flags.append(f"Low match confidence: {conf:.2f} < {self.min_confidence}")
            severity = "WARNING" if severity == "SUCCESS" else severity

        # Diagnostic recommendation
        if severity == "SUCCESS":
            recommendation = "Optimal registration. Safe for selenographic mapping, orthorectification, and mosaic blending."
        elif severity == "WARNING":
            recommendation = "Marginal alignment quality. Recommend ECC refinement pass or adaptive grid loosening."
        else:
            recommendation = "Registration failed. Candidate rejection triggered. Trigger fallback multi-scale LunaDNA global search."

        return {
            "status": severity,
            "anomaly_flags": flags,
            "anomaly_count": len(flags),
            "recommendation": recommendation,
            "confidence_level": "High" if severity == "SUCCESS" else ("Medium" if severity == "WARNING" else "Low")
        }
