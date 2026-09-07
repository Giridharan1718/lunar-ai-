"""
LightGlue Deep Transformer Feature Matcher
Matches SuperPoint features using positional encodings, self-attention,
and cross-attention layers to resolve extreme lunar photometric/viewpoint differences.
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch


class LightGlueMatcher:
    """LightGlue deep matcher with fallback to Mutual Nearest Neighbors (MNN) + Lowe's Ratio."""

    def __init__(
        self,
        filter_threshold: float = 0.15,
        depth_confidence: float = 0.95,
        width_confidence: float = 0.99,
        device: str = "cpu"
    ):
        self.filter_threshold = filter_threshold
        self.depth_confidence = depth_confidence
        self.width_confidence = width_confidence
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from lightglue import LightGlue
            self.model = LightGlue(
                features="superpoint",
                filter_threshold=self.filter_threshold,
                depth_confidence=self.depth_confidence,
                width_confidence=self.width_confidence
            ).eval().to(self.device)
        except Exception:
            self.model = None

    def match(
        self,
        feats0: Dict[str, np.ndarray],
        feats1: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Match keypoints between feats0 (source) and feats1 (reference).
        """
        kpts0, desc0 = feats0["keypoints"], feats0["descriptors"]
        kpts1, desc1 = feats1["keypoints"], feats1["descriptors"]

        if len(kpts0) == 0 or len(kpts1) == 0:
            return {
                "matches": np.empty((0, 2), dtype=np.int64),
                "matched_kpts0": np.empty((0, 2), dtype=np.float32),
                "matched_kpts1": np.empty((0, 2), dtype=np.float32),
                "confidences": np.empty((0,), dtype=np.float32),
                "total_matches": 0,
                "average_confidence": 0.0,
                "match_density": 0.0
            }

        if self.model is not None:
            try:
                data = {
                    "image0": {
                        "keypoints": torch.from_numpy(kpts0).float().unsqueeze(0).to(self.device),
                        "descriptors": torch.from_numpy(desc0).float().unsqueeze(0).to(self.device),
                        "image_size": torch.tensor([feats0.get("image_size", (1024, 1024))]).to(self.device)
                    },
                    "image1": {
                        "keypoints": torch.from_numpy(kpts1).float().unsqueeze(0).to(self.device),
                        "descriptors": torch.from_numpy(desc1).float().unsqueeze(0).to(self.device),
                        "image_size": torch.tensor([feats1.get("image_size", (1024, 1024))]).to(self.device)
                    }
                }
                with torch.no_grad():
                    out = self.model(data)

                matches = out["matches"][0].cpu().numpy()
                scores = out["scores"][0].cpu().numpy()

                mkpts0 = kpts0[matches[:, 0]]
                mkpts1 = kpts1[matches[:, 1]]

                avg_conf = float(np.mean(scores)) if len(scores) > 0 else 0.0
                density = float(len(matches) / min(len(kpts0), len(kpts1))) if min(len(kpts0), len(kpts1)) > 0 else 0.0

                return {
                    "matches": matches,
                    "matched_kpts0": mkpts0,
                    "matched_kpts1": mkpts1,
                    "confidences": scores,
                    "total_matches": len(matches),
                    "average_confidence": round(avg_conf, 3),
                    "match_density": round(density, 3)
                }
            except Exception:
                pass

        # Fallback: Mutual Cosine Nearest-Neighbor matching + Second-best ratio test
        sim_matrix = np.dot(desc0, desc1.T) # [N0, N1] Cosine similarity
        best_0_to_1 = np.argmax(sim_matrix, axis=1)
        best_1_to_0 = np.argmax(sim_matrix, axis=0)

        matches = []
        confidences = []

        for i, j in enumerate(best_0_to_1):
            if best_1_to_0[j] == i: # Mutual check
                score = sim_matrix[i, j]
                # Ratio test against second highest in row
                row_sorted = np.sort(sim_matrix[i, :])
                if len(row_sorted) >= 2:
                    ratio = (1.0 - score) / (1.0 - row_sorted[-2] + 1e-5)
                    if ratio < 0.85 and score > 0.4:
                        matches.append([i, j])
                        confidences.append(float(score))
                elif score > 0.4:
                    matches.append([i, j])
                    confidences.append(float(score))

        matches_arr = np.array(matches, dtype=np.int64) if matches else np.empty((0, 2), dtype=np.int64)
        conf_arr = np.array(confidences, dtype=np.float32) if confidences else np.empty((0,), dtype=np.float32)

        mkpts0 = kpts0[matches_arr[:, 0]] if len(matches_arr) > 0 else np.empty((0, 2), dtype=np.float32)
        mkpts1 = kpts1[matches_arr[:, 1]] if len(matches_arr) > 0 else np.empty((0, 2), dtype=np.float32)

        avg_conf = float(np.mean(conf_arr)) if len(conf_arr) > 0 else 0.0
        density = float(len(matches_arr) / min(len(kpts0), len(kpts1))) if min(len(kpts0), len(kpts1)) > 0 else 0.0

        return {
            "matches": matches_arr,
            "matched_kpts0": mkpts0,
            "matched_kpts1": mkpts1,
            "confidences": conf_arr,
            "total_matches": len(matches_arr),
            "average_confidence": round(avg_conf, 3),
            "match_density": round(density, 3)
        }
