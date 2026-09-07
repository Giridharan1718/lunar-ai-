"""
Spatial Grid Uniform Keypoint Distribution Engine
Partitions image into grid cells and selects high-confidence points per cell
to prevent spatial clustering (e.g. over-clustering around single crater rims).
"""

from typing import Dict, Any, Tuple
import numpy as np


class UniformGridDistributor:
    """Enforces spatial homogeneity of keypoint distributions across lunar surface tiles."""

    def __init__(
        self,
        grid_rows: int = 8,
        grid_cols: int = 8,
        max_points_per_cell: int = 25,
        min_distance_px: float = 8.0
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_points_per_cell = max_points_per_cell
        self.min_distance_px = min_distance_px

    def distribute(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        descriptors: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Subsample keypoints to ensure uniform grid representation.
        """
        h, w = image_shape[:2]
        if len(keypoints) == 0:
            return {
                "keypoints": keypoints,
                "scores": scores,
                "descriptors": descriptors,
                "coverage_score": 0.0,
                "uniform_score": 0.0,
                "grid_occupancy": 0.0,
                "occupied_cells": 0,
                "total_cells": self.grid_rows * self.grid_cols
            }

        cell_h = h / float(self.grid_rows)
        cell_w = w / float(self.grid_cols)

        # Assign each point to a grid cell
        grid_indices = [[] for _ in range(self.grid_rows * self.grid_cols)]
        for idx, (x, y) in enumerate(keypoints):
            r = min(int(y / cell_h), self.grid_rows - 1)
            c = min(int(x / cell_w), self.grid_cols - 1)
            cell_id = r * self.grid_cols + c
            grid_indices[cell_id].append(idx)

        selected_indices = []
        cell_counts = []

        for cell_id, p_indices in enumerate(grid_indices):
            if not p_indices:
                cell_counts.append(0)
                continue

            # Sort points in this cell by descending score
            p_indices_sorted = sorted(p_indices, key=lambda i: scores[i], reverse=True)

            # Spatial distance suppression inside cell
            cell_selected = []
            for candidate_idx in p_indices_sorted:
                if len(cell_selected) >= self.max_points_per_cell:
                    break
                cand_pt = keypoints[candidate_idx]
                too_close = False
                for sel_idx in cell_selected:
                    sel_pt = keypoints[sel_idx]
                    dist = np.hypot(cand_pt[0] - sel_pt[0], cand_pt[1] - sel_pt[1])
                    if dist < self.min_distance_px:
                        too_close = True
                        break
                if not too_close:
                    cell_selected.append(candidate_idx)

            selected_indices.extend(cell_selected)
            cell_counts.append(len(cell_selected))

        selected_indices = np.array(selected_indices, dtype=np.int64)

        # Compute Metrics
        total_cells = self.grid_rows * self.grid_cols
        occupied_cells = sum(1 for count in cell_counts if count > 0)
        grid_occupancy = float(occupied_cells / total_cells)

        # Coverage score: percentage of active cells
        coverage_score = round(grid_occupancy * 100.0, 2)

        # Uniform Distribution Score: Inverse of coefficient of variation of cell counts
        non_empty = [c for c in cell_counts if c > 0]
        if len(non_empty) > 1 and np.mean(cell_counts) > 0:
            cv_val = np.std(cell_counts) / (np.mean(cell_counts) + 1e-5)
            # Higher is more uniform (1.0 = perfectly uniform)
            uniform_score = round(float(np.clip(1.0 / (1.0 + cv_val), 0.0, 1.0)), 3)
        else:
            uniform_score = 1.0 if occupied_cells > 0 else 0.0

        return {
            "keypoints": keypoints[selected_indices] if len(selected_indices) > 0 else np.empty((0, 2), dtype=np.float32),
            "scores": scores[selected_indices] if len(selected_indices) > 0 else np.empty((0,), dtype=np.float32),
            "descriptors": descriptors[selected_indices] if len(selected_indices) > 0 else np.empty((0, 256), dtype=np.float32),
            "coverage_score": coverage_score,
            "uniform_score": uniform_score,
            "grid_occupancy": round(grid_occupancy, 3),
            "occupied_cells": occupied_cells,
            "total_cells": total_cells,
            "cell_counts": cell_counts
        }
