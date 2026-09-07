"""
Configuration Loader & Pydantic Dataclass Definitions
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
try:
    import yaml
except ImportError:
    yaml = None
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    data_ground_truth: str = "data/ground_truth"
    data_reference: str = "data/reference"
    data_synthetic: str = "data/synthetic_samples"
    outputs_reports: str = "outputs/reports"
    outputs_visualizations: str = "outputs/visualizations"
    outputs_features: str = "outputs/features"
    outputs_matches: str = "outputs/matches"
    outputs_registered: str = "outputs/registered"
    outputs_lunadna: str = "outputs/lunadna"
    outputs_faiss: str = "outputs/faiss_index"
    outputs_mosaics: str = "outputs/mosaics"
    outputs_logs: str = "outputs/logs"


@dataclass
class HardwareConfig:
    device: str = "cuda"
    num_workers: int = 4
    precision: str = "float32"
    seed: int = 42


@dataclass
class LunarConfig:
    project_name: str = "Chandrayaan-2 Lunar Optical Image Registration"
    version: str = "1.0.0"
    paths: PathConfig = field(default_factory=PathConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    def ensure_directories(self):
        """Create all configured data and output directories if they do not exist."""
        path_dict = vars(self.paths)
        for _, rel_path in path_dict.items():
            Path(rel_path).mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> LunarConfig:
    """Load configuration from YAML file or return default configuration."""
    if config_path is None:
        possible_paths = [
            "config/default_config.yaml",
            "../config/default_config.yaml",
            "../../config/default_config.yaml"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                config_path = p
                break

    if config_path and os.path.exists(config_path) and yaml is not None:
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
    else:
        data = {}

    paths_data = data.get("paths", {})
    hardware_data = data.get("hardware", {})

    paths_cfg = PathConfig(**{k: v for k, v in paths_data.items() if hasattr(PathConfig, k)})
    hardware_cfg = HardwareConfig(**{k: v for k, v in hardware_data.items() if hasattr(HardwareConfig, k)})

    cfg = LunarConfig(
        project_name=data.get("project", {}).get("name", "Chandrayaan-2 Lunar Optical Image Registration"),
        version=data.get("project", {}).get("version", "1.0.0"),
        paths=paths_cfg,
        hardware=hardware_cfg,
        raw_dict=data
    )
    cfg.ensure_directories()
    return cfg
