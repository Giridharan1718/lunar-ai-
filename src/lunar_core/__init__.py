"""
Lunar Core - ISRO Chandrayaan-2 High Precision Image Registration Package
"""

__version__ = "1.0.0"
__author__ = "ISRO Computer Vision & Remote Sensing Architecture Group"

from .config import load_config, LunarConfig
from .logger import setup_logger
from .data_loader import LunarDatasetScanner, LunarImageLoader
from .terrain import LunarTerrainAnalyzer
from .illumination import IlluminationAnalyzer
from .lunadna import LunaDNAGenerator
from .retrieval import FAISSRetrievalEngine
from .features import SuperPointExtractor
from .distribution import UniformGridDistributor
from .matching import LightGlueMatcher
from .magsac import MAGSACFilter
from .registration import RegistrationEngine
from .ecc import ECCRefinementEngine
from .evaluation import RegistrationEvaluator
from .failure_detector import FailureDetector
from .mosaic import LunarMosaicBuilder
from .synthetic_data import LunarSyntheticGenerator

__all__ = [
    "load_config",
    "LunarConfig",
    "setup_logger",
    "LunarDatasetScanner",
    "LunarImageLoader",
    "LunarTerrainAnalyzer",
    "IlluminationAnalyzer",
    "LunaDNAGenerator",
    "FAISSRetrievalEngine",
    "SuperPointExtractor",
    "UniformGridDistributor",
    "LightGlueMatcher",
    "MAGSACFilter",
    "RegistrationEngine",
    "ECCRefinementEngine",
    "RegistrationEvaluator",
    "FailureDetector",
    "LunarMosaicBuilder",
    "LunarSyntheticGenerator",
]
