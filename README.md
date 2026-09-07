# 🌙 ISRO Chandrayaan-2 Lunar Optical Image Registration System

An enterprise-grade, high-precision remote sensing and computer vision framework designed to register **Chandrayaan-2 optical imagery** (OHRC, TMC-2, IIR) against global lunar reference basemaps (LROC QuickMap, USGS Lunar Reconnaissance Orbiter DEMs) under extreme illumination variations, scale differences, and steep viewing angles.

---

## 🚀 Key Architectural Highlights

- **Sub-Pixel Precision**: Multi-stage coarse-to-fine registration combining **SuperPoint** deep feature detection, **Uniform Spatial Grid Regularization**, **LightGlue** transformer matching, **MAGSAC++** robust marginalizing consensus, and **ECC** (Enhanced Correlation Coefficient) optimization ($< 0.1\text{ px}$ residual error).
- **LunaDNA Global Indexing**: 256-dimensional topological lunar surface fingerprints capturing crater spatial morphology, multi-scale Laplacian responses, directional gradient histograms, and 2D FFT radial roughness spectra.
- **FAISS Candidate Retrieval**: Sub-millisecond similarity search across millions of reference tiles using Inner Product / Cosine vector indexing.
- **Automated Failure & Anomaly Triage**: Real-time detection of low inlier ratios, high reprojection residuals, degenerate homographies, and insufficient spatial coverage.
- **Multi-Band Lunar Mosaic Builder**: Distance-weighted feather blending for wide-field seamless cartographic composition.
- **Interactive Streamlit Suite**: Web GUI for mission planning, interactive visual inspection, and telemetry audits.

---

## 📂 Project Structure

```
.
├── config/                          # Central YAML Pipeline Configurations
│   ├── default_config.yaml          # Global pipeline hyperparameters
│   ├── superpoint_lightglue.yaml    # Deep matcher configurations
│   ├── faiss_retrieval.yaml         # Dense vector index specifications
│   └── logging_config.yaml          # Audit logging formatting
├── data/                            # Lunar Imagery Data Store
│   ├── raw/                         # Raw unprocessed mission data
│   ├── ohrc/                        # Orbiter High Resolution Camera (25 cm/pixel)
│   ├── tmc/                         # Terrain Mapping Camera-2 (5 m/pixel)
│   ├── iirc/                        # Imaging Infra-Red Spectrometer
│   ├── quickmap-lroc/               # LROC WAC / NAC Reference Tiles
│   ├── reference/                   # Georeferenced selenographic reference base
│   └── ground_truth/                # Ground truth tie-points and homographies
├── notebook/                        # 15 Interactive Jupyter Analysis Notebooks
│   ├── 01_Dataset_Analysis.ipynb
│   ├── 02_Terrain_Analysis.ipynb
│   ├── 03_Illumination_Analysis.ipynb
│   ├── 04_GroundTruth_Preparation.ipynb
│   ├── 05_LunaDNA_Generator.ipynb
│   ├── 06_FAISS_Retrieval.ipynb
│   ├── 07_SuperPoint_Feature_Extraction.ipynb
│   ├── 08_Uniform_Distribution.ipynb
│   ├── 09_LightGlue_Matching.ipynb
│   ├── 10_MAGSAC_Filtering.ipynb
│   ├── 11_Registration_Engine.ipynb
│   ├── 12_ECC_Refinement.ipynb
│   ├── 13_Evaluation_Engine.ipynb
│   ├── 14_Failure_Detection.ipynb
│   └── 15_Lunar_Mosaic_Builder.ipynb
├── outputs/                         # Pipeline Execution Artifacts & Reports
│   ├── faiss_index/                 # Serialized FAISS indices
│   ├── features/                    # Compressed NPZ keypoints & descriptors
│   ├── logs/                        # Rotating system audit logs
│   ├── lunadna/                     # LunaDNA vector databases (.npy, .csv)
│   ├── matches/                     # Inliers, outliers, and correspondence JSONs
│   ├── mosaics/                     # Blended high-resolution lunar maps
│   ├── registered/                  # Warped registered frames & matrices
│   ├── reports/                     # Audit CSVs, summaries, HTML & PDF reports
│   └── visualizations/              # 300-DPI high-resolution analytical plots
├── src/                             # Python Core Package
│   └── lunar_core/
│       ├── __init__.py
│       ├── config.py                # Configuration loader & validator
│       ├── logger.py                # Centralized logging setup
│       ├── data_loader.py           # Scanner, MD5 duplicate detector & loader
│       ├── terrain.py               # Crater, ridge, valley & GLCM texture analysis
│       ├── illumination.py          # Photometric classification & solar heatmaps
│       ├── lunadna.py               # 256-D topological fingerprint generator
│       ├── retrieval.py             # FAISS indexing & candidate search
│       ├── features.py              # SuperPoint deep keypoint extractor
│       ├── distribution.py          # Spatial grid uniform distributor
│       ├── matching.py              # LightGlue deep transformer matcher
│       ├── magsac.py                # MAGSAC++ robust outlier filter
│       ├── registration.py          # Projective homography warping engine
│       ├── ecc.py                   # Sub-pixel ECC photometric refinement
│       ├── evaluation.py            # Comprehensive benchmarking metrics
│       ├── failure_detector.py      # Automated mission anomaly diagnostics
│       ├── mosaic.py                # Multi-image feather seam blender
│       └── synthetic_data.py        # Photorealistic lunar terrain generator
├── app/
│   └── streamlit_app.py             # Interactive Mission Control Streamlit App
├── tools/
│   └── build_notebooks.py           # Automated Jupyter notebook compiler
├── environment.yml                  # Conda environment definition
├── requirements.txt                 # Pip requirements manifest
├── setup.py                         # Setuptools packaging script
├── pyproject.toml                   # Modern PEP-621 build configuration
└── .env.example                     # Environment variables template
```

---

## 🔄 End-to-End Processing Workflow

$$\begin{matrix}
\text{Dataset Analysis} & \rightarrow & \text{Terrain Geomorphology} & \rightarrow & \text{Illumination Analysis} \\
\downarrow & & & & \downarrow \\
\text{LunaDNA Extraction} & \rightarrow & \text{FAISS Top-K Search} & \rightarrow & \text{SuperPoint Features} \\
\downarrow & & & & \downarrow \\
\text{Uniform Grid Dist.} & \rightarrow & \text{LightGlue Matching} & \rightarrow & \text{MAGSAC++ Inliers} \\
\downarrow & & & & \downarrow \\
\text{Homography Warp} & \rightarrow & \text{ECC Sub-Pixel Refinement} & \rightarrow & \text{Anomaly Triage} \\
\downarrow & & & & \downarrow \\
\text{Evaluation Scorecard} & \rightarrow & \text{Multi-Band Mosaic} & \rightarrow & \text{Global Report}
\end{matrix}$$

---

## 🛠️ Environment Setup & Installation

### Option 1: Conda Environment
```bash
conda env create -f environment.yml
conda activate lunar_reg_env
pip install -e .
```

### Option 2: Standard Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 💻 Running the System

### 1. Launch Interactive Streamlit GUI:
```bash
streamlit run app/streamlit_app.py
```

### 2. Launch Jupyter Notebook Pipeline:
```bash
jupyter lab notebook/
```
Execute notebooks sequentially from `01_Dataset_Analysis.ipynb` to `15_Lunar_Mosaic_Builder.ipynb`.

---

## 📊 Benchmarking & Quality Metrics

The system calculates and exports the following metrics:
- **RMSE** (Root Mean Squared Error across valid overlapping pixels)
- **Mean Reprojection Error (MRE)** (Sub-pixel residual error in px)
- **Inlier Ratio** ($\frac{\text{MAGSAC++ Inliers}}{\text{Total Matches}}$)
- **SSIM** (Structural Similarity Index Measure)
- **NCC** (Normalized Cross Correlation)
- **Coverage Score** (% of active non-clustered spatial grid cells)
- **Uniform Distribution Score** (Normalized inverse coefficient of variation)
- **Top-1 / Top-5 / Top-10 Retrieval Accuracy** (FAISS candidate precision)
- **Sub-Pixel ECC Improvement** (% error reduction after photometric gradient ascent)

---

## 🛰️ Mission Heritage & Attribution
Developed for high-precision lunar cartography, landing site terrain mapping, and selenographic orthorectification of ISRO Chandrayaan-2 payloads.
