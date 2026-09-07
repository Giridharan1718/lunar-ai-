"""
Streamlit Web Application: ISRO Chandrayaan-2 Lunar Optical Image Registration Suite
Provides an interactive multi-stage GUI for dataset analytics, terrain feature inspection,
SuperPoint+LightGlue deep matching, MAGSAC++ filtering, Sub-pixel ECC warping, and Mosaic building.
"""

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# Import Lunar Core Modules
from lunar_core.config import load_config
from lunar_core.data_loader import LunarDatasetScanner, LunarImageLoader
from lunar_core.terrain import LunarTerrainAnalyzer
from lunar_core.illumination import IlluminationAnalyzer
from lunar_core.lunadna import LunaDNAGenerator
from lunar_core.retrieval import FAISSRetrievalEngine
from lunar_core.features import SuperPointExtractor
from lunar_core.distribution import UniformGridDistributor
from lunar_core.matching import LightGlueMatcher
from lunar_core.magsac import MAGSACFilter
from lunar_core.registration import RegistrationEngine
from lunar_core.ecc import ECCRefinementEngine
from lunar_core.evaluation import RegistrationEvaluator
from lunar_core.failure_detector import FailureDetector
from lunar_core.mosaic import LunarMosaicBuilder
from lunar_core.synthetic_data import LunarSyntheticGenerator

st.set_page_config(
    page_title="ISRO Chandrayaan-2 Lunar Image Registration",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# App Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg", width=110)
st.sidebar.title("🌙 Navigation")
pipeline_stage = st.sidebar.radio(
    "Select Mission Stage:",
    [
        "1. Dataset Scanner & Analytics",
        "2. Terrain Geomorphology",
        "3. Illumination Variation",
        "4. LunaDNA & FAISS Retrieval",
        "5. Deep Feature Matching (SuperPoint + LightGlue)",
        "6. MAGSAC++ & Geometric Registration",
        "7. Sub-Pixel ECC Refinement",
        "8. Evaluation & Anomaly Triage",
        "9. Global Lunar Mosaic Builder"
    ]
)

config = load_config()

# Pre-populate sample datasets if data/ is empty
gen = LunarSyntheticGenerator(size=(512, 512), seed=42)
if not list(Path("data").glob("*/*.png")):
    with st.spinner("Initializing sample Chandrayaan-2 & LROC datasets..."):
        gen.populate_sample_datasets("data")

st.markdown("<div class='main-header'>ISRO Chandrayaan-2 Lunar Image Registration System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Sub-Pixel Optical Frame Alignment, LunaDNA Topological Retrieval & MAGSAC++ Consensus Pipeline</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# STAGE 1: Dataset Scanner
# -------------------------------------------------------------
if pipeline_stage == "1. Dataset Scanner & Analytics":
    st.header("Stage 01: Dataset Scan & Quality Metrics")
    scanner = LunarDatasetScanner(["data/ohrc", "data/tmc", "data/iirc", "data/quickmap-lroc", "data/reference"])
    df, summary = scanner.scan()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Scanned Images", summary["total_images_scanned"])
    col2.metric("Valid Frames", summary["valid_images_count"])
    col3.metric("Duplicate Count", summary["duplicate_images_count"])
    col4.metric("Mean Resolution", summary["mean_resolution"])

    st.subheader("Image Repository Inventory")
    st.dataframe(df[["file_name", "folder", "extension", "file_size_kb", "width", "height", "mean_brightness", "contrast_ratio"]], use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_b = px.histogram(df, x="mean_brightness", nbins=20, title="Dataset Mean Brightness Distribution", color_discrete_sequence=["#2563EB"])
        st.plotly_chart(fig_b, use_container_width=True)
    with c2:
        fig_c = px.histogram(df, x="contrast_ratio", nbins=20, title="RMS Contrast Ratio Distribution", color_discrete_sequence=["#10B981"])
        st.plotly_chart(fig_c, use_container_width=True)

# -------------------------------------------------------------
# STAGE 2: Terrain Analysis
# -------------------------------------------------------------
elif pipeline_stage == "2. Terrain Geomorphology":
    st.header("Stage 02: Lunar Surface Geomorphology & Crater Detection")
    sample_paths = list(Path("data").glob("*/*.png"))
    if sample_paths:
        selected_file = st.selectbox("Select Lunar Image Tile:", [str(p) for p in sample_paths])
        img = LunarImageLoader.load_image(selected_file)

        analyzer = LunarTerrainAnalyzer()
        res = analyzer.analyze(img)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detected Craters", res["num_craters_detected"])
        c2.metric("Crater Spatial Density", f"{res['crater_density']}/km²")
        c3.metric("Shadow Coverage", f"{res['shadow_percentage']}%")
        c4.metric("Texture Complexity", res["texture_complexity"])

        col_a, col_b, col_c = st.columns(3)
        # Annotated Craters
        img_annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for cx, cy, r in res["craters_list"]:
            cv2.circle(img_annotated, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(img_annotated, (cx, cy), 2, (0, 0, 255), -1)

        col_a.image(img_annotated, caption="Crater Boundary Extractions", use_container_width=True)
        col_b.image(res["crater_density_map"], caption="Crater Spatial Density Heatmap", use_container_width=True)
        col_c.image(res["shadow_mask"], caption="Binary Shadow Segmentations", use_container_width=True)

# -------------------------------------------------------------
# STAGE 3: Illumination Analysis
# -------------------------------------------------------------
elif pipeline_stage == "3. Illumination Variation":
    st.header("Stage 03: Photometric & Solar Incidence Analysis")
    sample_paths = list(Path("data").glob("*/*.png"))
    if sample_paths:
        selected_file = st.selectbox("Select Lunar Image Tile:", [str(p) for p in sample_paths])
        img = LunarImageLoader.load_image(selected_file)

        illum = IlluminationAnalyzer()
        metrics = illum.compute_illumination_metrics(img)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Illumination Class", metrics["classification"])
        c2.metric("Mean Luminance", metrics["mean_brightness"])
        c3.metric("RMS Contrast", metrics["rms_contrast"])
        c4.metric("Shadow Coverage", f"{metrics['shadow_coverage_pct']}%")

        col1, col2 = st.columns(2)
        col1.image(img, caption="Raw Lunar Frame", use_container_width=True)
        col2.image(metrics["illumination_heatmap"], caption="Photometric Solar Gradient Heatmap", use_container_width=True)

# -------------------------------------------------------------
# STAGE 4: LunaDNA & FAISS Retrieval
# -------------------------------------------------------------
elif pipeline_stage == "4. LunaDNA & FAISS Retrieval":
    st.header("Stage 04 & 05: LunaDNA 256-D Descriptor & FAISS Vector Indexing")
    st.write("Extract topological crater geometry, directional gradients, and spatial pyramid moments into fixed 256-D vectors.")

    generator = LunaDNAGenerator(vector_dim=256)
    sample_paths = list(Path("data").glob("*/*.png"))[:10]

    vectors = []
    names = []
    for p in sample_paths:
        im = LunarImageLoader.load_image(str(p))
        if im is not None:
            vec = generator.generate_fingerprint(im)
            vectors.append(vec)
            names.append(p.name)

    vectors_arr = np.array(vectors, dtype=np.float32)
    faiss_engine = FAISSRetrievalEngine(dimension=256)
    faiss_engine.build_index(vectors_arr, names)

    st.success(f"Built FAISS Index with {len(names)} Lunar Surface Reference Signatures.")

    query_idx = st.selectbox("Select Query Tile:", range(len(names)), format_func=lambda i: names[i])
    q_vec = vectors_arr[query_idx]

    matched_ids, scores, latency = faiss_engine.query(q_vec, top_k=5)

    st.metric("Query Latency", f"{latency:.3f} ms")
    st.subheader("Top-K Retrieved Reference Tiles")

    cols = st.columns(len(matched_ids))
    for idx, col in enumerate(cols):
        col.write(f"**Rank {idx+1}:** {matched_ids[idx]}")
        col.write(f"**Cosine Sim:** {scores[idx]:.4f}")

# -------------------------------------------------------------
# STAGE 5: Deep Feature Matching
# -------------------------------------------------------------
elif pipeline_stage == "5. Deep Feature Matching (SuperPoint + LightGlue)":
    st.header("Stage 07 & 09: SuperPoint Deep Keypoints & LightGlue Transformer Matching")

    pair = gen.generate_registered_pair(rotation_deg=15.0, scale=1.05, tx=35.0, ty=-20.0)
    ref_img, src_img = pair["reference_image"], pair["source_image"]

    sp = SuperPointExtractor(max_keypoints=1024)
    distr = UniformGridDistributor(grid_rows=8, grid_cols=8, max_points_per_cell=20)
    matcher = LightGlueMatcher()

    with st.spinner("Extracting SuperPoint Keypoints & Running Uniform Spatial Grid Distribution..."):
        raw_feats_ref = sp.extract(ref_img)
        raw_feats_src = sp.extract(src_img)

        feats_ref = distr.distribute(raw_feats_ref["keypoints"], raw_feats_ref["scores"], raw_feats_ref["descriptors"], ref_img.shape)
        feats_src = distr.distribute(raw_feats_src["keypoints"], raw_feats_src["scores"], raw_feats_src["descriptors"], src_img.shape)

        match_res = matcher.match(feats_src, feats_ref)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ref Keypoints", len(feats_ref["keypoints"]))
    c2.metric("Src Keypoints", len(feats_src["keypoints"]))
    c3.metric("Total Matches", match_res["total_matches"])
    c4.metric("Avg Confidence", match_res["average_confidence"])

    # Draw Matches Side-by-Side
    h = max(ref_img.shape[0], src_img.shape[0])
    w = ref_img.shape[1] + src_img.shape[1]
    match_vis = np.zeros((h, w, 3), dtype=np.uint8)
    match_vis[:src_img.shape[0], :src_img.shape[1]] = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
    match_vis[:ref_img.shape[0], src_img.shape[1]:] = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)

    offset_x = src_img.shape[1]
    for pt_src, pt_ref in zip(match_res["matched_kpts0"][:150], match_res["matched_kpts1"][:150]):
        p1 = (int(pt_src[0]), int(pt_src[1]))
        p2 = (int(pt_ref[0] + offset_x), int(pt_ref[1]))
        cv2.line(match_vis, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(match_vis, p1, 3, (0, 0, 255), -1)
        cv2.circle(match_vis, p2, 3, (255, 0, 0), -1)

    st.image(match_vis, caption="LightGlue Transformer Correspondence Points", use_container_width=True)

# -------------------------------------------------------------
# STAGE 6: MAGSAC++ & Geometric Registration
# -------------------------------------------------------------
elif pipeline_stage == "6. MAGSAC++ & Geometric Registration":
    st.header("Stage 10 & 11: MAGSAC++ Outlier Filtering & Homography Warping")

    pair = gen.generate_registered_pair(rotation_deg=12.0, scale=1.06, tx=30.0, ty=-25.0)
    ref_img, src_img = pair["reference_image"], pair["source_image"]

    sp = SuperPointExtractor(max_keypoints=1024)
    matcher = LightGlueMatcher()
    magsac = MAGSACFilter(threshold_px=3.0)
    reg_engine = RegistrationEngine()

    feats_ref = sp.extract(ref_img)
    feats_src = sp.extract(src_img)
    match_res = matcher.match(feats_src, feats_ref)

    filter_res = magsac.filter(match_res["matched_kpts0"], match_res["matched_kpts1"])
    reg_res = reg_engine.register(src_img, ref_img, filter_res["matrix"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inlier Count", filter_res["inlier_count"])
    c2.metric("Inlier Ratio", f"{filter_res['inlier_ratio']*100:.1f}%")
    c3.metric("Estimated Rotation", f"{reg_res['decomposition']['rotation_deg']}°")
    c4.metric("Estimated Scale", f"{reg_res['decomposition']['scale_mean']}")

    col1, col2, col3 = st.columns(3)
    col1.image(reg_res["registered_image"], caption="Warped Registered Frame", use_container_width=True)
    col2.image(reg_res["checkerboard"], caption="Checkerboard Verification", use_container_width=True)
    col3.image(reg_res["difference_map"], caption="Alignment Difference Residuals", use_container_width=True)

# -------------------------------------------------------------
# STAGE 7: ECC Refinement
# -------------------------------------------------------------
elif pipeline_stage == "7. Sub-Pixel ECC Refinement":
    st.header("Stage 12: Enhanced Correlation Coefficient (ECC) Maximization")
    pair = gen.generate_registered_pair(rotation_deg=8.0, scale=1.02, tx=15.0, ty=-10.0)
    ref_img, src_img = pair["reference_image"], pair["source_image"]

    sp = SuperPointExtractor(max_keypoints=1024)
    matcher = LightGlueMatcher()
    magsac = MAGSACFilter(threshold_px=3.0)
    reg_engine = RegistrationEngine()
    ecc_engine = ECCRefinementEngine(max_iterations=100)

    feats_ref = sp.extract(ref_img)
    feats_src = sp.extract(src_img)
    match_res = matcher.match(feats_src, feats_ref)
    filter_res = magsac.filter(match_res["matched_kpts0"], match_res["matched_kpts1"])
    reg_res = reg_engine.register(src_img, ref_img, filter_res["matrix"])

    ecc_res = ecc_engine.refine(ref_img, reg_res["registered_image"], filter_res["matrix"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ECC Convergence", "Converged" if ecc_res["success"] else "Fallback")
    c2.metric("ECC Correlation Score", ecc_res["ecc_score"])
    c3.metric("Initial Mean Abs Error", ecc_res["initial_alignment_error"])
    c4.metric("Pixel Error Improvement", f"{ecc_res['pixel_improvement_pct']}%")

    col1, col2 = st.columns(2)
    col1.image(reg_res["registered_image"], caption="Coarse Registered Image (MAGSAC++)", use_container_width=True)
    col2.image(ecc_res["refined_image"], caption="Sub-Pixel Refined Image (ECC Maximized)", use_container_width=True)

# -------------------------------------------------------------
# STAGE 8: Evaluation & Anomaly Triage
# -------------------------------------------------------------
elif pipeline_stage == "8. Evaluation & Anomaly Triage":
    st.header("Stage 13 & 14: Comprehensive Evaluation & Anomaly Triage")
    evaluator = RegistrationEvaluator()
    detector = FailureDetector()

    pair = gen.generate_registered_pair(rotation_deg=10.0, scale=1.04, tx=20.0, ty=-15.0)
    ref_img, src_img = pair["reference_image"], pair["source_image"]

    sp = SuperPointExtractor(max_keypoints=1024)
    distr = UniformGridDistributor()
    matcher = LightGlueMatcher()
    magsac = MAGSACFilter()
    reg_engine = RegistrationEngine()

    f_ref = distr.distribute(*list(sp.extract(ref_img).values())[:3], ref_img.shape)
    f_src = distr.distribute(*list(sp.extract(src_img).values())[:3], src_img.shape)
    m_res = matcher.match(f_src, f_ref)
    filt = magsac.filter(m_res["matched_kpts0"], m_res["matched_kpts1"])
    reg = reg_engine.register(src_img, ref_img, filt["matrix"])

    eval_res = evaluator.evaluate_full_pipeline(
        ref_img=ref_img,
        registered_img=reg["registered_image"],
        inliers_src=filt["inliers_src"],
        inliers_ref=filt["inliers_ref"],
        total_matches=m_res["total_matches"],
        H=filt["matrix"],
        coverage_score=f_ref["coverage_score"],
        uniform_score=f_ref["uniform_score"]
    )
    eval_res["Average_Confidence"] = m_res["average_confidence"]

    diagnosis = detector.diagnose(eval_res)

    # Status Banner
    if diagnosis["status"] == "SUCCESS":
        st.success(f"STATUS: SUCCESS — {diagnosis['recommendation']}")
    elif diagnosis["status"] == "WARNING":
        st.warning(f"STATUS: WARNING — {diagnosis['recommendation']}")
    else:
        st.error(f"STATUS: FAILURE — {diagnosis['recommendation']}")

    st.subheader("Mission Evaluation Scorecard")
    st.json(eval_res)

# -------------------------------------------------------------
# STAGE 9: Global Lunar Mosaic Builder
# -------------------------------------------------------------
elif pipeline_stage == "9. Global Lunar Mosaic Builder":
    st.header("Stage 15: Global Lunar Mosaic & Seam Multi-Band Blending")
    builder = LunarMosaicBuilder(blend_mode="feather", feather_width=40)

    ref_img = gen.generate_lunar_surface(num_craters=40, seed=10)
    pair1 = gen.generate_registered_pair(rotation_deg=5.0, scale=1.0, tx=80.0, ty=30.0)
    pair2 = gen.generate_registered_pair(rotation_deg=-6.0, scale=1.0, tx=-70.0, ty=-40.0)

    pairs = [
        (pair1["source_image"], pair1["homography_ground_truth"]),
        (pair2["source_image"], pair2["homography_ground_truth"])
    ]

    with st.spinner("Stitching Lunar Mosaic with Feather Boundary Blending..."):
        mosaic_res = builder.build_mosaic(ref_img, pairs, canvas_scale=1.4)

    c1, c2, c3 = st.columns(3)
    c1.metric("Mosaic Dimensions", f"{mosaic_res['dimensions'][0]}x{mosaic_res['dimensions'][1]}")
    c2.metric("Effective Coverage", f"{mosaic_res['coverage_pct']}%")
    c3.metric("Stitched Tile Count", mosaic_res["total_tiles_stitched"])

    st.image(mosaic_res["mosaic_image"], caption="Chandrayaan-2 High-Resolution Composite Lunar Mosaic", use_container_width=True)
