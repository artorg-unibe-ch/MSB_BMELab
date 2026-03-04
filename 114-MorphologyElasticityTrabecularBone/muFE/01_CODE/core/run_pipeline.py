"""
BME Lab 114 — Morphology-Elasticity Relationship of Trabecular Bone
Full µFE pipeline: Preprocessing → Mesh → Simulation → Postprocessing

Author: Simone Poncioni, MSB
Date:   Spring Semester 2026

Usage:
    python run_pipeline.py
    # or call run_pipeline(paths, ds=4) directly from another script
"""

# %matplotlib inline  # remove if running outside Jupyter

import logging
import os
import subprocess
import sys
from pathlib import Path

import ccx2paraview
import itk
import meshio
import numpy as np
import pandas as pd

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

import ciclope
from core.segmentation import (
    crop_bbox,
    gaussian_smoothing_itk,
    keep_largest_component,
    morph_cleaning,
    otsu_threshold_itk_with_threshold,
)
from utils.img_io import read_image, write_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths that are shared across all samples
# ---------------------------------------------------------------------------
TEMPLATE_SIM = Path(
    "/home/bmelab/bmelabs/2026/testing/MSB_BMELab/"
    "114-MorphologyElasticityTrabecularBone/muFE/01_CODE/utils/template_sim.inp"
)
RESULTS_DIR = Path("../../02_RESULTS")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TSV = RESULTS_DIR / "results_muFE.tsv"

# Applied displacement as defined in template_sim.inp  ← was -0.01 in the TODO stub
DELTA_U = -0.1  # [mm]

OMP_THREADS = 8


# ===========================================================================
# Step 1 — Preprocessing  (Notebook 01)
# ===========================================================================
def preprocess(img_path: Path, outpath: Path, morph_diam: int = 3) -> Path:
    """
    Smooth → Otsu threshold → crop bbox → morphological cleaning → save.

    Returns
    -------
    write_path : Path
        Path to the saved segmented .mha file.
    """
    log.info("[1/4] Preprocessing %s", img_path.name)

    img_itk, _ = read_image(img_path)

    seg_smooth = gaussian_smoothing_itk(img_itk)
    seg_otsu, threshold = otsu_threshold_itk_with_threshold(seg_smooth)
    log.info("  Otsu threshold = %.1f", threshold)

    # Crop to bounding box with 10-voxel margin
    cropped, bbox = crop_bbox(seg_otsu, padding=10)

    # Morphological closing to fill small holes
    final_seg = morph_cleaning(cropped, morph_diam)

    write_path = outpath / f"{img_path.stem}_segmented.mha"
    write_image(final_seg, write_path)
    write_image(final_seg, write_path.with_suffix(".npy"))

    log.info("  Segmentation saved → %s", write_path)
    return write_path


# ===========================================================================
# Step 2 — Mesh generation  (Notebook 02)
# ===========================================================================
def generate_mesh(
    seg_path: Path,
    img_path: Path,
    ds: int = 4,
    voxel_size_mm: float = 0.0343994,
) -> tuple[meshio.Mesh, Path, Path]:
    """
    Downsample → keep largest component → voxel FE mesh → write VTK + INP.

    Returns
    -------
    mesh     : meshio.Mesh
    vtk_path : Path
    inp_path : Path
    """
    log.info("[2/4] Mesh generation (DS=%d)", ds)

    # NOTE: mesh is built from the *segmented* image, not the raw image
    bone, _ = read_image(seg_path)
    bone_ds = itk.shrink_image_filter(bone, shrink_factors=[ds, ds, ds])
    spacing = np.ones(3) * voxel_size_mm * ds

    bone_arr_full = itk.array_from_image(bone_ds)
    bone_arr = keep_largest_component(bone_arr_full)

    log.info("  Generating voxel mesh …")
    mesh = ciclope.core.voxelFE.vol2ugrid(bone_arr, spacing, verbose=True)

    # Use derived/ folder for all outputs (not the raw image parent)
    derived_dir = seg_path.parent
    stem = img_path.stem + "_segmented"

    vtk_path = derived_dir / f"{stem}.vtk"
    inp_path = derived_dir / f"{stem}.inp"

    mesh.write(vtk_path, binary=True)
    log.info("  VTK written → %s", vtk_path)

    ciclope.core.voxelFE.mesh2voxelfe(mesh, TEMPLATE_SIM, inp_path, verbose=True)
    log.info("  INP written → %s", inp_path)

    return mesh, vtk_path, inp_path


# ===========================================================================
# Step 3 — FE simulation  (Notebook 03)
# ===========================================================================
def run_simulation(inp_path: Path) -> bool:
    """
    Call the CalculiX solver on *inp_path* (without extension, as ccx expects).

    Returns True on success.
    """
    log.info("[3/4] Running CalculiX on %s", inp_path.name)

    # ccx expects the path *without* the .inp extension
    inp_no_ext = inp_path.with_suffix("")

    result = subprocess.run(
        f"export OMP_NUM_THREADS={OMP_THREADS} && ccx '{inp_no_ext}'",
        shell=True,
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.returncode != 0:
        log.error("CalculiX failed:\n%s", result.stderr)
        return False
    return True


# ===========================================================================
# Step 4 — Postprocessing  (Notebook 04)
# ===========================================================================
def _read_reaction_force(dat_path: Path) -> float:
    """
    Parse the CalculiX .dat file and return the total reaction force
    along the loading axis (Z = third RF component).

    The .dat file has a variable number of header/label rows, e.g.:

        forces (sum) for set NODES_Z0 and load case 1

         total          1.23456E+03  -4.56789E+01   5.36221E+03

    Strategy (most-to-least specific):
    1. Look for a line starting with "total" — CalculiX writes this when
       TOTALS=ONLY is set in the *NODE PRINT card.  Take its 4th token (RF3).
    2. Fallback: collect every line that has exactly 4 floats after skipping
       all header/label rows, and sum the last column (RF3).
    """
    text = dat_path.read_text(errors="replace")

    # --- Strategy 1: "total" header — values on same OR next line ---
    #
    # Variant A (older ccx / TOTALS=ONLY):
    #   total          1.23E+03  -4.56E+01   5.36E+03
    #
    # Variant B (newer ccx):
    #   total force (fx,fy,fz) for set NODES_Z0 and time  1.00E+00
    #          -3.06E-12  -1.37E-11   5.66E+02
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.strip().lower().startswith("total"):
            continue
        tokens = line.split()
        # Variant A: 4 tokens on the same line → ["total", RF1, RF2, RF3]
        if len(tokens) == 4:
            try:
                return float(tokens[3])
            except ValueError:
                pass
        # Variant B: values on the next non-empty line with exactly 3 tokens
        for next_line in lines[i + 1:]:
            next_tokens = next_line.split()
            if not next_tokens:
                continue
            if len(next_tokens) == 3:
                try:
                    return float(next_tokens[2])   # RF3 = Fz
                except ValueError:
                    pass
            break  # only check the immediately following data line

    # --- Strategy 2: accumulate per-node RF3 values ---
    # We only collect lines that look like:  <int>  <float>  <float>  <float>
    rf3_values = []
    in_force_section = False
    for line in text.splitlines():
        low = line.strip().lower()
        # Detect the reaction-force block header
        if "force" in low and "set" in low:
            in_force_section = True
            continue
        # Leave force section if we hit another section header
        if in_force_section and low and not low[0].isdigit() and not low.startswith("-"):
            in_force_section = False
        if not in_force_section:
            continue
        tokens = line.split()
        if len(tokens) != 4:
            continue
        try:
            # first token is node id (int), rest are RF1 RF2 RF3
            int(tokens[0])
            rf3_values.append(float(tokens[3]))
        except ValueError:
            continue  # skip label rows like "NODE RF1 RF2 RF3"

    if rf3_values:
        return float(np.sum(rf3_values))

    raise ValueError(
        f"Could not parse reaction force from {dat_path}.\n"
        f"File contents:\n{text[:800]}"
    )


def postprocess(
    vtk_path: Path,
    seg_npy_path: Path,
    sample_id: str,
) -> dict:
    """
    Read FRD results → compute E_app and BV/TV → return summary dict.

    Bug fixes applied vs. original notebook:
    - A now computed from mesh bounding box (was hardcoded as 14, the diameter)
    - delta_u = -0.1 mm  (matches template; notebook TODO stub said -0.01)
    """
    log.info("[4/4] Postprocessing %s", sample_id)

    frd_path = vtk_path.with_suffix(".frd")
    dat_path = vtk_path.with_suffix(".dat")

    # Convert CalculiX output to VTK
    ccx2paraview.Converter(str(frd_path.resolve()), ["vtk"]).run()

    # ------------------------------------------------------------------
    # Reaction force from .dat file
    # ------------------------------------------------------------------
    # CalculiX .dat files contain section headers and label rows, e.g.:
    #
    #   forces (sum) for set NODES_Z0 and load case 1
    #
    #    total          1.23E+03   -4.56E+01    5.36E+03
    #
    # We look for the "total" summary line and take the Z component (col 3).
    # Fallback: collect every numeric 4-column line after the RF section
    # header and sum the last column.
    F = _read_reaction_force(dat_path)
    log.info("  Reaction force F = %.4f N", F)

    # ------------------------------------------------------------------
    # Geometry from mesh bounding box
    # ------------------------------------------------------------------
    mesh = meshio.read(vtk_path)
    pts = mesh.points                          # (N, 3)  X Y Z columns

    x_range = pts[:, 0].max() - pts[:, 0].min()   # [mm]
    y_range = pts[:, 1].max() - pts[:, 1].min()   # [mm]
    L       = pts[:, 2].max() - pts[:, 2].min()   # height along loading axis Z [mm]

    # Cross-sectional area from bounding-box footprint
    # BUG FIX: was `A = 14`  (the diameter in mm, not the area in mm²)
    A = x_range * y_range                         # [mm²]
    log.info("  Bounding box: %.2f × %.2f × %.2f mm  →  A = %.2f mm²", x_range, y_range, L, A)

    # ------------------------------------------------------------------
    # Apparent modulus
    # ------------------------------------------------------------------
    # BUG FIX: delta_u = -0.1 mm (matches template_sim.inp)
    #          The TODO stub in the original notebook used -0.01 mm (10× off)
    epsilon  = DELTA_U / L            # apparent strain [-]
    sigma    = F / A                  # apparent stress [MPa]
    E_app    = sigma / epsilon        # apparent modulus [MPa]
    stiffness = abs(F / DELTA_U)     # [N/mm]

    print(
        f"\n{'─'*45}\n"
        f"  Reaction force F     = {F:.4f} N\n"
        f"  Cross-section A      = {A:.4f} mm²\n"
        f"  Sample height L      = {L:.4f} mm\n"
        f"  Applied Δu           = {DELTA_U} mm\n"
        f"  Stiffness            = {stiffness:.2f} N/mm\n"
        f"  Apparent strain ε    = {epsilon:.6f}\n"
        f"  Apparent stress σ    = {sigma:.6f} MPa\n"
        f"  Apparent modulus E   = {E_app:.2f} MPa\n"
        f"{'─'*45}"
    )

    # ------------------------------------------------------------------
    # BV/TV from saved segmented array
    # ------------------------------------------------------------------
    bone_np = np.load(seg_npy_path)
    TV   = float(np.prod(bone_np.shape))
    BV   = float(np.sum(bone_np > 0))
    BVTV = BV / TV
    log.info("  BV/TV = %.4f (%.2f %%)", BVTV, BVTV * 100)

    return {
        "sample_id":        sample_id,
        "bvtv":             round(BVTV, 6),
        "reaction_force_N": round(F, 4),
        "stiffness_Nmm":    round(stiffness, 2),
        "app_strain":       round(epsilon, 6),
        "app_stress_MPa":   round(sigma, 6),
        "app_modulus_MPa":  round(E_app, 2),
        "cross_section_mm2": round(A, 4),
        "height_mm":        round(L, 4),
    }


# ===========================================================================
# Full pipeline for one sample
# ===========================================================================
def run_pipeline_single(img_path: Path, ds: int = 4) -> dict | None:
    """
    Run the full µFE pipeline for a single sample.

    Parameters
    ----------
    img_path : Path   raw .ISQ (or .mha) image
    ds       : int    downsampling factor for meshing

    Returns
    -------
    result dict, or None if simulation failed
    """
    sample_id = img_path.stem
    outpath   = img_path.parent / "derived"
    outpath.mkdir(parents=True, exist_ok=True)

    log.info("\n%s\nSAMPLE: %s\n%s", "=" * 50, sample_id, "=" * 50)

    try:
        seg_path = preprocess(img_path, outpath)
        _, vtk_path, inp_path = generate_mesh(seg_path, img_path, ds=ds)

        ok = run_simulation(inp_path)
        if not ok:
            log.error("Simulation failed for %s — skipping postprocessing.", sample_id)
            return None

        seg_npy = seg_path.with_suffix(".npy")
        result  = postprocess(vtk_path, seg_npy, sample_id)
        return result

    except Exception as exc:
        log.exception("Pipeline failed for %s: %s", sample_id, exc)
        return None


# ===========================================================================
# Entry point — loop over samples
# ===========================================================================
def run_pipeline(paths: list[Path], ds: int = 4) -> pd.DataFrame:
    """
    Run the full pipeline for every sample in *paths* and save a results TSV.

    Parameters
    ----------
    paths : list[Path]   one Path per raw image file
    ds    : int          shared downsampling factor
    """
    rows = []
    for p in paths:
        result = run_pipeline_single(p, ds=ds)
        if result is not None:
            rows.append(result)

    df = pd.DataFrame(rows)
    if not df.empty:
        write_header = not OUTPUT_TSV.exists()
        df.to_csv(OUTPUT_TSV, mode="a", header=write_header, index=False, sep="\t")
        log.info("\nResults saved → %s", OUTPUT_TSV)
        print(df.to_string(index=False))
        df.to_csv("/home/bmelab/bmelabs/2026/testing/MSB_BMELab/114-MorphologyElasticityTrabecularBone/muFE/02_RESULTS/pipeline_group01.tsv")
    else:
        log.warning("No results to save.")

    return df


def main():
    paths = [
        Path("00_DATA/group01/A1/C0004351.ISQ"),
        Path("00_DATA/group01/A1/C0004353.ISQ"),
        Path("00_DATA/group01/A2/C0004352.ISQ"),
        Path("00_DATA/group01/A2/C0004354.ISQ"),

    ]
    run_pipeline(paths, ds=6)
    print("\nDone!")


if __name__ == "__main__":
    main()
