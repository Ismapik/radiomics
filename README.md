# HR-CTV radiomics in cervical cancer

Code for *Multimodel Cervical Cancer Recurrence Prediction With
Clinical, Dosimetric and Radiomic Features: A Dutch Single-Centre Study*
(Amsterdam UMC, 2026).

The pipeline extracts PyRadiomics features from the high-risk clinical target volume
(HR-CTV) on the planning MRI of the first brachytherapy fraction, filters the feature
set, and fits a ladder of six nested logistic regression models for recurrence within
five years.

## Files

| File | What it does |
|---|---|
| `radiomics_v3.py` | Extracts 1,130 features per patient from the DICOM files (planning MRI + RTSTRUCT): 14 shape features and 93 first-order/texture features from the original image, 8 wavelet decompositions and 3 LoG-filtered images (σ = 2, 3, 4 mm). Writes `radiomics_v3.csv` and `.xlsx`. |
| `filter_features.py` | Applies the three pre-specified filters: missing in more than 30% of patients, near-constant features (caret rule), and one of each pair with \|r\| > 0.90 (shape and non-shape pruned separately). Writes the retained features, the removed features with the reason for removal, and the counts per family and per image filter. |
| `radiomics_pipeline9.py` | Fits the six nested models in repeated stratified 5-fold cross-validation. The LASSO radiomics signature is fitted inside every training fold; the script returns discrimination, calibration, decision curve analysis, operating points and patient-level bootstrap confidence intervals. |

## Data

The scripts expect one subdirectory per patient, containing the DICOM files of the
planning MRI (first brachytherapy fraction) and the RTSTRUCT with the HR-CTV contour.
Patient folders are recognised by the numbers in their name (`15`, `P15_x`); folders
without a number are skipped and reported.

No patient data are included in this repository and none can be shared. The imaging,
the delineations and the derived feature tables contain patient-level information.
Requests for access go to the Department of Radiation Oncology, Amsterdam UMC.

## Requirements

Python 3.9 is the last version for which PyRadiomics 3.1.0 ships wheels; on newer
versions it has to be built from source. The scripts need `numpy`, `pandas`,
`pydicom`, `SimpleITK`, `Pillow`, `pyradiomics`, `scikit-learn`, `matplotlib` and
optionally `openpyxl` for the xlsx output.

```
pip install numpy pandas pydicom SimpleITK Pillow pyradiomics scikit-learn matplotlib openpyxl
```

## Usage

1. Feature extraction:

```
python radiomics_v3.py --root /path/to/patient_folders --out /path/to/output
```

2. Feature filtering (run it from the directory that contains
   `radiomics_wide_filtered.xlsx`):

```
python filter_features.py
```

3. Modelling:

```
python radiomics_pipeline9.py --data radiomics_wide_filtered.xlsx --repeats 100 --bootstrap 200 --bootstrap-repeats 3 --out results
```

The project used 100 repetitions of stratified 5-fold cross-validation, 200
patient-level bootstrap samples with three cross-validation repetitions each, and
ridge regression with `C = 1.0` in the final model. The defaults of the script are
lighter (`--repeats 5`, `--bootstrap 0`); the command above reproduces the reported
numbers.

## Note on the analysis file

`filter_features.py` and `radiomics_pipeline9.py` read `radiomics_wide_filtered.xlsx`:
one row per patient, with the clinical variables (`histological typing`,
`lymph node metastasis`, `stadium`), the dosimetry column `ctvhr_d90iso_gy` and the
radiomics columns prefixed `set1_`. That file is assembled from the radiomics export
and the institutional registry and is not part of this repository. Without it the
scripts cannot run end to end; they are shared for transparency, not as a turnkey
pipeline.
