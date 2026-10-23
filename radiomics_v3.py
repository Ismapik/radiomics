#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
radiomics_v3.py  --  HR-CTV radiomics, set1, met vastgezette instellingen.

Gebaseerd op radiomics_wide_filtered.py v2.3 (dezelfde DICOM-logica).
Alleen de extractie-instellingen zijn gecorrigeerd. Geen instellingen nodig:

    python radiomics_v3.py

Wat er anders is dan v2.3:

  * binWidth 25 -> 5.  Met normalizeScale 100 gaf binWidth 25 maar vier
    grijswaarden; nu zijn het er twintig, binnen de gebruikelijke 16-64.

  * resampledPixelSpacing [2, 2, 2] toegevoegd met B-spline interpolatie.
    Kenmerken worden nu op isotrope voxels berekend en zijn daarmee
    vergelijkbaar tussen patienten met verschillende sliceditkte. 2 mm is
    gekozen omdat de MR-series in dit cohort 3-5 mm sliceditkte hebben;
    naar 1 mm resamplen zou de door-de-snede-richting 3 tot 5 keer
    opblazen en textuur verzinnen.

  * preCrop en correctMask aangezet: sneller en robuuster.

  * Drie beeldtypen: original, wavelet (8 decomposities) en LoG
    (sigma 2/3/4 mm). Samen 1 + 8 + 3 = 12 gefilterde sets.

    De pixeltranformaties (square, squareroot, logarithm, exponential),
    de gradient en LBP3D zijn weggelaten. Ze leveren kenmerken die sterk
    correleren met het originele beeld en die de correlatiefilter er
    daarna toch uitgooit. Minder beeldtypen betekent ook minder
    kenmerken en dus minder multiple testing. Niet instelbaar.

  * Alleen set1 (het eerste tijdpunt). Niet instelbaar.

  * Alle submappen met een patientnummer in de naam worden verwerkt
    ('15', 'P15_x', ...). Mappen zonder nummer worden overgeslagen en
    dat wordt aan het eind gemeld.

Uitvoer:
    radiomics_v3.xlsx   en   radiomics_v3.csv

Benodigd: pydicom, SimpleITK, numpy, Pillow, pyradiomics, pandas, openpyxl
"""

from __future__ import annotations

import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from PIL import Image, ImageDraw
from radiomics import featureextractor

warnings.filterwarnings("ignore")

__version__ = "3.0"

MR_UID = "1.2.840.10008.5.1.4.1.1.4"
CT_UID = "1.2.840.10008.5.1.4.1.1.2"
RS_UID = "1.2.840.10008.5.1.4.1.1.481.3"

TARGET_ROI = "HR-CTV"
OUTPUT_STEM = "radiomics_v3"


SETTINGS = {
    "binWidth": 5,                      # 20 grijswaarden op een schaal van 0-100
    "normalize": True,
    "normalizeScale": 100,
    "resampledPixelSpacing": [2, 2, 2],  # isotroop, mm
    "interpolator": "sitkBSpline",
    "label": 1,
    "geometryTolerance": 1e-5,
    "preCrop": True,
    "correctMask": True,
}

LOG_SIGMA = [2.0, 3.0, 4.0]

IMAGE_TYPES = {
    "Original": {},
    "Wavelet": {},
    "LoG": {"sigma": LOG_SIGMA},
}


def build_params() -> dict:
    """Het volledige PyRadiomics parameterdict."""
    return {"imageType": IMAGE_TYPES, "setting": dict(SETTINGS)}


def _tag_str(ds, tag) -> str:
    v = ds.get(tag)
    if v is None:
        return ""
    try:
        val = v.value
    except AttributeError:
        return str(v).strip()
    if val is None:
        return ""
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="replace").strip()
        except Exception:
            return str(val)
    return str(val).strip()


def _sop_uid(ds) -> str:
    try:
        v = ds.get((0x0008, 0x0016))
        return "" if v is None else str(v.value).strip()
    except Exception:
        return ""


def _norm_dicom_date(s):
    s = (s or "").strip()
    if len(s) >= 8:
        y, m, d = s[0:4], s[4:6], s[6:8]
        if y.isdigit() and m.isdigit() and d.isdigit():
            return f"{y}-{m}-{d}"
    return None


def extract_date(text: str):
    if not text:
        return None
    t = text.replace("\\", "/")
    m = re.search(r"(19|20)(\d{2})[-_./](\d{1,2})[-_./](\d{1,2})", t)
    if m:
        y, mo, d = int(m.group(1) + m.group(2)), int(m.group(3)), int(m.group(4))
        if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    m = re.search(r"(19|20)(\d{2})(\d{2})(\d{2})", t)
    if m:
        y, mo, d = int(m.group(1) + m.group(2)), int(m.group(3)), int(m.group(4))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    return None


def _safe_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s or "").strip("._")
    return s or "unknown"


def collect_dicoms(patient_dir: Path):
    buckets = {"mr": [], "rs": [], "ct": [], "other": []}
    for fp in patient_dir.rglob("*"):
        if not fp.is_file() or fp.name.startswith("."):
            continue
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
        except Exception:
            continue
        sop = _sop_uid(ds)
        if sop == MR_UID:
            buckets["mr"].append(fp)
        elif sop == RS_UID:
            buckets["rs"].append(fp)
        elif sop == CT_UID:
            buckets["ct"].append(fp)
        else:
            buckets["other"].append(fp)
    return buckets


def group_mr_series(mr_files):
    groups = defaultdict(list)
    for f in mr_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        groups[_tag_str(ds, (0x0020, 0x000E))].append(f)

    result = {}
    for suid, files in groups.items():
        if not files:
            continue
        try:
            ds0 = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
            study_date = _norm_dicom_date(_tag_str(ds0, (0x0008, 0x0020)))
            desc = _tag_str(ds0, (0x0008, 0x103E))
        except Exception:
            study_date, desc = None, ""
        result[suid] = {"files": sorted(str(f) for f in files),
                        "n_slices": len(files),
                        "study_date": study_date, "desc": desc}
    return result


def _rs_ref_suid(ds):
    fof = ds.get((0x3006, 0x0010))
    if not fof:
        return None
    for fr in fof:
        study = fr.get((0x3006, 0x0012))
        if study:
            for st in study:
                series = st.get((0x3006, 0x0014))
                if series:
                    for s in series:
                        suid = s.get((0x0020, 0x000E))
                        if suid:
                            return str(suid.value).strip()
    return None


def _rs_date(ds):
    for tag in [(0x3006, 0x0008), (0x0008, 0x0020)]:
        d = _norm_dicom_date(_tag_str(ds, tag))
        if d:
            return d
    return None


def parse_rs(rs_files, patient_dir: Path):
    out = []
    for fp in sorted(rs_files):
        try:
            ds = pydicom.dcmread(str(fp), force=True)
        except Exception:
            continue
        rel_folder = str(fp.relative_to(patient_dir).parent)
        out.append({"path": fp, "ds": ds, "folder": rel_folder,
                    "date": extract_date(rel_folder),
                    "dicom_date": _rs_date(ds),
                    "ref_suid": _rs_ref_suid(ds)})
    return out


def _is_practice(rs_info) -> bool:
    rel = rs_info["folder"].lower()
    return ("oefen" in rel) or ("oz_mr" in rel)


def find_roi_number(ds, label: str) -> int:
    structs = [(int(s.ROINumber), str(s.ROIName))
               for s in ds.StructureSetROISequence]
    target = label.strip().upper()
    for num, name in structs:
        if name.strip().upper() == target:
            return num
    for num, name in structs:
        if target in name.upper():
            return num
    raise ValueError(f"ROI '{label}' niet gevonden. Beschikbaar: "
                     f"{[n for _, n in structs]}")


def extract_contour_points(ds, roi_number: int):
    for c in ds.ROIContourSequence:
        if c.ReferencedROINumber == roi_number:
            return [np.array(cont.ContourData).reshape(-1, 3)
                    for cont in c.ContourSequence]
    raise ValueError(f"Geen contourdelen voor ROI {roi_number}")


def _polygon_to_slice(coords, size_x, size_y, origin, spacing):
    pixel_coords = [
        (int(round((pt[0] - origin[0]) / spacing[0])),
         size_y - 1 - int(round((pt[1] - origin[1]) / spacing[1])))
        for pt in coords
    ]
    img = Image.new("L", (size_x, size_y), 0)
    ImageDraw.Draw(img).polygon(pixel_coords, outline=1, fill=1)
    return np.array(img)


def build_mask(contour_slices, image: sitk.Image) -> np.ndarray:
    size, spacing, origin = image.GetSize(), image.GetSpacing(), image.GetOrigin()
    mask = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    for coords in contour_slices:
        z_idx = int(round((coords[0, 2] - origin[2]) / spacing[2]))
        if not (0 <= z_idx < size[2]):
            continue
        mask[z_idx, :, :] = _polygon_to_slice(coords, size[0], size[1],
                                              origin, spacing)
    return mask


def load_series(file_paths) -> sitk.Image:
    file_paths = [str(f) for f in file_paths]
    if not file_paths:
        raise ValueError("lege serie")
    try:
        return _load_series_manual(file_paths)
    except Exception as e:
        print(f"    [load_series] handmatige loader faalde: {e}")
        print("    [load_series] terugval op ImageSeriesReader "
              "(kan een verkeerde slice-afstand geven)")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_paths)
        return reader.Execute()


def _load_series_manual(file_paths) -> sitk.Image:
    records = []
    for fp in file_paths:
        ds = pydicom.dcmread(fp, force=True)
        ipp = ds.get((0x0020, 0x0032))
        records.append((float(ipp.value[2]) if ipp is not None else 0.0, ds))
    records.sort(key=lambda r: r[0])

    positions = [r[0] for r in records]
    arrays = [r[1].pixel_array.astype(np.float32) for r in records]
    vol = np.stack(arrays, axis=0)

    ds0 = records[0][1]
    ipp0 = [float(v) for v in ds0.ImagePositionPatient]
    ps = [float(v) for v in ds0.PixelSpacing]
    iop = [float(v) for v in ds0.ImageOrientationPatient]
    x_dir, y_dir = iop[0:3], iop[3:6]
    z_dir = np.cross(np.array(x_dir, float), np.array(y_dir, float))
    n = float(np.linalg.norm(z_dir))
    z_dir = list(z_dir / n) if n > 1e-9 else [0.0, 0.0, 1.0]

    if len(positions) >= 2:
        steps = [positions[i + 1] - positions[i]
                 for i in range(len(positions) - 1)]
        z_spacing = float(np.median(steps))
    else:
        z_spacing = 1.0

    image = sitk.GetImageFromArray(vol)
    image.SetOrigin([ipp0[0], ipp0[1], ipp0[2]])
    image.SetSpacing([ps[0], ps[1], abs(z_spacing) or 1.0])
    image.SetDirection(x_dir + y_dir + z_dir)
    return image


def extract_radiomics(image, mask_img, params) -> dict:
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    result = extractor.execute(image, mask_img)
    return {k: v for k, v in result.items() if "diagnostics" not in k}


def _mr_candidates(mr_series, rs_info, date_label):
    seen = []

    def add(suid):
        if suid not in seen:
            seen.append(suid)

    by_slices = sorted(mr_series.items(), key=lambda kv: -kv[1]["n_slices"])

    for suid, info in by_slices:
        if "BTFE" in info["desc"].upper() and info["study_date"] == date_label:
            add(suid)
    ref = rs_info["ref_suid"]
    if ref and ref in mr_series:
        add(ref)
    for suid, info in by_slices:
        if info["study_date"] == date_label:
            add(suid)
    for suid, info in by_slices:
        if "BTFE" in info["desc"].upper():
            add(suid)
    for suid, info in by_slices:
        add(suid)

    return [(suid, mr_series[suid]) for suid in seen]


def process_set(date_label, rs_infos, mr_series, params, roi):
    best_rs, best_contours, best_n = None, None, -1
    for r in rs_infos:
        try:
            roi_num = find_roi_number(r["ds"], roi)
            contours = extract_contour_points(r["ds"], roi_num)
        except Exception:
            continue
        if len(contours) > best_n:
            best_rs, best_contours, best_n = r, contours, len(contours)

    if best_rs is None:
        print(f"      [set {date_label}] geen {roi}-contour")
        return None

    for suid, info in _mr_candidates(mr_series, best_rs, date_label):
        try:
            image = load_series(info["files"])
        except Exception:
            continue
        try:
            mask = build_mask(best_contours, image)
        except Exception:
            continue
        if not np.any(mask > 0):
            continue

        mask_img = sitk.GetImageFromArray(mask)
        mask_img.CopyInformation(image)

        try:
            return extract_radiomics(image, mask_img, params)
        except Exception as e:
            print(f"      [set {date_label}] radiomics faalde: {e}")
            return None

    print(f"      [set {date_label}] geen MR-serie gaf een geldig {roi}-masker")
    return None


def process_patient(patient_dir: Path, params, roi):
    """Geeft de features van het EERSTE tijdpunt, of None."""
    pid = patient_dir.name
    buckets = collect_dicoms(patient_dir)
    mr_series = group_mr_series(buckets["mr"])
    if not mr_series:
        return None, "geen MR-serie"

    rs_infos = [r for r in parse_rs(buckets["rs"], patient_dir)
                if not _is_practice(r)]
    if not rs_infos:
        return None, "geen RS-bestanden"

    by_date = defaultdict(list)
    for r in rs_infos:
        by_date[r["date"] or r["dicom_date"] or r["folder"]].append(r)

    for date_label, infos in sorted(by_date.items(),
                                    key=lambda kv: _date_sort_key(kv[0])):
        feats = process_set(date_label, infos, mr_series, params, roi)
        if feats is not None:
            return (date_label, feats), None

    return None, f"geen geldige {roi}-radiomics in set1"


def _date_sort_key(label: str):
    return (0, label) if re.match(r"\d{4}-\d{2}-\d{2}", label) else (1, label)


def folder_id(name: str):
    """Haalt het patientnummer uit een mapnaam: '15' -> 15, 'P15_x' -> 15."""
    if name.isdigit():
        return int(name)
    digits = re.findall(r"\d+", name)
    return int(digits[-1]) if digits else None


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="HR-CTV radiomics, set1, vaste instellingen")
    parser.add_argument("--root", default=".",
                        help="map met de patient-submappen (default: .)")
    parser.add_argument("--out", default=".",
                        help="uitvoermap (default: .)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    params = build_params()

    all_dirs = [d for d in root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
                and d.resolve() != out]

    patient_dirs = [d for d in all_dirs if folder_id(d.name) is not None]
    patient_dirs.sort(key=lambda d: _natural_key(d.name))
    overgeslagen = len(all_dirs) - len(patient_dirs)

    if not patient_dirs:
        print("Geen enkele map met een patientnummer gevonden.", file=sys.stderr)
        print(f"Mappen in {root}: {len(all_dirs)}.", file=sys.stderr)
        return 1

    print(f"radiomics_v3.py v{__version__}  (set1, vaste instellingen)")
    print(f"Patienten:        {len(patient_dirs)}"
          f"  ({overgeslagen} mappen zonder nummer overgeslagen)")
    print(f"Uitvoer:          {out}")
    print(f"ROI:              {TARGET_ROI}")
    print(f"binWidth:         {SETTINGS['binWidth']}  "
          f"(~{SETTINGS['normalizeScale'] // SETTINGS['binWidth']} grijswaarden)")
    print(f"resampling:       {SETTINGS['resampledPixelSpacing']} mm, "
          f"{SETTINGS['interpolator']}")
    print(f"beeldtypen:       {len(IMAGE_TYPES)} "
          f"({', '.join(IMAGE_TYPES)})")
    print()

    rows = []
    for d in patient_dirs:
        result, msg = process_patient(d, params, TARGET_ROI)
        if result is None:
            print(f"[{d.name}] overgeslagen: {msg}")
            continue
        date_label, feats = result
        row = {"patient_id": d.name, "datum_set1": date_label}
        for k in sorted(feats):
            row[f"set1_{k}"] = feats[k]
        rows.append(row)
        groups = sorted({k.split("_", 1)[0] for k in feats})
        print(f"[{d.name}] {date_label}  {len(feats)} kenmerken  "
              f"({len(groups)} beeldtypen)")

    if not rows:
        print("\nGeen enkele patient leverde radiomics op.")
        return 1

    df = pd.DataFrame(rows)
    csv_path = out / f"{OUTPUT_STEM}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    try:
        import openpyxl  # noqa: F401
        xlsx_path = out / f"{OUTPUT_STEM}.xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f"Wrote {xlsx_path}")
    except ImportError:
        print("openpyxl niet geinstalleerd -> alleen CSV")

    print(f"\nKlaar: {len(df)} patienten x {df.shape[1] - 2} kenmerken "
          f"(set1 only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
