#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
filter_features.py

Stap 1 van de analyse: radiomics-kenmerken filteren.
Geen selectie, geen model, geen instellingen. Run en klaar.

    python filter_features.py

Drie filters, in deze volgorde:

    1. missende waarden   kenmerken met >30% missend vallen af
    2. variantie          de caret-regel voor bijna-constante kenmerken
    3. correlatie         van elk paar met |r| > 0,90 blijft er een over

Filteren gebruikt de uitkomst NOOIT en is daarom geen lekkage. Selectie
gebruikt de uitkomst wel en moet dus binnen de cross-validatie gebeuren.

Uitvoer:
    kenmerken_na_filter.csv        de kenmerken die overblijven
    kenmerken_verwijderd.csv       elk verwijderd kenmerk met de reden
    kenmerkfamilies.csv            hoeveel varianten per familie overblijven
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd


DATA = "radiomics_wide_filtered.xlsx"

MAX_MISSING = 0.30      # maximaal aandeel missende waarden per kenmerk
FREQ_RATIO = 95 / 5     # caret: meest voorkomende waarde / op een na meest
MIN_UNIQUE = 0.10       # caret: minimaal aandeel unieke waarden
MAX_CORR = 0.90         # |r| waarboven twee kenmerken als duplicaat gelden

FILTER_PREFIXES = (
    "wavelet-LLL_", "wavelet-LLH_", "wavelet-LHL_", "wavelet-LHH_",
    "wavelet-HLL_", "wavelet-HLH_", "wavelet-HHL_", "wavelet-HHH_",
    "log-sigma-1-0-mm-3D_", "log-sigma-2-0-mm-3D_", "log-sigma-3-0-mm-3D_",
    "log-sigma-4-0-mm-3D_", "log-sigma-5-0-mm-3D_",
    "lbp-3D-m1_", "lbp-3D-m2_", "lbp-3D-k_",
    "square_", "squareroot_", "logarithm_", "exponential_", "gradient_",
    "original_",
)


def family_of(name):
    """Stript het filter-voorvoegsel zodat varianten van hetzelfde kenmerk
    in dezelfde familie vallen. set1_wavelet-HLL_gldm_Foo -> gldm_Foo."""
    text = str(name)
    if text.startswith("set1_"):
        text = text[5:]
    for prefix in FILTER_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def filter_of(name):
    """Geeft het beeld-filter terug waarmee een kenmerk is berekend."""
    text = str(name)
    if text.startswith("set1_"):
        text = text[5:]
    for prefix in FILTER_PREFIXES:
        if text.startswith(prefix):
            return prefix.rstrip("_")
    return "overig"


def load_radiomics(path):
    """Leest het bestand en houdt de radiomics-kolommen over."""
    df = pd.read_excel(path)
    columns = [c for c in df.columns if str(c).startswith("set1_")]
    return df[columns].apply(pd.to_numeric, errors="coerce")


def filter_missing(X):
    """Verwijdert kenmerken met te veel missende waarden."""
    fraction = X.isna().mean()
    keep = fraction[fraction <= MAX_MISSING].index.tolist()
    removed = [(c, "missende waarden",
                f"missend in {fraction[c]:.0%} van de patienten")
               for c in X.columns if c not in keep]
    print(f"  {len(X.columns) - len(keep):5d} verwijderd   "
          f"{len(keep):5d} over   (drempel {MAX_MISSING:.0%} missend)")
    return X[keep], removed


def filter_variance(X):
    """Verwijdert constante en bijna-constante kenmerken.

    De caret-regel: een kenmerk is bijna constant als de meest voorkomende
    waarde meer dan 19x zo vaak voorkomt als de op een na meest voorkomende
    (95/5), of als minder dan 10% van de waarden uniek is.
    """
    keep, removed = [], []
    for column in X.columns:
        values = X[column].dropna()
        if len(values) == 0:
            removed.append((column, "variantie", "geen waarden"))
            continue

        counts = values.value_counts()
        top = counts.iloc[0]
        second = counts.iloc[1] if len(counts) > 1 else 1
        ratio = top / second
        unique_fraction = len(counts) / len(values)

        if ratio > FREQ_RATIO:
            removed.append((column, "variantie",
                            f"bijna constant (zelfde waarde in {top / len(values):.0%})"))
        elif unique_fraction < MIN_UNIQUE:
            removed.append((column, "variantie",
                            f"slechts {unique_fraction:.0%} unieke waarden"))
        else:
            keep.append(column)

    print(f"  {len(X.columns) - len(keep):5d} verwijderd   "
          f"{len(keep):5d} over   (caret-regel)")
    return X[keep], removed


def is_shape(name):
    """Vormkenmerken beschrijven de geometrie, niet het intensiteitspatroon."""
    return "shape" in str(name)


def _prune_within(X, columns):
    """Correlatiefilter binnen een groep kolommen."""
    sub = X[columns]
    correlation = sub.corr().abs().to_numpy()
    correlation = np.nan_to_num(correlation, nan=0.0)

    drop = {}
    for i in range(correlation.shape[0]):
        if i in drop:
            continue
        for j in range(i + 1, correlation.shape[1]):
            if j not in drop and correlation[i, j] > MAX_CORR:
                drop[j] = (columns[i], correlation[i, j])
    return drop


def filter_correlation(X):
    """Verwijdert van elk sterk gecorreld paar het tweede kenmerk.

    Vorm en niet-vorm worden APART gefilterd. Zonder die scheiding kan een
    textuurkenmerk dat toevallig hoog correleert met het volume het
    volumekenmerk wegfilteren -- en dat is precies wat je wilt behouden.
    """
    shape_cols = [c for c in X.columns if is_shape(c)]
    other_cols = [c for c in X.columns if not is_shape(c)]

    drop_info = {}
    for group in (shape_cols, other_cols):
        for j, (keeper, r) in _prune_within(X, group).items():
            drop_info[group[j]] = (keeper, r)

    keep = [c for c in X.columns if c not in drop_info]
    removed = [(c, "correlatie", f"r = {r:.2f} met {keeper}")
               for c, (keeper, r) in drop_info.items()]
    print(f"  {len(X.columns) - len(keep):5d} verwijderd   "
          f"{len(keep):5d} over   (drempel |r| > {MAX_CORR})")
    return X[keep], removed


def main() -> int:
    if not os.path.exists(DATA):
        print(f"FOUT: {DATA} niet gevonden in {os.getcwd()}")
        return 1

    X = load_radiomics(DATA)
    total = X.shape[1]
    before = pd.Series([family_of(c) for c in X.columns]).value_counts()

    print(f"Radiomics-kenmerken in het bestand: {total}\n")
    print("Filteren (de uitkomst wordt hier niet gebruikt)\n")

    X, missing_removed = filter_missing(X)
    X, variance_removed = filter_variance(X)
    X, corr_removed = filter_correlation(X)

    kept = X.shape[1]
    print(f"\nVan {total} naar {kept} kenmerken "
          f"({100 * kept / total:.1f}% behouden)\n")


    after = pd.Series([family_of(c) for c in X.columns]).value_counts()
    families = pd.DataFrame({"voor_filter": before, "na_filter": after})
    families = families.fillna(0).astype(int)
    families.index.name = "familie"
    families = families.sort_values("voor_filter", ascending=False)
    families.to_csv("kenmerkfamilies.csv")

    collapsed = families[families["voor_filter"] > 1].head(25)
    print("Families met meer dan een variant, voor en na het filteren\n")
    print(f"  {'familie':44s} {'voor':>5s} {'na':>5s}")
    print("  " + "-" * 56)
    for name, row in collapsed.iterrows():
        print(f"  {str(name)[:44]:44s} {row['voor_filter']:5d} "
              f"{row['na_filter']:5d}")

    many = families[families["voor_filter"] >= 10]
    if len(many):
        print(f"\n  {len(many)} families hadden 10 of meer varianten; samen "
              f"{int(many['voor_filter'].sum())} kenmerken, nu "
              f"{int(many['na_filter'].sum())}.")
        print("  Dat is het multipliciteitsprobleem: een signaal dat zich "
              "onder veel\n  filters herhaalt, telt in de ruwe data ook veel "
              "keer mee.")


    filters_before = pd.Series([filter_of(c) for c in
                                load_radiomics(DATA).columns]).value_counts()
    filters_after = pd.Series([filter_of(c) for c in X.columns]).value_counts()
    filters = pd.DataFrame({"voor_filter": filters_before,
                            "na_filter": filters_after})
    filters = filters.fillna(0).astype(int)
    filters.index.name = "beeld_filter"
    filters = filters.sort_values("voor_filter", ascending=False)
    filters.to_csv("filters.csv")

    print("\nBeeld-filters waarmee de kenmerken zijn geextraheerd\n")
    print(f"  {'filter':30s} {'voor':>6s} {'na':>6s}")
    print("  " + "-" * 44)
    for name, row in filters.iterrows():
        print(f"  {str(name)[:30]:30s} {row['voor_filter']:6d} "
              f"{row['na_filter']:6d}")
    print(f"  {'TOTAAL':30s} {int(filters['voor_filter'].sum()):6d} "
          f"{int(filters['na_filter'].sum()):6d}")


    pd.DataFrame({"kenmerk": X.columns}).to_csv("kenmerken_na_filter.csv",
                                                index=False)
    rows = missing_removed + variance_removed + corr_removed
    pd.DataFrame(rows, columns=["kenmerk", "filter", "reden"]).to_csv(
        "kenmerken_verwijderd.csv", index=False)

    print(f"\n  kenmerken_na_filter.csv      {kept} kenmerken")
    print(f"  kenmerken_verwijderd.csv     {len(rows)} verwijderd, met reden")
    print(f"  kenmerkfamilies.csv          {len(families)} families")
    print(f"  filters.csv                  {len(filters)} beeld-filters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
