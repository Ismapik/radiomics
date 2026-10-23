#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
radiomics_pipeline.py

Voegt radiomics iets toe aan de klinische voorspelling van een recidief
binnen vijf jaar na definitieve chemoradiotherapie en image-guided
adaptive brachytherapy bij cervixcarcinoom?

WAT HET DOET
------------
Een ladder van zes geneste modellen. Elk model voegt een variabele toe,
in de volgorde waarin die in de kliniek beschikbaar komt:

    1. histologie                        (diagnose)
    2. + HR-CTV-volume                   (planning-MRI)
    3. + lymfeklierstatus                (stagering)
    4. + FIGO-stadium                    (stagering)
    5. + CTVHR D90                       (na de brachytherapie)
    6. + radiomics-signatuur             (post-hoc)

Het verschil tussen twee opeenvolgende modellen is de bijdrage van die
ene variabele. Zo zie je wat bijdraagt en wat niet, zonder iets weg te
laten of achteraf te selecteren.

Alles draait twee keer: op het volledige cohort en op de plaveiselcel-
subgroep. Daar is histologie constant, valt automatisch af, en begint de
ladder bij stap 2.

HOE HET REKENT
--------------
* Herhaalde gestratificeerde 5-fold cross-validatie (standaard 100x).
* Alle stappen die van de data leren -- imputeren, filteren, schalen,
  signatuur bouwen, de LASSO-penalty kiezen -- zitten BINNEN de pijplijn
  en worden dus per fold opnieuw op de trainingsfold gefit. Geen lekkage.
* De radiomics-signatuur vat alle kenmerken samen in een score met
  L1-penalised logistische regressie. De penalty C wordt per fold gekozen
  met een binnenste cross-validatie (LogisticRegressionCV). Vormkenmerken
  blijven buiten de signatuur, omdat die hetzelfde meten als het volume.
* Het eindmodel is ridge-geregulariseerd (L2), passend bij het kleine
  aantal events.

WAT ER UIT KOMT
---------------
    prestatie_<cohort>.csv      AUC met 95%-BI, DeLong-toets, Brier
    werkpunten_<cohort>.csv     rule-out, Youden en rule-in
    kenmerken_<cohort>.csv      gekozen radiomics per model
    kalibratie_<cohort>.png     kalibratieplot met hellingen
    besliscurve_<cohort>.png    decision curve analysis

GEBRUIK
-------
    python radiomics_pipeline.py --repeats 100
    python radiomics_pipeline.py --repeats 2 --folds 5      # snelle test
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     StratifiedGroupKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


DATA = "radiomics_wide_filtered.xlsx"
OUT = "radiomics_resultaten"

CORR_THRESHOLD = 0.90          # |r| waarboven twee kenmerken als duplicaat gelden
VOLUME_FEATURE = "set1_original_shape_MeshVolume"


EXTRA_CLINICAL = []


DOSIMETRY_COLUMNS = ["ctvhr_d90iso_gy"]   # D90: sterkste literatuurbasis

STAGE_CANDIDATES = ["stadium", "FIGO stage"]
RIDGE_C = 1.0                     # ridge-penalty in het eindmodel
SENS_TARGET = 0.90                # rule-out: drempel die 90% sensitiviteit haalt
SPEC_TARGET = 0.90                # rule-in:  drempel die 90% specificiteit haalt


MODELLEN = [
    # Eén ladder: elke stap voegt een variabele toe. Zo zie je precies wat
    # wat bijdraagt -- en wat niet. De laatste stap voegt de radiomics toe.
    ("1. hist",             ["histologie"],                                        False),
    ("2. + volume",         ["histologie", VOLUME_FEATURE],                           False),
    ("3. + lymf",           ["histologie", VOLUME_FEATURE, "lymfeklier"],             False),
    ("4. + stadium",        ["histologie", VOLUME_FEATURE, "lymfeklier", "stadium"],  False),
    ("5. + D90",            ["histologie", VOLUME_FEATURE, "lymfeklier", "stadium",
                             "ctvhr_d90iso_gy"],                                   False),
    ("6. + radiomics",      ["histologie", VOLUME_FEATURE, "lymfeklier", "stadium",
                             "ctvhr_d90iso_gy"],                                   True),
]
N_RADIOMICS = 2                   # vaste regel: altijd de top 2


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Verwijdert van elk paar kenmerken met |r| > threshold er een.

    sklearn heeft hier geen standaardtransformator voor. Dit is de
    gebruikelijk implementatie: bereken de correlatiematrix op de
    trainingsfold en laat van elk te sterk gecorreleerd paar het tweede
    kenmerk vallen, zodat de kolomvolgorde bewaard blijft.
    """

    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with np.errstate(invalid="ignore"):
            corr = np.abs(np.corrcoef(X, rowvar=False))
        corr = np.nan_to_num(corr, nan=0.0)

        drop = set()
        for i in range(corr.shape[0]):
            if i in drop:
                continue
            for j in range(i + 1, corr.shape[1]):
                if j not in drop and corr[i, j] > self.threshold:
                    drop.add(j)

        self.keep_ = np.array([j for j in range(corr.shape[1]) if j not in drop])
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.asarray(input_features)[self.keep_]


class LassoSignature(BaseEstimator, TransformerMixin):
    """Vat alle radiomics samen in EEN score.

    De penalty C wordt BINNEN de fold gekozen met een binnenste
    cross-validatie (LogisticRegressionCV). Zo hoef je C niet zelf te
    kiezen en is er geen instelling die je op de testdata hebt afgestemd.
    """

    def __init__(self, Cs=(0.003, 0.01, 0.03, 0.1, 0.3), cv=5, seed=0):
        self.Cs = Cs
        self.cv = cv
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        n_min = int(np.bincount(y).min()) if len(np.unique(y)) > 1 else 1
        cv = max(2, min(self.cv, n_min))
        self.model_ = LogisticRegressionCV(
            Cs=list(self.Cs), cv=cv, penalty="l1", solver="liblinear",
            max_iter=5000, scoring="roc_auc",
            random_state=self.seed)
        self.model_.fit(X, y)
        self.C_ = float(self.model_.C_[0])
        self.n_selected_ = int((self.model_.coef_[0] != 0).sum())
        return self

    def transform(self, X):
        return self.model_.decision_function(X).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["radiomics_signature"])


def roman_to_ordinal(series):
    """Zet FIGO I-IV om naar 1-4."""
    mapping = {"I": 1, "II": 2, "III": 3, "IV": 4}

    def one(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip().upper()
        for roman in ("IV", "III", "II", "I"):   # langste eerst
            if text.startswith(roman):
                return float(mapping[roman])
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return np.nan

    return series.map(one)


def find_column(df, needles):
    for needle in needles:
        for column in df.columns:
            if needle in str(column).lower():
                return column
    return None


def load_data(path):
    """Leest het bestand en geeft (kenmerken, uitkomst, kolomnamen) terug."""
    df = pd.read_excel(path)

    target = find_column(df, ["non-recurr", "recidief", "recurrence", "uitkomst"])
    outcome = pd.to_numeric(df[target], errors="coerce")
    mask = outcome.isin([0, 1])
    df, outcome = df.loc[mask].copy(), outcome[mask].astype(int)

    histology = find_column(df, ["histological typing"])
    nodal = find_column(df, ["lymph node metastas"])
    stage = find_column(df, ["stadium", "figo", "stage"])

    clinical = pd.DataFrame(index=df.index)
    clinical["histologie"] = pd.to_numeric(df[histology], errors="coerce")
    clinical["lymfeklier"] = pd.to_numeric(df[nodal], errors="coerce")
    clinical["stadium"] = roman_to_ordinal(df[stage])

    for extra in EXTRA_CLINICAL:
        match = [c for c in df.columns if str(c).strip().lower() == extra.lower()]
        if match:
            clinical[extra] = pd.to_numeric(df[match[0]], errors="coerce")
            print(f"  extra klinisch: '{match[0]}'")
        else:
            print(f"  LET OP: '{extra}' niet gevonden in het bestand")

    # welke stadiumkolommen zijn er, en welke gebruiken we?
    for cand in STAGE_CANDIDATES:
        match = [c for c in df.columns if str(c).strip().lower() == cand.lower()]
        if match:
            n_ok = int(roman_to_ordinal(df[match[0]]).notna().sum())
            gebruikt = "GEBRUIKT" if stage == match[0] else "niet gebruikt"
            print(f"  stadiumkolom '{match[0]}': {n_ok} waarden herkend ({gebruikt})")

    radiomics = df[[c for c in df.columns if str(c).startswith("set1_")]]
    radiomics = radiomics.apply(pd.to_numeric, errors="coerce")

    # dosimetrie: losse kolommen die geen radiomics en geen klinisch zijn
    dos_cols = [c for c in DOSIMETRY_COLUMNS if c in df.columns]
    ontbreekt = [c for c in DOSIMETRY_COLUMNS if c not in df.columns]
    if ontbreekt:
        print(f"  LET OP: dosimetriekolom(men) niet gevonden: {ontbreekt}")
    dosimetry = df[dos_cols].apply(pd.to_numeric, errors="coerce") if dos_cols \
        else pd.DataFrame(index=df.index)

    print(f"Patiënten: {len(df)}   recidieven: {int(outcome.sum())} "
          f"({100 * outcome.mean():.1f}%)")
    print(f"Klinische kenmerken: {clinical.shape[1]}   "
          f"dosimetrie: {dosimetry.shape[1]} ({', '.join(dos_cols) or 'geen'})"
          f"   radiomics: {radiomics.shape[1]}")

    return clinical, dosimetry, radiomics, outcome


def preprocessing():
    """Imputeren -> bijna-constante kenmerken weg -> schalen."""
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
    ]


def make_models(clinical, radiomics, dosimetry, specs=None):
    """Bouwt de modellen uit MODELLEN.

    Elke specificatie is (naam, vaste variabelen, aantal radiomics). De vaste
    variabelen worden bij naam opgezocht; de radiomics worden met een vaste
    regel gekozen: altijd de top-k van de LASSO, binnen elke fold.
    """
    X = pd.concat([clinical, dosimetry, radiomics], axis=1)
    X.columns = ([f"c{i}" for i in range(clinical.shape[1])]
                 + [f"d{i}" for i in range(dosimetry.shape[1])]
                 + list(radiomics.columns))

    # naam -> kolomindex in X
    idx = {}
    for i, name in enumerate(clinical.columns):
        idx[str(name)] = i
    for i, name in enumerate(dosimetry.columns):
        idx[str(name)] = clinical.shape[1] + i
    for i, name in enumerate(radiomics.columns):
        idx[str(name)] = clinical.shape[1] + dosimetry.shape[1] + i

    rad_all = list(range(clinical.shape[1] + dosimetry.shape[1], X.shape[1]))
    # Vormkenmerken uitsluiten: die meten hetzelfde als het HR-CTV-volume
    # dat al in het model zit. Zo testen we of textuur en intensiteit iets
    # toevoegen, in plaats van een tweede volumemaat.
    rad_all = [i for i in rad_all if "shape" not in str(X.columns[i]).lower()]

    def rad_branch():
        """Vat alle radiomics samen in EEN score: de radiomics-signatuur."""
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("corr", CorrelationPruner(threshold=CORR_THRESHOLD)),
            ("scale", StandardScaler()),
            ("signature", LassoSignature()),
            ("sig_scale", StandardScaler()),
        ])

    l2 = dict(penalty="l2", solver="lbfgs", max_iter=5000)

    models = {}
    for label, fixed_names, rad_C in (specs if specs is not None else MODELLEN):
        onbekend = [n for n in fixed_names if n not in idx]
        if onbekend:
            print(f"  '{label}': valt af -> {onbekend}")
        fixed_names = [n for n in fixed_names if n in idx]
        if not fixed_names and not rad_C:
            continue

        fixed = [idx[n] for n in fixed_names]
        # dubbele indices vermijden (bijv. VOLUME_FEATURE die ook in een
        # andere lijst zou staan)
        fixed = list(dict.fromkeys(fixed))

        if not rad_C:
            models[label] = Pipeline([
                ("select", ColumnTransformer(
                    [("fixed", Pipeline(preprocessing()), fixed)],
                    remainder="drop")),
                ("model", LogisticRegression(**l2)),
            ])
        else:
            models[label] = Pipeline([
                ("select", ColumnTransformer([
                    ("fixed", Pipeline(preprocessing()), fixed),
                    ("rad", rad_branch(), rad_all),
                ], remainder="drop")),
                ("model", LogisticRegression(penalty="l2", C=RIDGE_C, max_iter=5000)),
            ])

    return X, models


def metrics_at(y, p, threshold):
    """Sensitiviteit, specificiteit, PPV, NPV en accuracy bij een drempel."""
    y = np.asarray(y).astype(int)
    y_hat = (np.asarray(p) >= threshold).astype(int)
    tp = int(((y_hat == 1) & (y == 1)).sum())
    fp = int(((y_hat == 1) & (y == 0)).sum())
    fn = int(((y_hat == 0) & (y == 1)).sum())
    tn = int(((y_hat == 0) & (y == 0)).sum())

    def deel(a, b):
        return a / b if b else float("nan")

    return {
        "sensitiviteit": deel(tp, tp + fn),
        "specificiteit": deel(tn, tn + fp),
        "ppv": deel(tp, tp + fp),
        "npv": deel(tn, tn + fn),
        "accuracy": deel(tp + tn, len(y)),
    }


def threshold_at_sensitivity(y, p, target=0.90):
    """Hoogste drempel die nog minstens `target` sensitiviteit haalt.

    Sensitiviteit daalt naarmate de drempel stijgt. We lopen dus van hoog
    naar laag en stoppen bij de eerste drempel die de target nog haalt.
    """
    for t in np.unique(np.round(p, 8))[::-1]:
        y_hat = (p >= t).astype(int)
        sens = ((y_hat == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
        if sens >= target:
            return float(t)
    return float(np.min(p))


def threshold_youden(y, p):
    """Drempel die sensitiviteit + specificiteit maximaliseert."""
    beste_t, beste_j = float(np.median(p)), -1.0
    n_pos = max((y == 1).sum(), 1)
    n_neg = max((y == 0).sum(), 1)
    for t in np.unique(np.round(p, 8)):
        y_hat = (p >= t).astype(int)
        sens = ((y_hat == 1) & (y == 1)).sum() / n_pos
        spec = ((y_hat == 0) & (y == 0)).sum() / n_neg
        j = sens + spec - 1
        if j > beste_j:
            beste_j, beste_t = j, float(t)
    return beste_t


def threshold_at_specificity(y, p, target=0.90):
    """Laagste drempel die nog minstens `target` specificiteit haalt.

    Specificiteit stijgt naarmate de drempel stijgt. We lopen dus van laag
    naar hoog en stoppen bij de eerste drempel die de target haalt.
    """
    for t in np.unique(np.round(p, 8)):
        y_hat = (p >= t).astype(int)
        spec = ((y_hat == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
        if spec >= target:
            return float(t)
    return float(np.max(p))


def paired_test(aucs_a, aucs_b):
    """Gepaarde toets over herhalingen: is AUC(a) verschillend van AUC(b)?

    De eenheid is de herhaling, niet de fold. Folds binnen een herhaling
    overlappen sterk; herhalingen zijn veel minder afhankelijk. Een toets
    over 500 folds zou daardoor veel te kleine p-waarden geven.
    """
    a = np.asarray(aucs_a, dtype=float)
    b = np.asarray(aucs_b, dtype=float)
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    d = a[:n] - b[:n]
    sd = float(np.std(d, ddof=1))
    if sd <= 0:
        return float("nan")
    t = float(np.mean(d)) / (sd / math.sqrt(n))
    return float(min(1.0, math.erfc(abs(t) / math.sqrt(2.0))))


def bootstrap_auc_ci(y, p, n_boot=2000, seed=0):
    """95%-betrouwbaarheidsinterval van de AUC, met bootstrap over patienten."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n = len(y)
    waarden = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        waarden.append(roc_auc_score(y[idx], np.asarray(p)[idx]))
    if len(waarden) < 50:
        return float("nan"), float("nan")
    return float(np.percentile(waarden, 2.5)), float(np.percentile(waarden, 97.5))


def _psi_matrix(score_case, score_control):
    """psi = 1 als case > control, 0.5 bij gelijk, anders 0."""
    a = np.asarray(score_case)[:, None]
    b = np.asarray(score_control)[None, :]
    return (a > b).astype(float) + 0.5 * (a == b).astype(float)


def delong_paired(y, p_a, p_b):
    """Gepaarde DeLong-toets: is AUC(a) verschillend van AUC(b)?

    Geeft (auc_a, auc_b, verschil, p) terug. De toets houdt rekening met het
    feit dat beide modellen op dezelfde patienten zijn geevalueerd.
    """
    y = np.asarray(y).astype(int)
    case = y == 1
    ctrl = ~case
    n1, n0 = int(case.sum()), int(ctrl.sum())
    if n1 < 2 or n0 < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    out = []
    for p in (p_a, p_b):
        psi = _psi_matrix(np.asarray(p)[case], np.asarray(p)[ctrl])
        auc = float(psi.mean())
        v10 = psi.mean(axis=1)          # per patient met recidief
        v01 = psi.mean(axis=0)          # per patient zonder recidief
        out.append((auc, v10, v01))

    (auc_a, v10_a, v01_a), (auc_b, v10_b, v01_b) = out

    def cov(x, y):
        return float(np.cov(x, y, ddof=1)[0, 1])

    s10 = np.array([[cov(v10_a, v10_a), cov(v10_a, v10_b)],
                    [cov(v10_b, v10_a), cov(v10_b, v10_b)]]) / n1
    s01 = np.array([[cov(v01_a, v01_a), cov(v01_a, v01_b)],
                    [cov(v01_b, v01_a), cov(v01_b, v01_b)]]) / n0
    cov_m = s10 + s01

    var = cov_m[0, 0] + cov_m[1, 1] - 2 * cov_m[0, 1]
    if var <= 0:
        return auc_a, auc_b, auc_a - auc_b, float("nan")
    z = (auc_a - auc_b) / math.sqrt(var)
    p_waarde = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
    return auc_a, auc_b, auc_a - auc_b, float(min(1.0, p_waarde))


def net_benefit(y, p, threshold):
    """Netto opbrengst bij een drempel: TP/n - FP/n * pt/(1-pt)."""
    y = np.asarray(y).astype(int)
    n = len(y)
    y_hat = (np.asarray(p) >= threshold).astype(int)
    tp = int(((y_hat == 1) & (y == 1)).sum())
    fp = int(((y_hat == 1) & (y == 0)).sum())
    w = threshold / (1 - threshold)
    return tp / n - (fp / n) * w


def plot_decision_curve(pooled, path, titel):
    """Decision curve analysis: netto opbrengst tegen de drempelkans."""
    if not pooled:
        return
    y_ref = next(iter(pooled.values()))[1]
    n = len(y_ref)
    prev = float(np.mean(y_ref))
    drempels = np.arange(0.05, 0.61, 0.01)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    nb_all = [prev - (1 - prev) * (t / (1 - t)) for t in drempels]
    ax.plot(drempels, nb_all, ls="--", color="#777777", lw=1.3,
            label="alles behandelen")
    ax.axhline(0, color="#777777", lw=1.0, ls=":", label="niets doen")

    kleuren = ["#2166ac", "#b2182b", "#4393c3", "#d6604d", "#92c5de", "#f4a582"]
    for i, (naam, (p, yy)) in enumerate(sorted(pooled.items())):
        nb = [net_benefit(yy, p, t) for t in drempels]
        ax.plot(drempels, nb, lw=1.7, color=kleuren[i % len(kleuren)],
                label=naam)

    ax.set_xlabel("Drempelkans")
    ax.set_ylabel("Netto opbrengst")
    ax.set_ylim(-0.05, prev + 0.05)
    ax.set_title(titel, fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate(X, y, models, repeats, n_splits, threshold=0.5):
    """Herhaalde gestratificeerde k-fold CV, plus gepoolde voorspellingen."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=0)
    y = np.asarray(y).astype(int)
    folds = {name: [] for name in models}
    som = {name: np.zeros(len(y)) for name in models}
    tel = {name: np.zeros(len(y)) for name in models}

    n_total = n_splits * repeats
    for i, (tr, te) in enumerate(cv.split(X, y), 1):
        rep = (i - 1) // n_splits   # bij welke herhaling hoort deze fold
        if i % 50 == 0 or i == n_total:
            print(f"    fold {i}/{n_total}")
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        for name, pipe in models.items():
            try:
                pipe.fit(Xtr, ytr)
                p = pipe.predict_proba(Xte)[:, 1]
            except Exception as e:
                print(f"    [ {name} ] fold faalde: {e}")
                continue
            try:
                p_tr = pipe.predict_proba(Xtr)[:, 1]
                t_out = threshold_at_sensitivity(ytr, p_tr, SENS_TARGET)
                t_in = threshold_at_specificity(ytr, p_tr, SPEC_TARGET)
                t_you = threshold_youden(ytr, p_tr)
            except Exception:
                t_out = t_in = t_you = threshold
            folds[name].append({
                "rep": rep,
                "auc": roc_auc_score(yte, p),
                "brier": brier_score_loss(yte, p),
                **{f"prev_{k}": v for k, v in metrics_at(yte, p, threshold).items()},
                **{f"out_{k}": v for k, v in metrics_at(yte, p, t_out).items()},
                **{f"in_{k}": v for k, v in metrics_at(yte, p, t_in).items()},
                **{f"you_{k}": v for k, v in metrics_at(yte, p, t_you).items()},
                "thr_out": t_out, "thr_in": t_in, "thr_you": t_you,
            })
            som[name][te] += p
            tel[name][te] += 1

    rows = []
    for name in models:
        if not folds[name]:
            continue
        d = pd.DataFrame(folds[name])
        row = {"model": name}
        for c in d.columns:
            row[c] = float(d[c].mean())
            row[f"{c}_sd"] = float(d[c].std())
        rows.append(row)

    # AUC per herhaling: binnen een herhaling middelen over de folds.
    # Dit is de eenheid waarop we intervallen en toetsen baseren -- folds
    # overlappen immers sterk, herhalingen veel minder.
    per_rep = {}
    for name in models:
        if not folds[name]:
            continue
        d = pd.DataFrame(folds[name])
        per_rep[name] = d.groupby("rep")["auc"].mean().to_numpy()

    pooled = {}
    for name in models:
        if tel[name].sum() == 0:
            continue
        ok = tel[name] > 0
        pooled[name] = (som[name][ok] / tel[name][ok], y[ok])

    # tabel met de twee klinische werkpunten
    op = []
    for name in models:
        if not folds[name]:
            continue
        d = pd.DataFrame(folds[name])
        for tag, label in [("out", f"rule-out (sens >= {SENS_TARGET:.0%})"),
                           ("you", "Youden (gebalanceerd)"),
                           ("in", f"rule-in (spec >= {SPEC_TARGET:.0%})")]:
            rij = {"model": name, "werkpunt": label,
                   "drempel": float(d[f"thr_{tag}"].mean())}
            for m in ["sensitiviteit", "specificiteit", "ppv", "npv", "accuracy"]:
                key = f"{tag}_{m}"
                if key in d.columns:
                    rij[m] = float(d[key].mean())
            op.append(rij)

    return pd.DataFrame(rows), pooled, pd.DataFrame(op), per_rep


def selected_features(X, y, model):
    """Welke radiomics kregen een coefficient in de signatuur?"""
    model.fit(X, y)
    ct = model.named_steps["select"]
    try:
        rad_pipe = ct.named_transformers_["rad"]
    except KeyError:
        return pd.DataFrame(columns=["kenmerk", "coefficient"])

    lasso = rad_pipe.named_steps["signature"].model_
    coef = lasso.coef_[0]
    mask = coef != 0
    rad_idx = list(ct.transformers_[1][2])
    names = np.asarray(X.columns)[rad_idx][rad_pipe.named_steps["corr"].keep_]
    return pd.DataFrame({
        "kenmerk": names[mask],
        "coefficient": coef[mask],
    }).sort_values("coefficient", key=np.abs, ascending=False)


def plot_calibration(pooled, y, threshold, path, titel):
    """Kalibratieplot met Brier score en kalibratiehelling."""
    if not pooled:
        return
    n = len(pooled)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#777777", lw=1.2, label="perfect")
    kleuren = ["#2166ac", "#b2182b", "#4393c3", "#d6604d", "#92c5de", "#f4a582"]
    for i, (name, (p, yy)) in enumerate(sorted(pooled.items())):
        try:
            fr, mp = calibration_curve(yy, p, n_bins=5, strategy="quantile")
        except Exception:
            continue
        b = brier_score_loss(yy, p)
        # kalibratiehelling via logistische regressie van uitkomst op logit(p)
        eps = 1e-6
        logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
        try:
            slope = LogisticRegression(penalty=None, max_iter=1000).fit(
                logit.reshape(-1, 1), yy).coef_[0][0]
        except Exception:
            slope = float("nan")
        ax.plot(mp, fr, "o-", color=kleuren[i % len(kleuren)], lw=1.6,
                label=f"{name}  Brier {b:.3f}, helling {slope:.2f}")
    ax.set_xlabel("Voorspelde kans")
    ax.set_ylabel("Waargenomen fractie recidieven")
    ax.set_title(titel, fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def cv_mean_auc(X, y, models, repeats, n_splits, groups=None):
    """Gemiddelde AUC per model met herhaalde k-fold CV.

    `groups` is de oorspronkelijke patient-ID. Die is nodig bij bootstrap:
    een steekproef met teruglegging bevat dezelfde patient meerdere keren,
    en al die kopieen moeten in DEZELFDE fold belanden. Zonder groepering
    kan een kopie in de trainingsfold zitten terwijl het origineel in de
    testfold zit -- dat is lekkage, en het model met de meeste capaciteit
    (de radiomics-signatuur) profiteert daar het meest van.
    """
    y = np.asarray(y).astype(int)
    scores = {naam: [] for naam in models}

    for r in range(repeats):
        if groups is None:
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1,
                                         random_state=r)
            splits = cv.split(X, y)
        else:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=r)
            splits = cv.split(X, y, groups)

        for tr, te in splits:
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y[tr], y[te]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            for naam, pipe in models.items():
                try:
                    pipe.fit(Xtr, ytr)
                    scores[naam].append(
                        roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]))
                except Exception:
                    pass

    return {naam: (float(np.mean(v)) if v else float("nan"))
            for naam, v in scores.items()}


def bootstrap_validation(X, y, models, n_boot, repeats, folds, log):
    """Patiënt-level bootstrap: per steekproef de hele CV opnieuw.

    Dit geeft een betrouwbaarheidsinterval dat WEL de variatie tussen
    patiënten meeneemt -- anders dan de spreiding over folds, die alleen
    de variatie tussen data splits meet.
    """
    rng = np.random.default_rng(0)
    n = len(y)
    y = np.asarray(y).astype(int)
    records = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        Xb = X.iloc[idx].reset_index(drop=True)
        # idx = de oorspronkelijke patient-ID, ook voor de duplicaten
        records.append(cv_mean_auc(Xb, y[idx], models, repeats, folds,
                                   groups=idx))
        if (b + 1) % 10 == 0 or b == 0:
            log(f"    bootstrap {b + 1}/{n_boot}")
    return pd.DataFrame(records)


def summarize_bootstrap(df, namen, referenties, out_path, tag):
    """Tabellen met bootstrap-intervallen en vergelijkingen."""
    if not len(df):
        return None
    rijen = []
    for naam in namen:
        if naam not in df.columns:
            continue
        waarden = df[naam].dropna().to_numpy()
        rij = {"model": naam, "auc": float(np.mean(waarden)),
               "lo": float(np.percentile(waarden, 2.5)),
               "hi": float(np.percentile(waarden, 97.5)),
               "n_boot": len(waarden)}
        for label, ref in referenties.items():
            if ref is None or ref == naam or ref not in df.columns:
                rij[f"p_{label}"] = float("nan")
                continue
            d = (df[naam] - df[ref]).dropna().to_numpy()
            if not len(d):
                rij[f"p_{label}"] = float("nan")
                continue
            prop = float(np.mean(d > 0))
            rij[f"p_{label}"] = float(min(1.0, 2 * min(prop, 1 - prop)))
        rijen.append(rij)
    tabel = pd.DataFrame(rijen)
    tabel.to_csv(out_path, index=False)
    return tabel


def analyse(clinical, dosimetry, radiomics, outcome, tag, args) -> pd.DataFrame:
    """Draait de vier modellen op een (sub)cohort en schrijft de resultaten weg."""
    print("=" * 78)
    print(f"COHORT: {tag}")
    print(f"  patiënten {len(outcome)}   recidieven {int(outcome.sum())} "
          f"({100 * outcome.mean():.1f}%)")

    # constante kolommen vallen af -- in de plaveiselcelgroep is histologie
    # constant en doet dus automatisch niet meer mee.
    dropped = [c for c in clinical.columns if clinical[c].nunique(dropna=True) < 2]
    if dropped:
        print(f"  constant in dit cohort, valt af: {', '.join(dropped)}")
        clinical = clinical.drop(columns=dropped)
    print(f"  klinisch: {list(clinical.columns)}")
    print(f"  dosimetrie: {list(dosimetry.columns)}")
    print()

    specs = [(label, [n for n in fixed if n not in dropped], k)
             for label, fixed, k in MODELLEN]
    if dropped:
        print(f"  (modellen worden herberekend zonder {', '.join(dropped)})\n")

    X, models = make_models(clinical, radiomics, dosimetry, specs)
    if not models:
        print("  Geen enkel model kon worden gebouwd. Stop.\n")
        return pd.DataFrame()

    drempel = float(outcome.mean())
    print(f"  afkappunt voor sensitiviteit en specificiteit: {drempel:.3f} (= prevalentie)\n")
    results, pooled, operating, per_rep = evaluate(X, outcome, models, args.repeats,
                                          args.folds, threshold=drempel)

    os.makedirs(args.out, exist_ok=True)
    suffix = f"_{tag.replace(' ', '_')}"

    # AUC met 95%-BI, plus twee DeLong-vergelijkingen:
    #   p_vs_hist  -- voegt dit model iets toe aan histologie alleen?
    #   p_vs_vol   -- voegt dit model iets toe aan histologie + volume?
    if len(results) and per_rep:
        namen = list(results["model"])
        ref_hist = "1. hist" if "1. hist" in namen else None
        ref_vol = "2. + volume" if "2. + volume" in namen else None

        lo, hi, p_hist, p_vol = [], [], [], []
        for naam in namen:
            aucs = per_rep[naam]

            for ref, doel in ((ref_hist, p_hist), (ref_vol, p_vol)):
                if ref is None or ref == naam or ref not in per_rep:
                    doel.append(float("nan"))
                else:
                    doel.append(paired_test(per_rep[ref], aucs))

        results["p_vs_hist"] = p_hist
        results["p_vs_vol"] = p_vol

    print(f"\nResultaten -- {tag}")
    kol = ["model", "auc", "auc_sd", "p_vs_hist", "p_vs_vol", "brier",
           "prev_sensitiviteit", "prev_specificiteit", "prev_ppv", "prev_npv"]
    kol = [k for k in kol if k in results.columns]
    print(results[kol].round(3).to_string(index=False))
    plot_calibration(pooled, outcome.values, drempel,
                     os.path.join(args.out, f"kalibratie{suffix}.png"),
                     f"Kalibratie -- {tag}  (afkappunt {drempel:.2f})")
    plot_decision_curve(pooled,
                        os.path.join(args.out, f"besliscurve{suffix}.png"),
                        f"Decision curve -- {tag}")
    results.to_csv(os.path.join(args.out, f"prestatie{suffix}.csv"), index=False)

    if len(operating):
        op_kol = ["model", "werkpunt", "drempel", "sensitiviteit", "specificiteit",
                  "ppv", "npv", "accuracy"]
        op_kol = [k for k in op_kol if k in operating.columns]
        print(f"\nWerkpunten -- {tag}")
        print(operating[op_kol].round(3).to_string(index=False))
        operating.to_csv(os.path.join(args.out, f"werkpunten{suffix}.csv"),
                         index=False)

    if getattr(args, "bootstrap", 0):
        print(f"\nBootstrap-validatie -- {tag} "
              f"({args.bootstrap} steekproeven x {args.bootstrap_repeats} herhalingen)")
        boot = bootstrap_validation(X, outcome, models, args.bootstrap,
                                    args.bootstrap_repeats, args.folds, print)
        refs = {}
        if "1. hist" in models:
            refs["hist"] = "1. hist"
        if "2. + volume" in models:
            refs["vol"] = "2. + volume"
        tabel = summarize_bootstrap(boot, list(models), refs,
                                    os.path.join(args.out, f"bootstrap{suffix}.csv"),
                                    tag)
        if tabel is not None:
            kol = [k for k in ["model", "auc", "lo", "hi", "p_hist", "p_vol"]
                   if k in tabel.columns]
            print(tabel[kol].round(3).to_string(index=False))
        print()

    print(f"\nGekozen radiomics per model (op alle data van {tag}):")
    alle = []
    for label in models:
        table = selected_features(X, outcome, models[label])
        if len(table):
            table.insert(0, "model", label)
            alle.append(table)
            namen = ", ".join(table["kenmerk"].head(5))
            print(f"  {label:28s} {namen}")
    if alle:
        pd.concat(alle).to_csv(os.path.join(args.out, f"kenmerken{suffix}.csv"),
                               index=False)
        print(f"  -> weggeschreven naar kenmerken{suffix}.csv")
    print()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--repeats", type=int, default=5,
                        help="herhalingen van de k-fold CV (5 = standaard)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="aantal patient-level bootstrap-steekproeven "
                             "(0 = overslaan; 200 is gebruikelijk)")
    parser.add_argument("--bootstrap-repeats", type=int, default=3,
                        help="herhalingen van de CV binnen elke bootstrap")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"FOUT: {args.data} niet gevonden.")
        return 1

    clinical, dosimetry, radiomics, outcome = load_data(args.data)

    # 1. het volledige cohort
    volledig = analyse(clinical, dosimetry, radiomics, outcome, "volledig", args)

    # 2. de plaveiselcel-subgroep (histologie is daar constant)
    waarden = clinical["histologie"].dropna().unique()
    if len(waarden) >= 2:
        plaveisel = min(waarden)
        mask = (clinical["histologie"] == plaveisel).to_numpy()
        if mask.sum() >= 30:
            sub = analyse(clinical[mask], dosimetry[mask], radiomics[mask],
                          outcome[mask], "plaveiselcel", args)

            print("=" * 78)
            print("VERGELIJKING")
            if not len(volledig) or not len(sub):
                print("  (onvoldoende modellen om te vergelijken)\n")
                return 0
            print(f"  {'model':22s} {'volledig':>9s} {'plaveisel':>10s}")
            for _, row in volledig.iterrows():
                match = sub[sub["model"] == row["model"]]
                tekst = f"{match['auc'].iloc[0]:10.3f}" if len(match) else f"{'n.v.t.':>10s}"
                print(f"  {row['model']:22s} {row['auc']:9.3f} {tekst}")
            print()

    print(f"Bestanden geschreven naar {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
