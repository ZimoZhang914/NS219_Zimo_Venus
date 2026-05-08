"""
Supervised cell-state classifier on the VAE latent space.

We train a calibrated classifier on cells that have a reference label
(e.g. published microglia subsets like *Homeostatic*, *DAM*, *IRM*,
*Cycling*, *PVM*) and predict for all microglia.

Hyperparameters are tuned with Optuna; predictions are calibrated with
isotonic regression on a held-out set so the confidence threshold is
meaningful.

Following the scNym (Kimmel & Kelley 2021) and scArches/scANVI
(Lotfollahi et al. 2022) pattern of using a reference-trained classifier
on top of a learned embedding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class ClassifierArtifacts:
    model: Any
    label_encoder: LabelEncoder
    classes_: np.ndarray
    metrics: Dict[str, Any]
    best_params: Dict[str, Any]


# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------


def _maybe_smote(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        log.warning("imblearn not installed — skipping SMOTE")
        return X, y

    counts = pd.Series(y).value_counts()
    if counts.min() < 6 or len(counts) < 2:
        log.warning("class too small for SMOTE — skipping (min=%d)", counts.min())
        return X, y
    k = min(5, int(counts.min()) - 1)
    try:
        sm = SMOTE(random_state=seed, k_neighbors=k)
        Xr, yr = sm.fit_resample(X, y)
        log.info("SMOTE: %d -> %d samples", len(y), len(yr))
        return Xr, yr
    except Exception as e:
        log.warning("SMOTE failed (%s) — using original data", e)
        return X, y


def _build_model(name: str, params: Dict[str, Any], seed: int):
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            random_state=seed,
            tree_method="hist",
            eval_metric="mlogloss",
            **params,
        )
    if name == "logistic":
        return LogisticRegression(
            random_state=seed,
            max_iter=2000,
            **params,
        )
    if name == "svc":
        return LinearSVC(random_state=seed, dual="auto", **params)
    raise ValueError(f"unknown classifier model: {name}")


def _suggest_params(trial, name: str) -> Dict[str, Any]:
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
    if name == "logistic":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
        }
    if name == "svc":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    raise ValueError(name)


# -----------------------------------------------------------------------
# training
# -----------------------------------------------------------------------


def train_classifier(
    latent: np.ndarray,
    labels: pd.Series,
    cfg: Config,
) -> Tuple[ClassifierArtifacts, np.ndarray]:
    """
    Train a classifier on labeled cells in the latent space.

    ``labels`` is a pandas Series aligned to ``latent`` rows; cells with
    NaN / 'Unknown' are excluded from training but still get predictions.

    Returns artifacts and a boolean mask marking the labeled cells.
    """
    cc = cfg.classifier
    seed = cfg.seed

    # mask of labeled cells
    lbl = labels.astype(str).str.strip()
    valid = (~lbl.isna()) & (lbl != "") & (~lbl.str.lower().isin(["nan", "unknown", "na"]))
    labeled_mask = valid.to_numpy()
    if labeled_mask.sum() < 50:
        raise ValueError(
            f"Not enough labeled cells to train ({int(labeled_mask.sum())}). "
            f"Set classifier.reference_label_col to a column with cell-state labels."
        )

    X = latent[labeled_mask]
    y_str = lbl[labeled_mask].to_numpy()
    le = LabelEncoder().fit(y_str)
    y = le.transform(y_str)

    log.info("Training classifier on %d cells, %d classes: %s",
             X.shape[0], len(le.classes_), le.classes_.tolist())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cc.test_fraction, random_state=seed, stratify=y
    )

    if cc.use_smote:
        X_tr, y_tr = _maybe_smote(X_tr, y_tr, seed)

    # ---------- HPO -------------------------------------------------------
    best_params: Dict[str, Any] = {}
    if cc.use_optuna:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                params = _suggest_params(trial, cc.model)
                model = _build_model(cc.model, params, seed)
                model.fit(X_tr, y_tr)
                return f1_score(y_te, model.predict(X_te), average="macro")

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=cc.optuna_trials, show_progress_bar=False)
            best_params = study.best_params
            log.info("Optuna best macro-F1=%.3f params=%s", study.best_value, best_params)
        except ImportError:
            log.warning("optuna not installed — using defaults")

    base = _build_model(cc.model, best_params, seed)

    # ---------- calibration ----------------------------------------------
    # LinearSVC has no predict_proba; calibrate via sigmoid. Tree/logistic via isotonic.
    method = "sigmoid" if cc.model == "svc" else "isotonic"
    model = CalibratedClassifierCV(base, method=method, cv=3)
    model.fit(X_tr, y_tr)

    # ---------- evaluation -----------------------------------------------
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)
    metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "kappa": float(cohen_kappa_score(y_te, y_pred)),
        "macro_f1": float(f1_score(y_te, y_pred, average="macro")),
        "per_class": classification_report(
            y_te, y_pred, target_names=le.classes_, output_dict=True, zero_division=0
        ),
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "mean_top1_confidence": float(y_proba.max(axis=1).mean()),
    }
    log.info("Test acc=%.3f kappa=%.3f macroF1=%.3f",
             metrics["accuracy"], metrics["kappa"], metrics["macro_f1"])

    artifacts = ClassifierArtifacts(
        model=model,
        label_encoder=le,
        classes_=le.classes_,
        metrics=metrics,
        best_params=best_params,
    )
    return artifacts, labeled_mask


# -----------------------------------------------------------------------
# inference
# -----------------------------------------------------------------------


def predict(
    art: ClassifierArtifacts,
    latent: np.ndarray,
    confidence_threshold: float = 0.7,
) -> pd.DataFrame:
    """Predict labels and confidence; flag low-confidence as 'Uncertain'."""
    proba = art.model.predict_proba(latent)
    top1 = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    pred = art.label_encoder.inverse_transform(top1)
    flagged = np.where(conf >= confidence_threshold, pred, "Uncertain")

    df = pd.DataFrame(
        {
            "predicted_state": flagged,
            "predicted_state_raw": pred,
            "confidence": conf,
        }
    )
    # also store full probability matrix
    for j, cls in enumerate(art.classes_):
        df[f"prob_{cls}"] = proba[:, j]
    return df
