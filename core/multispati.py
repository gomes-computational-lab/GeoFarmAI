import numpy as np
import pandas as pd
import subprocess
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from geofarmai.exceptions import MultispatiUnavailableError, RMultispatiUnavailableError

R_AVAILABLE = None


def standardized_predictors(X: pd.DataFrame):
    """Return the existing standardized, unreduced predictor representation."""

    scaler = StandardScaler()
    values = scaler.fit_transform(X)
    return pd.DataFrame(values, index=X.index, columns=X.columns), scaler


def pca_components(X: pd.DataFrame, n_components=3):
    """Run the existing standardized PCA pathway and retain fitted metadata."""

    standardized, scaler = standardized_predictors(X)
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(standardized.to_numpy())
    scores = pd.DataFrame(
        Z,
        index=X.index,
        columns=[f"PC{i+1}" for i in range(Z.shape[1])],
    )
    return scores, pca, scaler


def python_multispati_components(
    X: pd.DataFrame,
    connectivity,
    n_components=3,
    random_state=42,
):
    """Run the established ``multispaeti.MultispatiPCA`` implementation.

    Importing the optional scientific implementation is deliberately lazy so
    importing GeoFarmAI never initializes an unavailable decomposition engine.
    An explicit request never falls back to ordinary PCA.
    """

    try:
        from multispaeti import MultispatiPCA
    except Exception as exc:
        raise MultispatiUnavailableError(
            "Python MULTISPATI was explicitly requested, but the 'multispaeti' "
            "package is unavailable or could not be initialized. Install the "
            "GeoFarmAI base scientific dependencies and retry."
        ) from exc

    standardized, scaler = standardized_predictors(X)
    model = MultispatiPCA(
        n_components=n_components,
        connectivity=connectivity,
        random_state=random_state,
    )
    try:
        Z = model.fit_transform(standardized.to_numpy())
    except Exception as exc:
        raise MultispatiUnavailableError(
            "Python MULTISPATI was explicitly requested but execution failed. "
            "Check the multispaeti installation and spatial connectivity settings."
        ) from exc
    scores = pd.DataFrame(
        Z,
        index=X.index,
        columns=[f"SPC{i+1}" for i in range(Z.shape[1])],
    )
    return scores, model, scaler


def _preflight_r_multispati():
    probe = (
        "import rpy2.robjects as ro; "
        "ro.r('suppressPackageStartupMessages(library(ade4)); "
        "suppressPackageStartupMessages(library(spdep)); "
        "suppressPackageStartupMessages(library(adespatial))'); "
        "ro.r('1 + 1')"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            text=True,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RMultispatiUnavailableError(
            "R-backed MULTISPATI was explicitly requested, but its optional "
            "R/rpy2 preflight could not complete."
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        suffix = f" Last error: {detail[-1]}" if detail else ""
        raise RMultispatiUnavailableError(
            "R-backed MULTISPATI was explicitly requested, but its optional "
            f"R/rpy2 runtime failed preflight (exit {completed.returncode}).{suffix}"
        )


def multispati_components(X: pd.DataFrame, W, n_components=3, use_r=True):
    global R_AVAILABLE
    if not use_r:
        scores, _, _ = pca_components(X, n_components=n_components)
        return scores.reset_index(drop=True), False

    _preflight_r_multispati()
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.conversion import localconverter

        R_AVAILABLE = True
    except Exception as exc:
        R_AVAILABLE = False
        raise RMultispatiUnavailableError(
            "R-backed MULTISPATI was explicitly requested but could not be initialized. "
            "R support is optional; install GeoFarmAI with the 'r' extra and configure "
            "a compatible R_HOME with the ade4, spdep, and adespatial R packages."
        ) from exc

    # R MULTISPATI via ade4
    try:
        ro.r('suppressPackageStartupMessages(library(ade4)); '
              'suppressPackageStartupMessages(library(spdep)); '
              'suppressPackageStartupMessages(library(adespatial))')
        # Build spatial weights listw in R from Python KNN matrix
        converter = ro.default_converter + pandas2ri.converter + numpy2ri.converter

        with localconverter(converter):
            ro.globalenv['X'] = ro.conversion.py2rpy(X.astype(float))
        Wmat = np.asarray(W.full()[0], dtype=float)
        with localconverter(converter):
            ro.globalenv['W'] = ro.conversion.py2rpy(Wmat)
        ro.r('Wlist <- mat2listw(W, style="W")')
        ro.r('ac <- dudi.pca(scale(X), scannf=FALSE, nf=%d)' % n_components)
        ro.r('ms <- adespatial::multispati(ac, Wlist, scannf=FALSE)')
        with localconverter(ro.default_converter + pandas2ri.converter):
            Z = ro.conversion.rpy2py(ro.r('as.data.frame(ms$li)'))
        Z.columns = [f"SPC{i+1}" for i in range(Z.shape[1])]
        print("Using R MULTISPATI")
        return Z, True
    except Exception as exc:
        raise RMultispatiUnavailableError(
            "R-backed MULTISPATI was explicitly requested but execution failed. "
            "Verify R_HOME and the optional ade4, spdep, and adespatial R packages."
        ) from exc
