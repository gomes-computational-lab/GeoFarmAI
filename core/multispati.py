import numpy as np
import pandas as pd
import subprocess
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from geofarmai.exceptions import RMultispatiUnavailableError

R_AVAILABLE = None


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
        Z = PCA(n_components=n_components).fit_transform(StandardScaler().fit_transform(X))
        return pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(Z.shape[1])]), False

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
