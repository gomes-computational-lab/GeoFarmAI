import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

R_AVAILABLE = None

from .exceptions import OptionalDependencyError


def multispati_components(X: pd.DataFrame, W, n_components=3, use_r=True):
    """Compute components using R MULTISPATI when explicitly requested, otherwise PCA.

    The historical vector workflow used standard PCA whenever ``use_r`` was
    false. That behavior is preserved. An unavailable explicitly requested R
    pathway now raises a diagnostic error instead of silently changing methods.
    """
    global R_AVAILABLE
    # Fallback: standard PCA if R not available
    if use_r and R_AVAILABLE is not False:
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri, numpy2ri
            from rpy2.robjects.conversion import localconverter

            R_AVAILABLE = True
        except Exception as exc:
            R_AVAILABLE = False
            raise OptionalDependencyError(
                "R-backed MULTISPATI was requested, but rpy2 and a working R installation "
                "with ade4, spdep, and adespatial are required. Install the 'r' optional "
                "dependencies or set spatial_pca.use_r_multispati to false."
            ) from exc

    if not use_r or not R_AVAILABLE:
        Z = PCA(n_components=n_components).fit_transform(StandardScaler().fit_transform(X))
        return pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(Z.shape[1])]), False

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
        raise OptionalDependencyError(
            "R-backed MULTISPATI failed. Confirm that R packages ade4, spdep, and "
            "adespatial are installed and compatible."
        ) from exc
