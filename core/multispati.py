import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

R_AVAILABLE = None


def multispati_components(X: pd.DataFrame, W, n_components=3, use_r=True):
    global R_AVAILABLE
    # Fallback: standard PCA if R not available
    if use_r and R_AVAILABLE is not False:
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri, numpy2ri
            from rpy2.robjects.conversion import localconverter

            R_AVAILABLE = True
        except Exception:
            R_AVAILABLE = False

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
        Z = PCA(n_components=n_components).fit_transform(StandardScaler().fit_transform(X))
        print(f"Falling back to PCA: {exc!r}")
        return pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(Z.shape[1])]), False
