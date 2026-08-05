import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import skfuzzy as fuzz

from .exceptions import ConfigurationError


SUPPORTED_ALGORITHMS = ("kmeans", "agglomerative", "gmm", "fcm")


def _validate_input(X, k):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ConfigurationError("Clustering input must be a two-dimensional matrix.")
    if not isinstance(k, (int, np.integer)) or k < 2:
        raise ConfigurationError("k must be an integer greater than or equal to 2.")
    if k >= X.shape[0]:
        raise ConfigurationError(
            f"k must be smaller than the number of samples ({X.shape[0]}); received {k}."
        )
    return X


def _silhouette(X, labels, k, sample_size=None):
    if X.shape[0] <= k:
        return np.nan
    if sample_size is not None and X.shape[0] > sample_size:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        return float(silhouette_score(X[idx], labels[idx]))
    return float(silhouette_score(X, labels))


def _calinski_harabasz(X, labels, k):
    unique_labels = np.unique(labels)
    if X.shape[0] <= k or len(unique_labels) < 2:
        return np.nan
    return float(calinski_harabasz_score(X, labels))


def _cluster_quality(X, labels, k, sample_size=None):
    return {
        "asc": _silhouette(X, labels, k, sample_size),
        "ch_score": _calinski_harabasz(X, labels, k),
    }


def run_kmeans(X, k, random_state=42, sample_size=None):
    X = _validate_input(X, k)
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return labels, _cluster_quality(X, labels, k, sample_size)


def run_agglomerative(X, k, connectivity=None, sample_size=None):
    X = _validate_input(X, k)
    model = AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=connectivity)
    labels = model.fit_predict(X)
    return labels, _cluster_quality(X, labels, k, sample_size)


def run_gmm(X, k, random_state=42, sample_size=None):
    X = _validate_input(X, k)
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
    labels = gmm.fit_predict(X)
    return labels, _cluster_quality(X, labels, k, sample_size)


def run_fcm(X, k, random_state=None, sample_size=None):
    X = _validate_input(X, k)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X.T,
        c=k,
        m=2.0,
        error=0.005,
        maxiter=1000,
        init=None,
        seed=random_state,
    )
    labels = u.argmax(axis=0)
    return labels, {**_cluster_quality(X, labels, k, sample_size), 'fpc': float(fpc)}


def cluster(X, algorithm, k, random_state=42, connectivity=None, sample_size=None):
    """Run one of the supported clustering algorithms without changing its implementation."""
    if algorithm == "kmeans":
        return run_kmeans(X, k, random_state=random_state, sample_size=sample_size)
    if algorithm == "agglomerative":
        return run_agglomerative(X, k, connectivity=connectivity, sample_size=sample_size)
    if algorithm == "gmm":
        return run_gmm(X, k, random_state=random_state, sample_size=sample_size)
    if algorithm == "fcm":
        return run_fcm(X, k, random_state=random_state, sample_size=sample_size)
    raise ConfigurationError(
        f"Unsupported clustering method '{algorithm}'. Supported methods: {', '.join(SUPPORTED_ALGORITHMS)}."
    )
