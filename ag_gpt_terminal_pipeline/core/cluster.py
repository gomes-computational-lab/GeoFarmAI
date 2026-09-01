import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import skfuzzy as fuzz


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
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return labels, _cluster_quality(X, labels, k, sample_size)


def run_agglomerative(X, k, connectivity=None, sample_size=None):
    model = AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=connectivity)
    labels = model.fit_predict(X)
    return labels, _cluster_quality(X, labels, k, sample_size)


def run_gmm(X, k, random_state=42, sample_size=None):
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
    labels = gmm.fit_predict(X)
    return labels, _cluster_quality(X, labels, k, sample_size)


def run_fcm(X, k, random_state=None, sample_size=None):
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
