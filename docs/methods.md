# Methods

## Raster-first workflow

Configured predictor and yield variables are ordinary-kriged with PyKrige, intersected and resampled to a common raster grid, and standardized. Features are represented by Python MULTISPATI-PCA when installed and selected, standard PCA as the historical fallback, or raw standardized raster values. A K-nearest-neighbor pixel connectivity graph supports spatial components and constrained agglomerative clustering.

## Vector/grid-cell workflow

Points are projected to a local UTM CRS when configured, reconciled to an adaptive rectangular grid using IDW, nearest-neighbor, or buffer means, and transformed with optional R MULTISPATI or standard PCA. This pathway is retained alongside raster processing.

## Clustering and selection

K-means, agglomerative Ward clustering, full-covariance Gaussian mixture models, and fuzzy C-means retain their original parameters. Seeded algorithms iterate configured seeds. Candidate metrics retain the output columns `asc`, `ch_score`, `fpc` where applicable, `vr`, and `anova_p`. The selected solution maximizes yield variance reduction and then silhouette score; the raster workflow also exports the best Calinski–Harabasz solution.

## Agronomic validation

Variance reduction compares sample yield variance with a zone-size-weighted average of within-zone sample variances. ANOVA uses an ordinary least-squares categorical-zone model and a type-II ANOVA table. These formulas were migrated without alteration.
