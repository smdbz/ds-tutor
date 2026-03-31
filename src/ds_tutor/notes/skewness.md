Skewness measures asymmetry of a distribution.
A skew of 0 is perfectly symmetric (e.g. a Gaussian).
|skew| > 1 indicates heavy tails that can distort linear
models and distance-based algorithms (KNN, SVM, PCA).
Tree-based models (RandomForest, XGBoost) are invariant
to monotonic transformations and do NOT need this step.