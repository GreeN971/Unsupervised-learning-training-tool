"""Configuration for the OCSVM anomaly detector."""

# OCSVM-specific parameters.
# Expected fraction of anomalies; sets the score quantile used as the threshold.
CONTAMINATION = 0.1
# Kernel mapping inputs into a higher-dimensional space where a hyperplane can
# separate the dense "normal" region from the rest.
KERNEL = "rbf"      # "linear", "poly", "rbf" or "sigmoid"
# Upper bound on the training-outlier fraction and lower bound on the support-
# vector fraction; directly controls how tight the decision boundary is.
NU = 0.5
# RBF bandwidth = radius of influence of each sample ("auto" = 1/n_features);
# small gamma = smooth boundary, large gamma = tight around points.
GAMMA = "auto"
# Polynomial-kernel degree; only used when KERNEL = "poly".
DEGREE = 3          # only used by the "poly" kernel

# Train/test split.
# Fraction of samples held out to estimate out-of-sample (generalization) error.
TEST_SIZE = 0.2
# Seed fixing every pseudo-random draw (split, subsampling, init) for reproducibility.
RANDOM_STATE = 42

# Keras surrogate network (hidden layer sizes, up to 3 -> max 4 layers total).
# Layer widths set model capacity: wider/deeper lowers bias but raises variance.
LAYERS = [16, 8]
# Gradient-descent step size: convergence speed vs. optimization stability.
LEARNING_RATE = 0.001
# Passes over the training set: too few underfits, too many overfits.
EPOCHS = 50
# Samples per gradient estimate: larger batches give lower-variance updates.
BATCH_SIZE = 32
# Log verbosity only (0 silent, 1 progress); no statistical effect.
VERBOSE = 1
