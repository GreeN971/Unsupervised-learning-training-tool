"""Configuration for the PCA anomaly detector."""

# PCA-specific parameters (N_COMPONENTS=None keeps all components).
# Expected fraction of anomalies; sets the score quantile used as the threshold.
CONTAMINATION = 0.1
# Number of principal components (eigenvectors) spanning the "normal" subspace;
# None keeps all, so the score becomes the full Mahalanobis distance.
N_COMPONENTS = None
# If True, rescale each component to unit variance before scoring.
WHITEN = False
# Weight each component's squared deviation by its eigenvalue (1/variance), i.e.
# a Mahalanobis-style score so low-variance directions count more.
WEIGHTED = True
# z-score features first so components are not dominated by high-variance features
# (makes the score scale-invariant).
STANDARDIZATION = True

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
