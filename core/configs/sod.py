"""Configuration for the SOD anomaly detector."""

# SOD-specific parameters (ref_set must be smaller than n_neighbors).
# Expected fraction of anomalies; sets the score quantile used as the threshold.
CONTAMINATION = 0.1
# Neighborhood size used to build each point's shared-nearest-neighbor set.
N_NEIGHBORS = 20
# Number of shared nearest neighbors forming the reference set that defines the
# local subspace (must be < N_NEIGHBORS).
REF_SET = 10
# Fraction of variance retained when selecting the relevant subspace; low-variance
# axes are the directions the outlierness is measured along.
ALPHA = 0.8

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
