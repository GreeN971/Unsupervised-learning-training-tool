"""Configuration for the COF anomaly detector."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_REPO_ROOT / "converter"))

# Location of the input CSV file (rows = samples/observations, columns = features).
CSV_PATH = _REPO_ROOT / "datasets" / "dataWithoutIdCol.csv"

# Feature columns modeled; each is a dimension of the multivariate distribution
# whose geometry/density the detector uses to separate normal from anomalous.
FEATURES = ["temperature"]

# COF-specific parameters.
# Expected fraction of anomalies; sets the score quantile used as the threshold.
CONTAMINATION = 0.1
# Neighborhood size for the chaining-distance (connectivity-based) local density
# estimate; larger k smooths the estimated density.
N_NEIGHBORS = 20
# Compute strategy (speed vs. memory); no statistical effect.
METHOD = "memory"  # "fast" or "memory"

# Train/test split.
# Fraction of samples held out to estimate out-of-sample (generalization) error.
TEST_SIZE = 0.2
# Seed fixing every pseudo-random draw (split, subsampling, init) for reproducibility.
RANDOM_STATE = 42

# Range the anomaly scores are scaled into.
# Min-max target interval so every detector emits comparable [0, 1] scores
# regardless of its native (unbounded) score scale.
SCORE_RANGE = (0.0, 1.0)

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
