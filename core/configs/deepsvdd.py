"""Configuration for the DeepSVDD anomaly detector."""

# DeepSVDD-specific parameters (its own network, trained on PyTorch).
# Expected fraction of anomalies; sets the score quantile used as the threshold.
CONTAMINATION = 0.1
# If True, use an autoencoder variant; else a plain encoder onto the hypersphere.
USE_AE = False
# Encoder layer widths mapping inputs to the representation space whose distance
# to the learned center is the anomaly score.
HIDDEN_NEURONS = [64, 32]
# Hidden-layer nonlinearity, enabling a non-linear "normal" region boundary.
HIDDEN_ACTIVATION = "relu"
# Activation of the representation (output) layer.
OUTPUT_ACTIVATION = "sigmoid"
# Optimizer minimizing the mean squared distance to the hypersphere center.
OPTIMIZER = "adam"
# Passes over the data when training the detector; too few underfits, too many overfits.
DSVDD_EPOCHS = 100
# Samples per gradient estimate; larger batches give lower-variance updates.
DSVDD_BATCH_SIZE = 32
# Dropout probability: randomly zeros units during training to reduce overfitting.
DROPOUT_RATE = 0.2
# L2 weight-decay strength: shrinks weights toward zero (variance reduction; also
# discourages the trivial all-zero collapse of the representation).
L2_REGULARIZER = 0.5e-6

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
