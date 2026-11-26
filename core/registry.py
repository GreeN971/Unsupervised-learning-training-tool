"""Central metadata for every algorithm.

Each entry names the ``core.algorithms`` module that runs it and the
parameter schema that drives the GUI's config form: label, type, bounds,
and the tooltip description (English translation of the thesis chapter
on PyOD parameter descriptions where one exists, otherwise lifted from the
corresponding ``core/configs/<name>.py`` comments).

Known simplification: RANDOM_STATE, LAYERS/HIDDEN_NEURONS and
VERBOSE are left as fixed internal defaults rather than exposed as GUI
params, since list-valued and log-only settings don't fit the current
scalar param types (int/float/bool/choice). Min/max bounds below are
usability caps to sanity-check, not hard ML constraints.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from converter.to_tflite import n_model_inputs
from core import notices, temp_csv
from core.algorithms import (
    cblof as _cblof,
    cof as _cof,
    copod as _copod,
    deepsvdd as _deepsvdd,
    ecod as _ecod,
    iforest as _iforest,
    knn as _knn,
    loda as _loda,
    lof as _lof,
    ocsvm as _ocsvm,
    pca as _pca,
    sod as _sod,
)


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    type: str  # "int" | "float" | "bool" | "checkbox" | "choice"
    default: Any
    description: str
    min: Optional[float] = None
    max: Optional[float] = None
    # A list shows the raw values; a dict maps value -> display label.
    choices: Optional[list | dict] = None
    # Shown in bold under the description in the info tooltip.
    warning: Optional[str] = None


@dataclass(frozen=True)
class AlgorithmEntry:
    key: str
    display_name: str
    run: Callable
    params: list = field(default_factory=list)
    # What the algorithm can't do, whatever the data or settings (shown on
    # the Configure page).
    limitations: tuple = ()
    # checks(params, staged_summary) -> [Notice]: problems with the staged
    # data and the chosen params; see core.notices.
    checks: Optional[Callable] = None
    # Fewest feature columns it can be trained on (core.common.TrainingSpec).
    min_columns: int = 1

    def default_params(self) -> dict:
        return {p.key: p.default for p in self.params}

    def detector_params(self) -> list:
        """The PyOD detector's own params (and the shared test split), in
        order: everything but the Keras surrogate's.
        """
        return [p for p in self.params if p.key not in _SURROGATE_KEYS]

    def surrogate_params(self) -> list:
        """The Keras surrogate network's params (training and export)."""
        return [p for p in self.params if p.key in _SURROGATE_KEYS]

    def notices(self, params: Optional[dict] = None) -> list:
        """Notices for ``params`` (defaults filled in) and the staged CSV:
        the common checks followed by this algorithm's own (see
        core.notices). Blocking ones first.
        """
        params = {**self.default_params(), **(params or {})}
        data = temp_csv.staged_summary()
        found = notices.common_checks(params, data, self.min_columns)
        if self.checks is not None:
            found += self.checks(params, data)
        return sorted(found, key=lambda notice: not notice.blocking)

    def n_inputs(self) -> Optional[int]:
        """Network inputs for the features it will be trained on: the
        staged CSV's columns (those the CSV wizard included), as a value
        each with one column and value/flag pairs with several. None when
        nothing is staged.
        """
        columns = temp_csv.staged_columns()
        return n_model_inputs(len(columns)) if columns else None


_CONTAMINATION = Param(
    "CONTAMINATION", "Contamination", "float", 0.1,
    "Expected proportion of anomalies (outliers) in the dataset, given as "
    "a fraction in the range (0, 0.5] - e.g. 0.1 means roughly 10% of "
    "samples are expected to be anomalous. PyOD uses it only to set the "
    "decision threshold; here the detector's training-score quantile "
    "1 - contamination is mapped to the exported model's fixed anomaly "
    "threshold of 0.5 [26].",
    min=0.01, max=0.5,
    warning="Exactly this share of the training data scores above 0.5. With "
    "0.1 the 10% least common training values are flagged, even if they are "
    "ordinary readings (e.g. the warmest days of a month) - lower it to flag "
    "only real extremes",
)
_TEST_SIZE = Param(
    "TEST_SIZE", "Test split fraction", "float", 0.2,
    "Fraction of samples held out to estimate out-of-sample "
    "(generalization) error. The same split is used to train and evaluate "
    "the Keras surrogate network.",
    min=0.05, max=0.5,
)
_SURROGATE_PARAMS = [
    Param(
        "EPOCHS", "Training epochs", "int", 50,
        "Passes over the training set when learning the exported surrogate "
        "network: too few underfits, too many overfits.",
        min=1, max=1000,
    ),
    Param(
        "LEARNING_RATE", "Learning rate", "float", 0.001,
        "Gradient-descent step size: convergence speed vs. optimization "
        "stability.",
        min=0.00001, max=1.0,
    ),
    Param(
        "BATCH_SIZE", "Batch size", "int", 32,
        "Samples per gradient estimate: larger batches give lower-variance "
        "updates.",
        min=1, max=1024,
    ),
    Param(
        "LOSS", "Loss function", "choice", "mse",
        "Keras loss the surrogate network minimizes while learning to "
        "reproduce the detector's anomaly scores.",
        choices={"mse": "Mean Squared Error"},
    ),
    # Not "OPTIMIZER": DeepSVDD already uses that key for its detector's own
    # optimizer, separate from this surrogate network's.
    Param(
        "SURROGATE_OPTIMIZER", "Training optimizer", "choice", "adam",
        "Keras optimizer that updates the surrogate network's weights, "
        "using the learning rate above.",
        choices={"adam": "Adam", "sgd": "SGD", "rmsprop": "RMSprop"},
    ),
    Param(
        "QUANTIZATION", "Quantization", "choice", "int8",
        "Post-training quantization of the exported .tflite. Int8 quantizes "
        "weights, activations, input and output: weights shrink ~4x and "
        "inference runs faster, at a small accuracy cost; activation ranges "
        "are calibrated on the training data. The accuracy loss against the "
        "unquantized network is measured on the test split after export.",
        choices={
            "none": "None (float32)",
            "int8": "Int8 (full integer)",
        },
        warning="The int8 model expects int8-quantized input and returns int8 "
        "output; the scale and zero point are saved in <name>_params.json",
    ),
    Param(
        "INPUT_MODE", "Model input", "choice", "raw",
        "What the exported .tflite takes. Raw: real sensor values - the "
        "min/max normalization is folded into the first layer's weights "
        "(same outputs, no extra layers or MAC), so Edge Impulse and the "
        "device feed measurements directly. Normalized: values scaled to "
        "[0, 1] with input_min/input_max from <name>_params.json, which the "
        "device has to compute itself.",
        choices={
            "raw": "Raw values",
            "normalized": "Normalized [0, 1]",
        },
        warning="With int8 and several features whose ranges differ a lot, "
        "raw input loses resolution (one int8 input scale is shared); the "
        "run report warns when that happens",
    ),
]
# Shown in their own Configure section, apart from the detector's params.
_SURROGATE_KEYS = frozenset(p.key for p in _SURROGATE_PARAMS)

# Shared by COPOD and ECOD.
_N_JOBS_DESC = (
    "Has no effect on the mathematical course of the algorithm, only purely "
    "on the model's training speed. If set to \"-1\", the model is trained "
    "using all processor cores. Positive values specify the number of cores "
    "used when training the model [43]."
)


REGISTRY: dict[str, AlgorithmEntry] = {
    "iforest": AlgorithmEntry(
        key="iforest",
        display_name="Isolation Forest",
        run=_iforest.run,
        limitations=(
            "Beyond the training range the score stops growing: the trees can't isolate a value any further than the most extreme ones they saw, so a far-out value is flagged only as strongly as those.",
            'Splits one feature at a time, so it is weak at unusual combinations of features that are each normal on their own.',
        ),
        params=[
            _CONTAMINATION,
            Param(
                "N_ESTIMATORS", "Number of trees", "int", 100,
                "iForest differs mainly in that it does not look for normal "
                "data but, on the contrary, searches for anomalous data by "
                "means of splitting. The splitting forms a forest, and this "
                "parameter determines the size of that forest. Increasing "
                "the number of trees beyond a certain threshold runs into "
                "the law of diminishing returns and needlessly increases "
                "computational cost. A suitable number can be determined "
                "through empirical testing [42].",
                min=1, max=1000,
            ),
            Param(
                "MAX_SAMPLES", "Max samples per tree", "choice", "auto",
                "The individual trees of the algorithm are not trained on "
                "the whole dataset, but only on a small, randomly selected "
                "subset of it, whose default size (often under the parameter "
                "\"auto\") is 256 data points. In practice this value is "
                "considered optimal for most datasets, because deliberately "
                "shrinking the sample effectively solves two common "
                "problems. It prevents normal data points from 'swamping' "
                "the outliers, and it prevents clusters of anomalies from "
                "'masking' each other, which would otherwise make their "
                "isolation more difficult. [35].",
                choices=["auto", 64, 128, 256, 512, 1024],
            ),
            Param(
                "MAX_FEATURES", "Max features fraction", "float", 1.0,
                "Fraction of features considered per tree (1.0 = all).",
                min=0.1, max=1.0,
            ),
            Param(
                "BOOTSTRAP", "Bootstrap sampling", "bool", False,
                "Whether per-tree subsampling is drawn with replacement "
                "(bootstrap).",
            ),
            _TEST_SIZE,
            *_SURROGATE_PARAMS,
        ],
    ),
    "cblof": AlgorithmEntry(
        key="cblof",
        display_name="CBLOF",
        run=_cblof.run,
        limitations=(
            "Some combinations of number of clusters, alpha and beta can't split the clusters into large and small ones; training then stops with an error - change one of them.",
            'The score jumps at cluster boundaries (distance to the nearest large cluster), which the small exported network can only approximate.',
        ),
        params=[
            _CONTAMINATION,
            Param(
                "N_CLUSTERS", "Number of clusters", "int", 8,
                "In the background it uses KNN to create clusters. It may "
                "seem that this is actually just another clustering "
                "algorithm, which are computationally expensive, but this "
                "is only the first step. CBLOF assumes that normal data "
                "belong to large clusters (alpha) and anomalous data to "
                "small clusters (beta) [26].",
                min=2, max=50,
            ),
            Param(
                "ALPHA", "Alpha (large-cluster coverage)", "float", 0.9,
                "Represents the percentage of data that the alpha cluster "
                "must contain. In principle, after dividing the data into "
                "clusters, CBLOF sorts them in descending order. It then "
                "sums up the contents of the clusters, and if the next "
                "cluster in line exceeds the assigned percentage during the "
                "summing, all clusters together with the one that exceeded "
                "it first are labeled as beta clusters; in other words, it "
                "tries to find the most optimal split satisfying the ALPHA "
                "and BETA parameters [40].",
                min=0.5, max=0.99,
            ),
            Param(
                "BETA", "Beta (cluster size ratio)", "float", 5,
                "Is a ratio threshold separating large clusters from small "
                "ones relative to their size; in other words, all clusters "
                "that become part of the large alpha cluster must be x "
                "times larger than the smallest cluster [40].",
                min=1, max=20,
            ),
            Param(
                "USE_WEIGHTS", "Weight by cluster size", "bool", False,
                "If set to \"true\", the cluster size is used when "
                "calculating the weights. A situation may arise where, with "
                "high variability in sizes and densities, including it in "
                "the anomaly score calculation can distort the results. For "
                "that reason it is recommended in the PyOD library to have "
                "this parameter set to \"false\", in which case only the "
                "distance to the centroid of the nearest alpha cluster "
                "plays a role in the anomaly score calculation [40] [35].",
            ),
            _TEST_SIZE,
            *_SURROGATE_PARAMS,
        ],
    ),
    "cof": AlgorithmEntry(
        key="cof",
        display_name="COF",
        run=_cof.run,
        limitations=(
            'Local: flags values that are sparse compared with their neighbors, not values that are rare overall - a smoothly thinning tail (e.g. the hottest days) mostly stays normal.',
            'The compute strategy only changes how the training scores are computed (speed vs. memory); new values are always scored against the training set.',
        ),
        checks=notices.cof,
        params=[
            _CONTAMINATION,
            Param(
                "N_NEIGHBORS", "Number of neighbors", "int", 20,
                "Unlike LOF, COF relies on chaining values instead of "
                "classic rigid hyperspheres. COF compares the average "
                "chaining distance of the examined point and its k "
                "neighbors with the average chaining distances that these "
                "neighbors have within their own neighborhoods [35]. If the "
                "parameter is too low, the model becomes too sensitive to "
                "noise, and a point that would normally be part of the "
                "chain is labeled as an anomaly. If the parameter is too "
                "high relative to the size of the dataset, the algorithm "
                "stops focusing on local density and will pull points that "
                "are too distant into the chain [41].",
                min=1, max=100,
            ),
            Param(
                "METHOD", "Compute strategy", "choice", "memory",
                "Very briefly, it has two options. The first is \"fast\", "
                "which computes the distances all at once using a pairwise "
                "distance matrix. It is a faster and more memory-intensive "
                "approach. The second is \"memory\", which computes the "
                "distances incrementally, and that makes recomputation "
                "necessary. It follows that it is less demanding on memory "
                "and performance [42].",
                choices=["fast", "memory"],
            ),
            _TEST_SIZE,
            *_SURROGATE_PARAMS,
        ],
    ),
    "copod": AlgorithmEntry(
        key="copod",
        display_name="COPOD",
        run=_copod.run,
        limitations=(
            "Scores every feature's tails independently: it flags unusual values of single features, not unusual combinations of individually normal values.",
            "Picks the tail by each feature's skewness: on a skewed feature the short tail is scored more leniently, so e.g. unusually low values of a right-skewed feature score lower than equally rare high ones.",
        ),
        checks=notices.parallel,
        params=[
            _CONTAMINATION,
            Param(
                "N_JOBS", "Parallel workers", "int", 1,
                _N_JOBS_DESC,
                min=1, max=16,
            ),
            _TEST_SIZE,
            *_SURROGATE_PARAMS,
        ],
    ),
    "ecod": AlgorithmEntry(
        key="ecod",
        display_name="ECOD",
        run=_ecod.run,
        limitations=(
            "Scores every feature's tails independently: it flags unusual values of single features, not unusual combinations of individually normal values.",
        ),
        checks=notices.parallel,
        params=[
            _CONTAMINATION,
            Param(
                "N_JOBS", "Parallel workers", "int", 1,
                _N_JOBS_DESC,
                min=1, max=16,
            ),
            _TEST_SIZE,
            *_SURROGATE_PARAMS,
        ],
    ),
    "knn": AlgorithmEntry(
        key="knn",
        display_name="KNN",
        run=_knn.run,
        limitations=(
            'Global distance to the k-th nearest training value: a group of more than k similar anomalies looks normal, so k should exceed the size of such groups.',
        ),
        checks=notices.knn,
        params=[
            _CONTAMINATION,
            Param(
                "N_NEIGHBORS", "Number of neighbors (k)", "int", 5,
                "The neighbor whose distance defines the outlier score; "
                "small k is sensitive to local sparsity, large k smooths "
                "it.",
                min=1, max=100,
            ),
            Param(
                "METHOD", "Score aggregation", "choice", "largest",
                "How the k neighbor distances are aggregated into a "
                "single score.",
                choices=["largest", "mean", "median"],
            ),
            _TEST_SIZE,
            *_SURROGATE_PARAMS,
        ],
    ),
    "loda": AlgorithmEntry(
        key="loda",
        display_name="LODA",
        run=_loda.run,
        limitations=(
            'Values beyond the training range get the density of the outermost histogram bin, so they are flagged no more strongly than the most extreme training values - on the denser side of the data possibly not at all.',
            'Built from random projections: few random cuts make the score noisy.',
        ),
        params=[
            _CONTAMINATION,
            Param(
                "N_BINS", "Histogram bins", "int", 10,
                "Histogram bins per random projection; bin width sets the "
                "density-estimate bias/variance tradeoff (few bins = "
                "biased, many = noisy).",
                min=2, max=100,
            ),
            Param(
                "N_RANDOM_CUTS", "Random projections", "int", 100,
                "Number of random 1-D projections whose histogram log-"
                "densities are averaged; more cuts reduce the variance of "
                "the ensemble anomaly score.",
                min=10, max=1000,
            ),
            _TEST_SIZE,
            *_SURROGATE_PARAMS,
        ],
    ),
    "lof": AlgorithmEntry(
        key="lof",
        display_name="LOF",
        run=_lof.run,
        limitations=(
            'Local: flags values that are sparse compared with their neighbors, not values that are rare overall - a smoothly thinning tail (e.g. the hottest days) mostly stays normal.',
            'On rounded measurements with many repeated values, the number of neighbors should exceed how often a value repeats; otherwise the scores are mostly noise and the exported model agrees less with the detector.',
        ),
        checks=notices.lof,
        params=[
            _CONTAMINATION,
            Param(
                "N_NEIGHBORS", "Number of neighbors (k)", "int", 20,
                "k for the local reachability-density estimate; the score "
                "compares a point's density to that of its k neighbors, "
                "so larger k smooths the local density.",
                min=1, max=100,
            ),
            _TEST_SIZE,
            *_SURROGATE_PARAMS,
        ],
    ),
    "ocsvm": AlgorithmEntry(
        key="ocsvm",
        display_name="OCSVM",
        run=_ocsvm.run,
        limitations=(
            'Only the RBF kernel encloses the normal data; the linear, polynomial and sigmoid kernels separate a half-space, so values on one side of the data are never flagged.',
            'Training time grows roughly with the square of the number of rows.',
        ),
        checks=notices.ocsvm,
        params=[
            _CONTAMINATION,
            Param(
                "KERNEL", "Kernel", "choice", "rbf",
                "Kernel mapping inputs into a higher-dimensional space "
                "where a hyperplane can separate the dense \"normal\" "
                "region from the rest.",
                choices=["linear", "poly", "rbf", "sigmoid"],
            ),
            Param(
                "NU", "Nu", "float", 0.5,
                "Upper bound on the training-outlier fraction and lower "
                "bound on the support-vector fraction; directly controls "
                "how tight the decision boundary is.",
                min=0.01, max=1.0,
            ),
            Param(
                "GAMMA", "Gamma", "choice", "auto",
                "RBF bandwidth = radius of influence of each sample "
                "(\"auto\" = 1/n_features); small gamma = smooth "
                "boundary, large gamma = tight around points.",
                choices=["auto", "scale"],
            ),
            Param(
                "DEGREE", "Polynomial degree", "int", 3,
                "Polynomial-kernel degree; only used when kernel = "
                "\"poly\".",
                min=1, max=10,
            ),
            _TEST_SIZE,
            *_SURROGATE_PARAMS,
        ],
    ),
    "pca": AlgorithmEntry(
        key="pca",
        display_name="PCA",
        run=_pca.run,
        limitations=(
            'Its strength is correlated features: with a single feature it reduces to a distance from the mean.',
        ),
        checks=notices.pca,
        params=[
            _CONTAMINATION,
            Param(
                "N_COMPONENTS", "Components kept", "choice", None,
                "Number of principal components (eigenvectors) spanning "
                "the \"normal\" subspace; \"all\" keeps every component, "
                "so the score becomes the full Mahalanobis distance.",
                choices=[None, 1, 2],
            ),
            Param(
                "WHITEN", "Whiten", "bool", False,
                "If True, rescale each component to unit variance before "
                "scoring.",
            ),
            Param(
                "WEIGHTED", "Weight by eigenvalue", "bool", True,
                "Weight each component's squared deviation by its "
                "eigenvalue (1/variance), i.e. a Mahalanobis-style score "
                "so low-variance directions count more.",
            ),
            Param(
                "STANDARDIZATION", "Standardize features first", "bool",
                True,
                "z-score features first so components are not dominated "
                "by high-variance features (makes the score scale-"
                "invariant).",
            ),
            _TEST_SIZE,
            *_SURROGATE_PARAMS,
        ],
    ),
    "sod": AlgorithmEntry(
        key="sod",
        display_name="SOD",
        run=_sod.run,
        limitations=(
            'Needs at least two feature columns: with one there is no subspace to select and every score is 0. Only feature subsets of two or more columns are trained, so the device must send at least two features.',
            'A value far from the data can still score 0 when no axis of its reference set has a clearly lower variance than the others.',
        ),
        checks=notices.sod,
        min_columns=_sod.TRAINING.min_pattern_columns,
        params=[
            _CONTAMINATION,
            Param(
                "N_NEIGHBORS", "Number of neighbors", "int", 20,
                "Neighborhood size used to build each point's shared-"
                "nearest-neighbor set.",
                min=2, max=100,
            ),
            Param(
                "REF_SET", "Reference set size", "int", 10,
                "Number of shared nearest neighbors forming the reference "
                "set that defines the local subspace (must be smaller "
                "than the number of neighbors).",
                min=1, max=99,
            ),
            Param(
                "ALPHA", "Alpha (variance retained)", "float", 0.8,
                "Fraction of variance retained when selecting the "
                "relevant subspace; low-variance axes are the directions "
                "the outlierness is measured along.",
                min=0.1, max=0.99,
            ),
            _TEST_SIZE,
            *_SURROGATE_PARAMS,
        ],
    ),
    "deepsvdd": AlgorithmEntry(
        key="deepsvdd",
        display_name="DeepSVDD",
        run=_deepsvdd.run,
        limitations=(
            'The autoencoder variant needs at least as many feature columns as the smallest hidden layer has neurons (32).',
            'Trains one PyTorch network per feature subset, so it is slow with several features.',
        ),
        checks=lambda params, data: notices.deepsvdd(params, data, _deepsvdd.DEFAULT_PARAMS["HIDDEN_NEURONS"]),
        params=[
            _CONTAMINATION,
            Param(
                "USE_AE", "Use autoencoder variant", "bool", False,
                "If True, use an autoencoder variant; else a plain "
                "encoder onto the hypersphere.",
            ),
            Param(
                "HIDDEN_ACTIVATION", "Hidden activation", "choice", "relu",
                "Hidden-layer nonlinearity, enabling a non-linear "
                "\"normal\" region boundary.",
                choices=["relu", "tanh", "sigmoid"],
            ),
            Param(
                "OUTPUT_ACTIVATION", "Output activation", "choice",
                "sigmoid",
                "Activation of the representation (output) layer.",
                choices=["sigmoid", "linear", "tanh"],
            ),
            Param(
                "OPTIMIZER", "Optimizer", "choice", "adam",
                "Optimizer minimizing the mean squared distance to the "
                "hypersphere center.",
                choices=["adam", "sgd", "rmsprop"],
            ),
            Param(
                "DSVDD_EPOCHS", "Detector training epochs", "int", 100,
                "Passes over the data when training the detector itself "
                "(separate from the surrogate network below); too few "
                "underfits, too many overfits.",
                min=1, max=1000,
            ),
            Param(
                "DSVDD_BATCH_SIZE", "Detector batch size", "int", 32,
                "Samples per gradient estimate for the detector network; "
                "larger batches give lower-variance updates.",
                min=1, max=1024,
            ),
            Param(
                "DROPOUT_RATE", "Dropout rate", "float", 0.2,
                "Dropout probability: randomly zeros units during "
                "training to reduce overfitting.",
                min=0.0, max=0.9,
            ),
            Param(
                "L2_REGULARIZER", "L2 weight decay", "float", 0.5e-6,
                "L2 weight-decay strength: shrinks weights toward zero "
                "(variance reduction; also discourages the trivial all-"
                "zero collapse of the representation).",
                min=0.0, max=0.01,
            ),
            _TEST_SIZE,
            *_SURROGATE_PARAMS,
        ],
    ),
}
